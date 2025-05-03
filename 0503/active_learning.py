import random
import math
import numpy as np
import torch
from tqdm import tqdm
import os
import yaml
import logging
from omegaconf import OmegaConf

from model_utils import predict

# 添加RL相关导入
try:
    from src.rl.base_environment import BaseEnvironment
    from src.rl.tkg_environment import TKGEnvironment
    from src.rl.agents.dqn_agent import DQNAgent
    # 可能还需要其他RL相关导入
    RL_AVAILABLE = True
except ImportError as e:
    import traceback
    print(f"警告: 强化学习模块导入失败: {e}")
    print(f"错误详情: {traceback.format_exc()}")
    print("RLStrategy将不可用")
    RL_AVAILABLE = False


def get_current_time_samples(test_data, timestamp=None, current_sample_index=None):
    """获取特定时间戳的所有样本，排除当前正在预测的样本
    
    Args:
        test_data: 测试数据列表（或按时间戳组织的字典）
        timestamp: 目标时间戳（如果test_data是字典，可以直接使用键）
        current_sample_index: 当前正在预测的样本索引（可选），如果提供则排除此样本
    
    Returns:
        当前时间戳的样本列表和对应的索引列表
    """
    # 适配不同的数据结构
    if timestamp is not None and isinstance(test_data, dict) and timestamp in test_data:
        # 如果test_data是按时间戳组织的字典
        samples_for_timestamp = test_data[timestamp]
        current_samples = []
        current_indices = []
        
        for i, item in enumerate(samples_for_timestamp):
            if i != current_sample_index:
                current_samples.append(item)
                current_indices.append(i)
                
        return current_samples, current_indices
    else:
        # 原始实现，遍历列表查找指定时间戳的样本
        current_samples = []
        current_indices = []
        
        for i, (sample, direction) in enumerate(test_data):
            if (timestamp is None or sample[3] == timestamp) and i != current_sample_index:
                # 返回原始样本（包含标签信息）
                current_samples.append((sample, direction))
                current_indices.append(i)
        
        return current_samples, current_indices


def integrate_active_samples(prompt, active_samples, args):
    """将主动学习选择的样本整合到提示中，包含真实标签（专家标注）"""
    if not active_samples:
        return prompt
        
    if args.active_integration == "direct":
        # 直接添加带标签的样本到提示
        for sample, direction in active_samples:
            entity, relation, targets, time = sample
            target = targets[0] if targets else "?"  # 使用第一个目标实体作为标签
            if not args.no_time:
                prompt += f"{time}:"
            if args.label:
                # 使用带标签格式
                prompt += f"[{entity},{relation},{targets.index(target) if target != '?' else '?'}. {target}]\n"
            else:
                # 使用实体格式
                prompt += f"[{entity},{relation},{target}]\n"
    elif args.active_integration == "labeled":
        # 添加标记为"专家标注"的样本
        prompt += "\n专家标注的当前事件:\n"
        for sample, direction in active_samples:
            entity, relation, targets, time = sample
            target = targets[0] if targets else "?"  # 使用第一个目标实体作为标签
            if not args.no_time:
                prompt += f"{time}:"
            if args.label:
                # 使用带标签格式
                prompt += f"[{entity},{relation},{targets.index(target) if target != '?' else '?'}. {target}]\n"
            else:
                # 使用实体格式
                prompt += f"[{entity},{relation},{target}]\n"
    
    return prompt


class BaseStrategy:
    """主动学习策略基类"""
    def __init__(self, name):
        self.name = name
    
    def select_samples(self, candidates, num_samples, model, tokenizer, history, args):
        """选择样本的基本方法，子类应重写此方法
        
        Args:
            candidates: 候选样本列表
            num_samples: 要选择的样本数量
            model: 使用的模型
            tokenizer: 使用的分词器
            history: 搜索空间/历史字典
            args: 参数
            
        Returns:
            (选中的样本索引列表, 未选中的样本索引列表)
        """
        raise NotImplementedError("子类必须实现select_samples方法")


class RandomStrategy(BaseStrategy):
    """随机选择策略"""
    def __init__(self):
        super().__init__("random")
    
    def select_samples(self, candidates, num_samples, model, tokenizer, history, args):
        """随机选择指定数量的样本"""
        num_to_select = min(num_samples, len(candidates))
        if num_to_select == 0:
            return [], list(range(len(candidates)))
        
        # 随机选择索引
        all_indices = list(range(len(candidates)))
        selected_indices = random.sample(all_indices, num_to_select)
        
        # 计算未选中的索引
        unselected_indices = [i for i in all_indices if i not in selected_indices]
        
        return selected_indices, unselected_indices




class RLStrategy(BaseStrategy):
    """使用强化学习选择样本的策略"""
    _config = None # 类变量缓存配置
    _agent = None # 类变量缓存代理
    _environment = None # 类变量缓存环境实例 (如果希望跨查询共享)
    # 或者使用实例变量，如果每个策略实例管理自己的Agent/Env

    def __init__(self, config_path=None):
        super().__init__("rl")
        if not RL_AVAILABLE:
            raise ImportError("强化学习模块未成功导入，无法使用RLStrategy")

        self.config_path = config_path
        self.logger = logging.getLogger("RLStrategy")
        self.config = None # 新增实例变量来存储配置

        # 立即加载配置并存储到实例变量
        if config_path:
            # 使用类方法加载，但结果存到实例变量 self.config
            # 注意：这里仍然会更新类缓存 cls._config，这可能不是最优设计，但暂时保留
            self.config = self.__class__._load_config(direct_path=config_path)
            if not self.config: # 如果加载失败，创建一个空的
                 self.logger.warning(f"在 __init__ 中加载配置 {config_path} 失败，使用空配置")
                 self.config = OmegaConf.create({})
        else:
             self.logger.warning("未提供 config_path，使用空配置")
             self.config = OmegaConf.create({})

        # 实例变量，用于存储单个查询的上下文和环境/代理
        self.environment = None
        self.agent = None
        self.current_historical_samples = []
        self.global_entity_ids = None
        self.global_relation_ids = None

        self.bert_model = None
        self.bert_tokenizer = None
        self._load_bert_model_if_needed() # 调用时它将使用 self.config
        
    def _load_bert_model_if_needed(self):
        print("DEBUG: Entering _load_bert_model_if_needed")
        # --- 修改这里：直接使用 self.config ---
        # config = self._load_config(direct_path=self.config_path) # 不再调用 _load_config
        if not self.config:
             print("DEBUG: self.config is None or empty in _load_bert_model_if_needed. Skipping.")
             self.bert_model = None
             self.bert_tokenizer = None
             return

        # 从 env_kwargs 中获取 feature_config
        env_kwargs = self.config.get("env_kwargs", {})
        feature_config = env_kwargs.get("feature_config", {})
        # --- 修改结束 ---

        print(f"DEBUG: Using self.config. feature_config keys: {list(feature_config.keys())}")
        if feature_config.get("use_plm_embeddings", False):
            encoder_path = feature_config.get("plm_encoder_path", "bert-base-uncased")
            print(f"DEBUG: use_plm_embeddings=True. encoder_path={encoder_path}")
            try:
                from transformers import AutoModel, AutoTokenizer
                from src.rl.plm_encoder import PLMEncoder

                self.logger.info(f"尝试加载 Hugging Face 模型: {encoder_path}")
                # --- 添加打印 ---
                print(f"DEBUG: Loading tokenizer: {encoder_path}")
                # --- 添加结束 ---
                hf_tokenizer = AutoTokenizer.from_pretrained(encoder_path)
                # --- 添加打印 ---
                print(f"DEBUG: Tokenizer loaded: {hf_tokenizer is not None}")
                print(f"DEBUG: Loading model: {encoder_path}")
                # --- 添加结束 ---
                hf_model = AutoModel.from_pretrained(encoder_path)
                # --- 添加打印 ---
                print(f"DEBUG: Model loaded: {hf_model is not None}")
                # --- 添加结束 ---

                # 将模型移动到合适的设备
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                
                # 如果RLStrategy有args属性，尝试使用它的gpu设置
                if hasattr(self, 'args') and hasattr(self.args, 'gpu'):
                    if self.args.gpu >= 0:
                        device = torch.device(f"cuda:{self.args.gpu}")
                    elif self.args.gpu == -2 and torch.cuda.is_available():
                        # 使用当前PyTorch默认CUDA设备或尝试自动选择
                        try:
                            device = torch.device(f"cuda:{torch.cuda.current_device()}")
                            self.logger.info(f"使用当前PyTorch默认CUDA设备: {device}")
                        except Exception:
                            # 如果无法获取当前设备，使用cuda:0
                            device = torch.device("cuda:0")
                            self.logger.warning(f"无法获取PyTorch默认CUDA设备，使用{device}")

                try:
                    hf_model = hf_model.to(device)
                    hf_model.eval()
                    self.logger.info(f"Hugging Face 模型已加载到设备: {device}")
                    # --- 添加打印 ---
                    print(f"DEBUG: Model moved to device: {device}")
                    # --- 添加结束 ---
                except Exception as e:
                    self.logger.error(f"无法将BERT模型移动到设备 {device}: {e}")
                    # --- 添加打印 ---
                    print(f"DEBUG: ERROR moving model to device: {e}")
                    # --- 添加结束 ---
                    self.bert_model = None
                    self.bert_tokenizer = None
                    return

                # --- 添加打印 ---
                print(f"DEBUG: Initializing PLMEncoder with model: {hf_model is not None}, tokenizer: {hf_tokenizer is not None}")
                # --- 添加结束 ---
                self.bert_model = PLMEncoder(hf_model, hf_tokenizer, model_type="encoder")
                self.bert_tokenizer = hf_tokenizer
                # --- 添加打印 ---
                print(f"DEBUG: PLMEncoder initialized. self.bert_model is None: {self.bert_model is None}. self.bert_tokenizer is None: {self.bert_tokenizer is None}")
                # --- 添加结束 ---

                self.logger.info(f"RL策略已加载特征提取器: {encoder_path}")
            except Exception as e:
                self.logger.error(f"加载PLM编码器 {encoder_path} 失败: {e}", exc_info=True)
                # --- 添加打印 ---
                print(f"DEBUG: ERROR during model/tokenizer loading or PLMEncoder init: {e}")
                # --- 添加结束 ---
                self.bert_model = None
                self.bert_tokenizer = None
        else:
            # --- 添加打印 ---
            print("DEBUG: use_plm_embeddings is False or not found in feature_config.")
            # --- 添加结束 ---
            self.bert_model = None
            self.bert_tokenizer = None
        # --- 添加打印 ---
        print("DEBUG: Exiting _load_bert_model_if_needed")
        # --- 添加结束 ---

    @classmethod
    def _load_config(cls, args=None, direct_path=None):
        # 优先使用直接传入的路径，然后是args中的路径，最后是类缓存
        config_path = direct_path or (getattr(args, 'rl_config', None) if args else None)
        if config_path and (cls._config is None or cls._config.get('_source_path') != config_path):
            try:
                # 添加路径处理和调试信息
                print(f"尝试加载RL配置文件: {config_path}")
                
                # 检查文件是否存在
                import os
                if not os.path.exists(config_path):
                    print(f"错误: 配置文件不存在: {config_path}")
                    if not os.path.isabs(config_path):
                        # 尝试相对于项目根目录解析
                        current_dir = os.path.dirname(os.path.abspath(__file__))
                        project_root = os.path.dirname(current_dir)
                        abs_path = os.path.join(project_root, config_path)
                        print(f"尝试相对路径: {abs_path}")
                        if os.path.exists(abs_path):
                            config_path = abs_path
                            print(f"找到配置文件: {config_path}")
                
                cls._config = OmegaConf.load(config_path)
                cls._config['_source_path'] = config_path # 记录来源
                
                # 处理配置文件键名差异（agent_kwargs -> agent_config）
                if 'agent_kwargs' in cls._config and 'agent_config' not in cls._config:
                    print("将配置文件中的 'agent_kwargs' 映射到 'agent_config'")
                    cls._config['agent_config'] = cls._config['agent_kwargs']
                
                # 处理agent_type键名差异
                if 'agent' in cls._config and 'agent_type' not in cls._config:
                    print("将配置文件中的 'agent' 映射到 'agent_type'")
                    cls._config['agent_type'] = cls._config['agent']
                
                logging.info(f"从 {config_path} 加载RL配置")
                print(f"成功加载RL配置，配置内容: {cls._config}")
            except Exception as e:
                logging.error(f"加载RL配置文件 {config_path} 失败: {e}")
                print(f"加载RL配置文件错误: {e}")
                import traceback
                traceback.print_exc()
                cls._config = OmegaConf.create({}) # 返回空配置
        elif cls._config is None:
            logging.warning("未提供RL配置文件路径，使用空配置")
            cls._config = OmegaConf.create({})
        return cls._config

    def _initialize_agent(self, environment, args, **extra_params):
        # --- 使用 self.config 获取配置 ---
        if not self.config:
             self.logger.error("无法在 _initialize_agent 中获取配置!")
             self.agent = None # 明确设置 agent 为 None
             return # 直接返回，让调用处处理 self.agent is None 的情况

        agent_type = self.config.get("agent_type", "dqn").lower()
        agent_config = self.config.get("agent_config", {}) # 获取 agent_config
        env_kwargs = self.config.get("env_kwargs", {}) # 获取 env_kwargs

        # 修改设备选择逻辑，使用args.gpu而非args.device
        # 优先使用args.gpu参数，兼容已有代码
        selected_gpu = None
        if hasattr(args, 'gpu'):
            if args.gpu >= 0:
                device = torch.device(f"cuda:{args.gpu}")
                selected_gpu = args.gpu
            # 如果是自动选择GPU的情况(-2)
            elif args.gpu == -2 and torch.cuda.is_available():
                import subprocess
                import re
                try:
                    output = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'])
                    memory_free = [int(x) for x in output.decode('utf-8').strip().split('\n')]
                    best_gpu = memory_free.index(max(memory_free))
                    device = torch.device(f"cuda:{best_gpu}")
                    selected_gpu = best_gpu
                    self.logger.info(f"自动选择GPU: {best_gpu} (可用内存: {max(memory_free)} MiB)")
                except Exception as e:
                    self.logger.warning(f"自动选择GPU失败: {e}，使用默认设备")
                    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                    selected_gpu = 0 if torch.cuda.is_available() else None
            else:
                device = torch.device("cpu")
                selected_gpu = None
        else:
            device = torch.device("cpu")
            selected_gpu = None
            self.logger.info("未指定GPU参数，使用CPU")

        self.logger.info(f"初始化RL Agent: 类型={agent_type}, 设备={device}")
        self.logger.debug(f"Agent配置: {agent_config}")

        # --- 创建 Agent 实例 (局部变量 agent) ---
        agent = None
        if agent_type == "dqn":
            from src.rl.agents.dqn_agent import DQNAgent

            # --- 修改 output_dir 的确定方式 ---
            # 优先使用命令行参数 args.output_dir，如果存在且非空
            if hasattr(args, 'output_dir') and args.output_dir:
                output_dir = args.output_dir
                self.logger.info(f"使用命令行的 output_dir: {output_dir}")
            else:
                # 否则，回退到使用 YAML 配置中的值
                output_dir = self.config.get("output_dir", "./rl_outputs") # 提供一个默认值以防万一
                self.logger.info(f"使用配置文件中的 output_dir: {output_dir}")
            # --- 修改结束 ---

            # 后续 dqn_params 使用修正后的 output_dir
            dqn_params = {'output_dir': output_dir}
            # 处理 agent_config
            mapped_config = {}
            param_mapping = {'epsilon_start': 'eps_start', 'epsilon_end': 'eps_end', 'epsilon_decay': 'decay_steps'}
            agent_cfg_dict = OmegaConf.to_container(self.config.get('agent_config', {}), resolve=True)
            for key, value in agent_cfg_dict.items():
                 if key == 'gamma':
                     self.logger.info(f"跳过配置文件中的 'gamma' 参数 ({value})")
                     continue
                 elif key in param_mapping:
                     mapped_config[param_mapping[key]] = value
                 elif key == 'network_params':
                     if isinstance(value, dict):
                          net_params_copy = value.copy()
                          if 'num_layers' in net_params_copy:
                              removed_layers = net_params_copy.pop('num_layers')
                              self.logger.info(f"从 network_params 中移除 'num_layers' 参数 ({removed_layers})")
                          mapped_config[key] = net_params_copy
                     else:
                          mapped_config[key] = {}
                 else:
                     mapped_config[key] = value
            if 'network_params' not in mapped_config:
                 mapped_config['network_params'] = {}

            dqn_params.update(mapped_config)
            dqn_params.update(extra_params)

            # 移除顶层的 gamma (如果之前没处理掉)
            if 'gamma' in dqn_params:
                 dqn_params.pop('gamma')

            # 确认 network_params 没有 num_layers
            if 'network_params' in dqn_params and 'num_layers' in dqn_params['network_params']:
                 self.logger.warning("发现 num_layers 仍然在 dqn_params['network_params'] 中，强制移除！")
                 dqn_params['network_params'].pop('num_layers')

            try:
                agent = DQNAgent(environment, **dqn_params)
            except TypeError as e:
                 # 捕获 TypeError 并打印传递的参数，帮助调试
                 self.logger.error(f"初始化 DQNAgent 时发生 TypeError: {e}")
                 self.logger.error(f"传递的参数 dqn_params: {dqn_params}")
                 agent = None # 初始化失败
            except Exception as e:
                 self.logger.error(f"初始化 DQNAgent 时发生其他错误: {e}", exc_info=True)
                 agent = None # 初始化失败

        elif agent_type == "random":
             from src.rl.agents.random_agent import RandomAgent
             # 确保 environment.action_dim 可用
             try:
                 action_dim = environment.action_dim
             except Exception as e:
                 self.logger.warning(f"无法获取 environment.action_dim: {e}, 使用默认值0")
                 action_dim = 0
             agent = RandomAgent(action_feature_dim=action_dim)
        else:
            self.logger.error(f"未知的Agent类型: {agent_type}，将使用随机Agent")
            from src.rl.agents.random_agent import RandomAgent
            try:
                 action_dim = environment.action_dim
            except Exception as e:
                 self.logger.warning(f"无法获取 environment.action_dim: {e}, 使用默认值0")
                 action_dim = 0
            agent = RandomAgent(action_feature_dim=action_dim)

        # --- 在加载模型前，将创建的 agent 赋给 self.agent ---
        self.agent = agent
        if self.agent is None:
             self.logger.error("未能成功创建 Agent 实例!")
             return # 如果 agent 创建失败则返回

        self.logger.info(f"RL Agent instance created and assigned to self.agent: {type(self.agent)}")
        # --- 修改结束 ---

        # --- 加载离线模型逻辑 (现在 self.agent 已被赋值) ---
        offline_model_path = None
        if hasattr(args, 'offline_model_checkpoint_path') and args.offline_model_checkpoint_path:
            offline_model_path = args.offline_model_checkpoint_path
        elif self.config.get("offline_model_checkpoint_path"): # 从 self.config 获取
            offline_model_path = self.config.get("offline_model_checkpoint_path")

        if offline_model_path:
            self.logger.info(f"检测到离线模型路径: {offline_model_path}，尝试加载...")
            # 检查实例变量 self.agent 是否支持加载模型
            if hasattr(self.agent, 'policy_net') and hasattr(self, 'load_model'):
                try:
                    # 调用 RLStrategy 的 load_model 方法，它会处理具体的加载逻辑
                    if self.load_model(offline_model_path): # load_model 内部会检查 self.agent 是否存在
                        self.logger.info(f"成功加载离线模型: {offline_model_path} into self.agent")
                    else:
                        # self.load_model 内部会打印错误日志
                        pass
                except Exception as e:
                    self.logger.error(f"加载离线模型 ({offline_model_path}) 时出错: {e}", exc_info=True)
            elif not hasattr(self, 'load_model'):
                 self.logger.error("RLStrategy 缺少 load_model 方法！")
            else:
                self.logger.warning(f"当前代理类型 {type(self.agent).__name__} 可能不支持加载策略网络 (policy_net)，无法加载离线模型")

        # --- 这个方法不再需要返回 agent ---
        # return agent

    def select_samples(self, candidates, num_samples, model, tokenizer, history, args):
        """
        使用RL代理选择样本。
        Args:
            candidates: 当前时间戳下可供选择的样本列表 [(sample_tuple, direction, targets), ...]
            num_samples: 要选择的样本数量
            model: 主LLM模型 (传递给环境用于评估)
            tokenizer: 主LLM的分词器
            history: 历史搜索空间字典
            args: 运行参数

        Returns:
            (选择的样本索引列表, 未选择的样本索引列表)
        """
        if not RL_AVAILABLE or not candidates:
            # 如果RL不可用或没有可用样本，使用随机策略
            self.logger.warning("RL不可用或无可选样本，回退到随机策略")
            return RandomStrategy().select_samples(candidates, num_samples, model, tokenizer, history, args)
        
        self.logger.info(f"RL策略选择样本: 候选数量={len(candidates)}, 需要选择={num_samples}")
        
        try:
            # 如有必要，初始化全局ID映射
            if self.global_entity_ids is None or self.global_relation_ids is None:
                try:
                    # 加载全局ID映射
                    if hasattr(args, 'use_global_ids') and args.use_global_ids:
                        from utils import load_global_id_mappings
                        
                        self.logger.info("加载全局ID映射...")
                        entity_ids, relation_ids = load_global_id_mappings()
                        
                        if entity_ids and relation_ids:
                            self.global_entity_ids = entity_ids
                            self.global_relation_ids = relation_ids
                            self.logger.info(f"加载全局ID映射: {len(entity_ids)}个实体, {len(relation_ids)}个关系")
                        else:
                            # 如果加载失败，尝试从当前样本创建
                            self._create_global_id_mappings(candidates)
                    else:
                        # 如果没有use_global_ids标志，使用老方法
                        from utils import load_all_entity_relation_mappings
                        self.logger.info("加载全局ID映射用于测试...")
                        entity_ids, relation_ids = load_all_entity_relation_mappings(args)
                        
                        if entity_ids and relation_ids:
                            self.global_entity_ids = entity_ids
                            self.global_relation_ids = relation_ids
                            self.logger.info(f"成功加载全局ID映射: {len(entity_ids)}个实体, {len(relation_ids)}个关系")
                        else:
                            # 如果加载失败，尝试从当前样本创建
                            self._create_global_id_mappings(candidates)
                except Exception as e:
                    self.logger.error(f"加载全局ID映射时出错: {e}，将使用当前样本创建映射")
                    self._create_global_id_mappings(candidates)

            # 初始化或更新环境
            if self.environment is None:
                self.logger.info("初始化RL环境实例...")
                # 确认这里传递的参数是正确的
                self.environment = TKGEnvironment(
                    interaction_data=candidates,
                    state_model=self.bert_model.model if self.bert_model else None,
                    state_tokenizer=self.bert_tokenizer,
                    reward_model=model,
                    reward_tokenizer=tokenizer,
                    train_history=history,
                    args=args,
                    # 使用 self.config 获取配置
                    state_repr=self.config.get("env_kwargs", {}).get("state_repr", ["query_features", "context_features"]),
                    max_steps=num_samples,
                    reward_scale=self.config.get("env_kwargs", {}).get("reward_scale", 1.0),
                    predef_entity_ids=self.global_entity_ids,
                    predef_relation_ids=self.global_relation_ids,
                    state_model_type="encoder",
                )
            else:
                # 更新现有环境
                self.environment.train_history = history
                self.environment.interaction_data = candidates # 更新候选样本
                self.environment.max_steps = num_samples
                self.environment.reset() # 重置环境以应用新的候选样本

            # --- 初始化或获取代理 (调整调用方式) ---
            if self.agent is None:
                self.logger.info("RL 代理尚未初始化，尝试初始化...")
                self._initialize_agent(self.environment, args) # 调用初始化，它会设置 self.agent
                # 检查初始化是否成功
                if self.agent is None:
                    self.logger.error("RL Agent 初始化失败! 回退到随机策略。")
                    return RandomStrategy().select_samples(candidates, num_samples, model, tokenizer, history, args)
                self.logger.info("RL 代理初始化完成。")
            # --- 修改结束 ---

            # --- 运行RL选择过程 ---
            self.logger.info(f"RL策略开始选择样本，最多选择 {num_samples} 个")
            selected_indices = [] # 存储选择的 *原始* 候选索引
            steps_taken = 0
            state = self.environment.state # 获取初始状态
            # all_indices = list(range(len(candidates))) # 这个可能不需要了

            while steps_taken < num_samples:
                # 获取当前可用的样本和它们的索引 (相对于原始 candidates 列表)
                current_available_samples, current_available_indices, current_timestamp = self.environment.get_available_samples()

                if not current_available_samples:
                    self.logger.warning("当前时间步没有可用的候选样本，选择终止。")
                    break

                # 1. 获取当前状态 (可能已在循环外获取，或在 step 后更新)
                # state = self.environment.state # 确保获取最新状态

                # 2. 获取当前完整的动作空间信息 (特征、ID等)
                #    action_space() 应该只返回当前可用动作的信息
                action_space_tuple = self.environment.action_space()

                # 3. 检查动作空间是否有效
                if action_space_tuple is None or not isinstance(action_space_tuple, tuple) or len(action_space_tuple) != 3:
                     self.logger.warning(f"从环境获取的动作空间无效: {action_space_tuple}，选择终止。")
                     break
                action_features, entity_ids, relation_ids = action_space_tuple
                # 检查特征是否是 Tensor 且非空
                if not torch.is_tensor(action_features) or action_features.shape[0] == 0:
                     # 如果是 list，尝试转换 (如果 DQNAgent.choose_action 没有处理)
                     if isinstance(action_features, list) and len(action_features) > 0:
                          try:
                              action_features = torch.stack(action_features) # 假设列表里是 Tensor
                              self.logger.warning("环境返回了 list 类型的 action_features，已尝试 stack。")
                          except Exception as stack_err:
                              self.logger.error(f"尝试 stack action_features list 失败: {stack_err}, 选择终止。")
                              break
                     else:
                          self.logger.warning(f"动作空间特征无效或为空 (类型: {type(action_features)}), 选择终止。")
                          break
                
                # 确保动作特征数量与可用索引数量一致
                if action_features.shape[0] != len(current_available_indices):
                    self.logger.error(f"动作特征数量 ({action_features.shape[0]}) 与可用索引数量 ({len(current_available_indices)}) 不匹配！选择终止。")
                    break

                num_available_actions = action_features.shape[0]
                # 创建一个表示 action_features 中所有索引都可用的列表
                available_env_indices_for_agent = list(range(num_available_actions))

                # !! 修改：调用 agent.select_action !!
                try:
                     # 调用 select_action，它会处理 available_actions
                     # select_action 返回的是在 available_actions 列表中的索引 (即相对索引)
                     selected_index_in_available, _, _ = self.agent.select_action(
                         state,
                         action_features,
                         entity_ids,
                         relation_ids,
                         available_actions=available_env_indices_for_agent # 告诉 agent 这些都是可选的
                     )
                     # selected_index_in_available 就是我们需要的相对于 action_features 的索引
                     relative_action_idx = selected_index_in_available
                     self.logger.debug(f"Agent select_action 选择了相对索引: {relative_action_idx}")

                except Exception as select_err:
                     self.logger.error(f"调用 self.agent.select_action 时出错: {select_err}", exc_info=True)
                     break # 无法选择，终止

                # 5. 检查返回的相对索引是否有效
                if relative_action_idx is None or relative_action_idx < 0 or relative_action_idx >= action_features.shape[0]:
                    self.logger.warning(f"Agent 返回了无效的相对动作索引 {relative_action_idx}，将随机选择。")
                    # 从当前可用的原始索引中随机选一个
                    if not current_available_indices: break
                    original_candidate_idx = random.choice(current_available_indices)
                    self.logger.debug(f"随机回退选择了原始索引: {original_candidate_idx}")
                    # 需要找到这个随机选择的索引在当前 action_features 中的相对位置吗？
                    # 可能不需要，因为我们直接用 original_candidate_idx 去 step
                else:
                    # 6. 将相对索引映射回原始候选列表中的索引
                    try:
                         original_candidate_idx = current_available_indices[relative_action_idx]
                         self.logger.debug(f"相对索引 {relative_action_idx} 映射到原始索引: {original_candidate_idx}")
                    except IndexError:
                         self.logger.error(f"无法映射相对索引 {relative_action_idx} 到可用索引列表 (长度 {len(current_available_indices)})！")
                         break

                # 7. 检查选择的原始索引是否重复（理论上环境更新后不应重复）
                if original_candidate_idx in selected_indices:
                     self.logger.warning(f"尝试选择的原始索引 {original_candidate_idx} 已在选择列表中，跳过此步骤。")
                     # 这里可能表示环境状态或动作空间更新逻辑有问题
                     # 可以选择 break 或者 continue 尝试下一次
                     continue

                # 8. 记录选择的原始索引
                selected_indices.append(original_candidate_idx)
                steps_taken += 1

                # 9. 调用 environment.step() 更新环境状态
                try:
                    # 使用选择的 *原始* 索引来执行步骤
                    next_state_tuple, reward, terminal, _ = self.environment.step(original_candidate_idx)
                    # 更新状态供下一次循环使用
                    state = next_state_tuple[0] if isinstance(next_state_tuple, tuple) else next_state_tuple
                    self.logger.info(f"步骤 {steps_taken}/{num_samples}: 选择索引 {original_candidate_idx}, Reward: {reward:.4f}, Terminal: {terminal}")
                    if terminal:
                        self.logger.info("环境达到终止状态或达到最大步骤。")
                        break
                except Exception as step_err:
                    self.logger.error(f"调用 environment.step(原始索引: {original_candidate_idx}) 时出错: {step_err}", exc_info=True)
                    break # 无法继续模拟
            # --- 循环结束 ---

            # 计算未选中的索引 (相对于原始 candidates)
            all_original_indices = list(range(len(candidates)))
            unselected_indices = [i for i in all_original_indices if i not in selected_indices]

            self.logger.info(f"RL策略选择完成，共选择了 {len(selected_indices)} 个样本")
            return selected_indices, unselected_indices

        except Exception as e:
            self.logger.error(f"RL样本选择过程中发生意外错误: {e}", exc_info=True)
            # 回退到随机策略
            self.logger.warning("RL策略失败，回退到随机策略")
            return RandomStrategy().select_samples(candidates, num_samples, model, tokenizer, history, args)

    def load_model(self, model_path=None):
        """加载预训练的RL模型（例如从训练阶段保存的模型）"""
        # 如果没有传入路径，使用配置中指定的路径
        if model_path is None:
            config = self.config
            model_path = config.get("pretrained_agent_path", None)
            if not model_path:
                self.logger.warning("未指定预训练模型路径，无法加载模型")
                return False
        
        if not self.agent:
            self.logger.warning("无法加载模型: RL代理尚未初始化")
            return False
            
        try:
            self.logger.info(f"尝试从 {model_path} 加载RL模型")
            
            # 根据路径格式判断加载方式
            if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "model_400.ckpt")):
                # 如果是检查点目录并且存在model_400.ckpt文件，使用load_model_at_step方法
                self.logger.info(f"检测到检查点目录，尝试加载step 400的模型")
                self.agent.load_model_at_step(400)  # 假设使用步骤400的模型
            elif os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "best_model.ckpt")):
                # 如果是检查点目录并且存在best_model.ckpt文件，使用load_best_model方法
                self.logger.info(f"检测到检查点目录，尝试加载最佳模型")
                self.agent.load_best_model()
            elif os.path.isfile(model_path):
                # 如果是单个文件，直接加载
                self.logger.info(f"检测到单个模型文件，直接加载")
                self.agent.policy_net.load_state_dict(torch.load(model_path, weights_only=True))
                self.agent.policy_net = self.agent.policy_net.to(self.agent.device)
                self.agent.target_update()  # 更新目标网络
            else:
                self.logger.error(f"无法识别的模型路径格式: {model_path}")
                return False
                
            # 设置为评估模式
            self.agent.mode = "eval"
            self.agent.policy_net.eval()
            self.agent.target_net.eval()
            
            self.logger.info("RL模型加载成功，已设置为评估模式")
            return True
        except Exception as e:
            self.logger.error(f"加载RL模型失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    # ... (其他方法如 _update_agent_online, update_query 不变)

    # !! 新增方法 !!
    def update_historical_context(self, historical_samples):
        """
        更新当前查询对应的历史上下文样本。

        Args:
            historical_samples: 从历史中检索到的样本列表 [(sample_tuple, direction), ...]
        """
        self.current_historical_samples = historical_samples or []
        self.logger.info(f"RL策略已更新历史上下文，样本数: {len(self.current_historical_samples)}")
        # 如果环境实例已存在，也更新它内部的上下文
        if self.environment is not None and hasattr(self.environment, 'set_historical_context'):
            self.logger.debug("同时更新现有环境实例的历史上下文")
            self.environment.set_historical_context(self.current_historical_samples)

    def update_query(self, query):
        """
        更新当前查询

        Args:
            query: 当前查询样本和方向的元组 ((entity, relation, targets, timestamp), direction)
        """
        # 存储查询，以便在重置环境时使用
        self.current_query = query
        self.logger.info(f"RL策略已更新查询: {query[0][0]}, {query[0][1]}, 方向: {query[1]}")

        if not RL_AVAILABLE:
            return

        # 如果环境实例已存在，则调用其更新查询方法（可能触发状态重置）
        if self.environment is not None and hasattr(self.environment, "update_query"):
            # 注意：环境的update_query可能会重置部分状态，如selected_samples
            self.environment.update_query(query)
            self.logger.debug("同时更新现有环境实例的查询")
        # 如果环境不存在，查询将在下次select_samples时通过reset传递

    def update_exploration(self, ratio=None):
        """
        更新探索率 (如果Agent支持)

        Args:
            ratio: 新的探索率，如果为None则使用配置中的默认值
        """
        if not RL_AVAILABLE or self.agent is None:
            return

        # 确定要设置的探索率
        target_ratio = ratio
        if target_ratio is None:
            config = self.config
            online_learning_cfg = config.get("online_learning", {})
            target_ratio = online_learning_cfg.get("exploration_ratio", 0.1) # 默认探索率

        # 尝试更新Agent的探索率
        updated = False
        if hasattr(self.agent, "set_exploration"):
            try:
                self.agent.set_exploration(target_ratio)
                updated = True
            except Exception as e:
                 self.logger.warning(f"调用 agent.set_exploration({target_ratio}) 出错: {e}")
        elif hasattr(self.agent, "epsilon"):
            try:
                self.agent.epsilon = target_ratio
                updated = True
            except Exception as e:
                 self.logger.warning(f"设置 agent.epsilon = {target_ratio} 出错: {e}")

        if updated:
            self.logger.info(f"RL Agent探索率已更新为: {target_ratio:.4f}")
        else:
             self.logger.warning(f"无法更新Agent的探索率 (Agent类型: {type(self.agent).__name__})")

    def _create_global_id_mappings(self, samples_list):
        """从给定的样本列表创建全局ID映射 (作为后备)"""
        self.logger.warning("正在从当前样本创建临时的全局ID映射")
        all_entities = set()
        all_relations = set()
        for sample, _ in samples_list:
            h, r, t_list, _ = sample
            all_entities.add(h)
            all_relations.add(r)
            all_entities.update(t_list)
        
        self.global_entity_ids = {e: i for i, e in enumerate(sorted(list(all_entities)))}
        self.global_relation_ids = {r: i for i, r in enumerate(sorted(list(all_relations)))}
        self.logger.info(f"创建临时全局ID映射完成: {len(self.global_entity_ids)}个实体, {len(self.global_relation_ids)}个关系")


def get_strategy(strategy_name, config_path=None):
    """根据策略名称获取对应的策略实例
    
    Args:
        strategy_name: 策略名称
        config_path: RL策略的配置文件路径（可选）
    
    Returns:
        策略实例
    """
    # 基本策略实例
    strategies = {
        "random": RandomStrategy(),
        # "max_entropy": MaxEntropyStrategy(),
        # "best_of_k": BestOfKStrategy(),
        # "random_balanced": RandomBalancedStrategy(),
    }
    
    # 特殊处理RL策略
    if strategy_name == "rl":
        strategies["rl"] = RLStrategy(config_path=config_path)
    
    if strategy_name not in strategies:
        print(f"警告: 未知策略 '{strategy_name}'，使用默认的随机策略")
        return strategies["random"]
    
    return strategies[strategy_name]
