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


class RandomBalancedStrategy(BaseStrategy):
    """按方向平衡的随机选择策略"""
    def __init__(self):
        super().__init__("random_balanced")
    
    def select_samples(self, candidates, num_samples, model, tokenizer, history, args):
        """平衡head和tail方向随机选择样本"""
        # 按方向分组
        head_indices = []
        tail_indices = []
        
        for i, (sample, direction, _) in enumerate(candidates):
            if direction == "head":
                head_indices.append(i)
            else:
                tail_indices.append(i)
        
        # 计算每个方向应选择的样本数
        total_samples = min(num_samples, len(candidates))
        head_count = min(len(head_indices), total_samples // 2 + total_samples % 2)
        tail_count = min(len(tail_indices), total_samples - head_count)
        
        # 如果一个方向的样本不足，从另一个方向补充
        if head_count < total_samples // 2 + total_samples % 2:
            tail_count = min(len(tail_indices), total_samples - head_count)
        
        # 随机选择每个方向的样本
        selected_head_idx = random.sample(head_indices, head_count) if head_count > 0 else []
        selected_tail_idx = random.sample(tail_indices, tail_count) if tail_count > 0 else []
        
        # 合并选中的索引
        selected_indices = selected_head_idx + selected_tail_idx
        
        # 随机打乱选中的索引顺序
        random.shuffle(selected_indices)
        
        # 计算未选中的索引
        all_indices = list(range(len(candidates)))
        unselected_indices = [i for i in all_indices if i not in selected_indices]
        
        return selected_indices, unselected_indices


class MaxEntropyStrategy(BaseStrategy):
    """最大熵策略"""
    def __init__(self):
        super().__init__("max_entropy")
    
    def _calculate_entropy(self, predictions):
        """计算预测结果的熵"""
        probs = [p[1] for p in predictions]
        entropy = -sum(p * math.log(p) for p in probs if p > 0)
        return entropy
    
    def select_samples(self, candidates, num_samples, model, tokenizer, history, args):
        """选择预测熵最大的样本"""
        if not candidates:
            return [], []
        
        print(f"MaxEntropyStrategy: 计算{len(candidates)}个样本的熵...")
        
        # 计算每个样本的熵
        sample_entropies = []
        
        for i, (sample, direction, _) in enumerate(tqdm(candidates)):
            try:
                # 创建空搜索空间（因为我们只需要计算熵，不需要实际预测）
                empty_search_space = {}
                
                # 准备输入 - 使用新的函数签名
                model_input, candidates_map, _ = prepare_input(sample, empty_search_space, args, return_prompt=True)
                
                # 预测并计算熵
                predictions = predict(tokenizer, model, model_input, args)
                entropy_value = self._calculate_entropy(predictions)
                
                sample_entropies.append((i, entropy_value))
            except Exception as e:
                print(f"计算样本{i}的熵时出错: {e}")
                # 出错时赋予最小熵值
                sample_entropies.append((i, -float('inf')))
        
        # 按熵降序排序并选择指定数量的样本
        sample_entropies.sort(key=lambda x: x[1], reverse=True)
        num_to_select = min(num_samples, len(sample_entropies))
        
        selected_indices = [idx for idx, _ in sample_entropies[:num_to_select]]
        
        # 计算未选中的索引
        all_indices = list(range(len(candidates)))
        unselected_indices = [i for i in all_indices if i not in selected_indices]
        
        return selected_indices, unselected_indices


class BestOfKStrategy(BaseStrategy):
    """从K个随机集合中选择最佳子集"""
    def __init__(self):
        super().__init__("best_of_k")
    
    def _calculate_entropy(self, predictions):
        """计算预测结果的熵"""
        probs = [p[1] for p in predictions]
        entropy = -sum(p * math.log(p) for p in probs if p > 0)
        return entropy
    
    def select_samples(self, candidates, num_samples, model, tokenizer, history, args):
        """从K个随机子集中选择平均熵最高的子集"""
        if not candidates:
            return [], []
        
        # 参数设置
        K = 5  # 尝试的随机子集数量
        subset_size = min(num_samples, len(candidates))
        
        if subset_size == 0:
            return [], list(range(len(candidates)))
        
        print(f"BestOfKStrategy: 评估{K}个随机子集...")
        
        best_subset = None
        best_entropy = -float('inf')
        
        for k in range(K):
            # 随机选择一个子集
            random_idx = random.sample(range(len(candidates)), subset_size)
            
            # 计算子集的平均熵
            entropy_sum = 0
            sample_count = 0
            
            for idx in random_idx:
                try:
                    sample, direction, _ = candidates[idx]
                    # 创建空搜索空间
                    empty_search_space = {}
                    
                    # 准备输入 - 使用新的函数签名
                    model_input, candidates_map, _ = prepare_input(sample, empty_search_space, args, return_prompt=True)
                    
                    # 预测并计算熵
                    predictions = predict(tokenizer, model, model_input, args)
                    entropy_value = self._calculate_entropy(predictions)
                    
                    entropy_sum += entropy_value
                    sample_count += 1
                except Exception as e:
                    print(f"计算样本熵时出错: {e}")
            
            # 计算平均熵
            if sample_count > 0:
                avg_entropy = entropy_sum / sample_count
                
                # 更新最佳子集
                if avg_entropy > best_entropy:
                    best_entropy = avg_entropy
                    best_subset = random_idx
        
        if best_subset:
            # 计算未选中的索引
            all_indices = list(range(len(candidates)))
            unselected_indices = [i for i in all_indices if i not in best_subset]
            return best_subset, unselected_indices
        else:
            # 如果没有找到最佳子集，回退到随机选择
            return RandomStrategy().select_samples(candidates, num_samples, model, tokenizer, history, args)


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

        # 立即加载配置 (新增)
        if config_path:
            self.__class__._load_config(direct_path=config_path)

        # 实例变量，用于存储单个查询的上下文和环境/代理
        self.environment = None
        self.agent = None
        self.current_historical_samples = [] # !! 新增: 存储历史上下文 !!
        self.global_entity_ids = None
        self.global_relation_ids = None

        # BERT模型（可选，用于特征提取）
        self.bert_model = None
        self.bert_tokenizer = None
        self._load_bert_model_if_needed()
        
    def _load_bert_model_if_needed(self):
        config = self._load_config(args=None)
        feature_config = config.get("feature_config", {})
        if feature_config.get("use_plm_embeddings", False):
            encoder_path = feature_config.get("plm_encoder_path", "bert-base-uncased")
            try:
                from src.rl.plm_encoder import PLMEncoder
                # 加载到实例变量
                self.bert_model = PLMEncoder(encoder_path)
                self.bert_tokenizer = self.bert_model.tokenizer # 假设编码器暴露tokenizer
                self.logger.info(f"RL策略已加载特征提取器: {encoder_path}")
            except Exception as e:
                self.logger.error(f"加载PLM编码器 {encoder_path} 失败: {e}", exc_info=True)
                self.bert_model = None
                self.bert_tokenizer = None

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

    def _initialize_agent(self, environment, args):
        config = self._load_config(args=args)
        agent_type = config.get("agent_type", "dqn").lower()
        agent_config = config.get("agent_config", {})
        
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

        if agent_type == "dqn":
            from src.rl.agents.dqn_agent import DQNAgent # 延迟导入
            
            # 设置PyTorch的默认CUDA设备，这样initialize_network会使用这个设备
            if selected_gpu is not None:
                self.logger.info(f"设置PyTorch默认CUDA设备为: cuda:{selected_gpu}")
                # 保存原始设备设置
                original_device = torch.cuda.current_device()
                # 设置新的默认设备
                torch.cuda.set_device(selected_gpu)
                # 记录新设置的设备
                self.logger.info(f"当前PyTorch CUDA设备设置为: {torch.cuda.current_device()}")
            
            # 获取output_dir参数（存储DQN模型的目录）
            output_dir = config.get("output_dir", "./rl_outputs")
            if hasattr(args, 'output_dir') and args.output_dir:
                output_dir = args.output_dir
                
            # 创建一个字典，包含DQNAgent所需的其他参数
            dqn_params = {
                'output_dir': output_dir,
            }
            
            # 确保environment被移动到正确的设备
            if hasattr(environment, 'to') and callable(environment.to):
                try:
                    environment.to(device)
                    self.logger.info(f"环境已移动到设备: {device}")
                except Exception as e:
                    self.logger.warning(f"无法将环境移动到设备 {device}: {e}")
            
            # 处理network_params - 确保不传递device参数
            # initialize_network内部会自己设置device
            if 'network_params' in agent_config:
                # 确保network_params是一个字典
                if not isinstance(agent_config['network_params'], dict):
                    agent_config['network_params'] = {}
                    
                # 确保移除device参数，防止传递给initialize_network
                if 'device' in agent_config['network_params']:
                    self.logger.info("从network_params中移除device参数")
                    agent_config['network_params'].pop('device')
            else:
                # 如果不存在，创建一个空字典
                agent_config['network_params'] = {}
            
            # 移除DQNAgent不支持的参数
            # 直接删除gamma参数（而不是使用过滤逻辑）
            if 'gamma' in agent_config:
                self.logger.info("删除gamma参数，使用硬编码值1")
                agent_config.pop('gamma')
                
            # 参数名称映射（配置文件->DQNAgent）
            param_mapping = {
                'epsilon_start': 'eps_start',
                'epsilon_end': 'eps_end',
                'epsilon_decay': 'decay_steps'
            }
            
            # 创建新的参数字典，映射参数名称
            mapped_config = {}
            for key, value in agent_config.items():
                if key in param_mapping:
                    self.logger.info(f"将参数 {key} 映射到 {param_mapping[key]}")
                    mapped_config[param_mapping[key]] = value
                else:
                    mapped_config[key] = value
            
            # 将映射后的参数添加到dqn_params中
            dqn_params.update(mapped_config)
            
            # 传递environment作为第一个参数，然后解包其他参数
            self.logger.info(f"创建DQNAgent，参数: {dqn_params}")
            agent = DQNAgent(environment, **dqn_params)
            
            # 如果之前修改了默认CUDA设备，现在恢复原始设置
            if selected_gpu is not None and 'original_device' in locals():
                self.logger.info(f"恢复PyTorch默认CUDA设备为: cuda:{original_device}")
                torch.cuda.set_device(original_device)
            
        elif agent_type == "random":
             from src.rl.agents.random_agent import RandomAgent # 延迟导入
             agent = RandomAgent(
                 action_feature_dim=environment.action_dim # 随机Agent也可能需要动作特征维度
             )
        else:
            self.logger.error(f"未知的Agent类型: {agent_type}，将使用随机Agent")
            from src.rl.agents.random_agent import RandomAgent
            agent = RandomAgent(action_feature_dim=environment.action_dim)

        # 尝试加载预训练模型
        pretrained_path = config.get("pretrained_agent_path", None)
        if pretrained_path and hasattr(agent, 'load_model'):
            try:
                agent.load_model(pretrained_path)
                self.logger.info(f"RL Agent模型已从 {pretrained_path} 加载")
            except Exception as e:
                self.logger.error(f"加载RL Agent预训练模型 {pretrained_path} 失败: {e}")

        return agent

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
                self.environment = TKGEnvironment(
                    interaction_data=candidates,  # 主要用于获取实体/关系映射
                    state_model=self.bert_model, # 使用传入的主模型进行评估
                    state_tokenizer=self.bert_tokenizer,
                    reward_model=model,
                    reward_tokenizer=tokenizer,
                    train_history=history,  # 传递搜索空间字典
                    args=args,
                    state_repr=self._load_config(args=args).get("state_repr", ["query_features", "context_features"]), # 从配置加载
                    max_steps=num_samples,
                    reward_scale=self._load_config(args=args).get("reward_scale", 1.0),
                    predef_entity_ids=self.global_entity_ids,
                    predef_relation_ids=self.global_relation_ids,
                    state_model_type="encoder", # 从args或模型类型推断
                )
            else:
                # 如果环境已存在，更新其配置
                self.environment.train_history = history
                self.environment.interaction_data = candidates
                self.environment.max_steps = num_samples
                # 重置环境状态
                self.environment.reset()

            # 初始化或获取代理
            if self.agent is None:
                self.agent = self._initialize_agent(self.environment, args)
        
            # 运行RL选择过程
            self.logger.info(f"RL策略开始选择样本，最多选择 {num_samples} 个")
            
            # 进行RL选择过程
            selected_indices = []
            steps_taken = 0
            
            state = self.environment.state
            all_indices = list(range(len(candidates)))
            
            while steps_taken < num_samples and len(selected_indices) < len(candidates):
                # 计算有效的动作索引（排除已选的）
                valid_actions = [i for i in all_indices if i not in selected_indices]
                
                if not valid_actions:
                    break
                
                # 选择动作
                action_idx = self.agent.choose_action(state, valid_actions)
                
                # 记录选择的索引
                selected_indices.append(action_idx)
                steps_taken += 1
                
                # 更新状态（简化处理）
                # 在实际环境中，这里应该调用environment.step(action_idx)
                
            # 计算未选中的索引
            unselected_indices = [i for i in all_indices if i not in selected_indices]
            
            self.logger.info(f"RL策略选择完成，共选择了 {len(selected_indices)} 个样本")
            return selected_indices, unselected_indices

        except Exception as e:
            self.logger.error(f"RL样本选择过程中出错: {e}")
            import traceback
            traceback.print_exc()
            # 回退到随机策略
            self.logger.warning("RL策略失败，回退到随机策略")
            return RandomStrategy().select_samples(candidates, num_samples, model, tokenizer, history, args)

    def load_model(self, model_path=None):
        """加载预训练的RL模型（例如从训练阶段保存的模型）"""
        # 如果没有传入路径，使用配置中指定的路径
        if model_path is None:
            config = self._load_config(args=None)
            model_path = config.get("pretrained_agent_path", None)
            if not model_path:
                self.logger.warning("未指定预训练模型路径，无法加载模型")
                return False
        
        if not self.agent:
            self.logger.warning("无法加载模型: RL代理尚未初始化")
            return False
            
        try:
            self.logger.info(f"尝试从 {model_path} 加载RL模型")
            self.agent.load_model(model_path)
            self.logger.info("RL模型加载成功")
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
            config = self._load_config()
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
        "max_entropy": MaxEntropyStrategy(),
        "best_of_k": BestOfKStrategy(),
        "random_balanced": RandomBalancedStrategy(),
    }
    
    # 特殊处理RL策略
    if strategy_name == "rl":
        strategies["rl"] = RLStrategy(config_path=config_path)
    
    if strategy_name not in strategies:
        print(f"警告: 未知策略 '{strategy_name}'，使用默认的随机策略")
        return strategies["random"]
    
    return strategies[strategy_name]
