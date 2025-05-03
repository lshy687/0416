import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, logging as tf_logging
import os
import random

from model_utils import predict
from utils import (
    HitsMetric,
    adjust_top_k,
    get_args,
    get_filename,
    load_data,
    prepare_input,
    update_history,
    update_metric,
    write_results,
)
# 导入主动学习相关模块
from active_learning import (
    get_current_time_samples,
    integrate_active_samples,
    get_strategy
)

# 设置 transformers 日志级别为仅显示错误信息
tf_logging.set_verbosity_error()


def train_rl_strategy(train_data, valid_data, model, tokenizer, args):
    """
    训练强化学习策略
    
    Args:
        train_data: 训练数据集 (用于构建历史)
        valid_data: 验证数据集 (用于RL环境的交互)
        model: 预测模型 (主LLM, 如Qwen)
        tokenizer: 分词器 (主LLM)
        args: 运行参数
    """
    # 判断是否需要训练RL策略
    if not args.rl_train and not args.offline_rl:
        return
    
    print("\n===== 强化学习训练模式 =====")
    if args.offline_rl:
        print("使用离线训练模式")
        if not args.offline_data_path:
            print("错误: 请指定离线数据路径 (--offline_data_path)")
            return
        print(f"离线数据文件: {args.offline_data_path}")
        print(f"离线训练步数: {args.offline_train_steps}")
        print(f"CQL损失权重: {args.cql_weight}")
    else:
        print("使用在线训练模式")
    
    # 获取RL策略
    strategy = get_strategy("rl", config_path=args.rl_config)
    
    # 检查RL是否可用
    if not hasattr(strategy, "agent") or strategy.agent is None:
        try:
            # 导入所需模块
            from src.rl.tkg_environment import TKGEnvironment
            from src.rl.agents.dqn_agent import DQNAgent
            
            # ---> 修复：将导入语句移到 if 条件之外 <---
            from transformers import AutoModel, AutoTokenizer

            # 加载BERT模型用于特征提取
            try:
                # 检查策略对象是否已经加载了BERT模型和分词器
                bert_already_loaded = (hasattr(strategy, 'bert_model') and strategy.bert_model is not None and
                                       hasattr(strategy, 'bert_tokenizer') and strategy.bert_tokenizer is not None)

                if not bert_already_loaded:
                    # ---> 移除这里的导入语句 <---
                    
                    print(f"为RL训练加载BERT模型: {args.bert_model}")
                    # 确保 bert_tokenizer 和 bert_model 使用 args.bert_model 加载
                    # 现在 AutoTokenizer 和 AutoModel 肯定是指向导入的类了
                    bert_tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
                    bert_model = AutoModel.from_pretrained(args.bert_model)
                    
                    # 将BERT模型移动到与预测模型相同的设备上
                    device = next(model.parameters()).device
                    bert_model = bert_model.to(device)
                    
                    # 设置为评估模式
                    bert_model.eval()
                    print(f"BERT模型已加载到设备: {next(bert_model.parameters()).device}")
                    
                    # 设置策略的BERT模型和分词器
                    strategy.bert_model = bert_model
                    strategy.bert_tokenizer = bert_tokenizer
                else:
                    print("使用策略对象中已有的BERT模型和分词器。")
                    # 如果需要，可以添加检查确保已有的模型在正确设备上
                    device = next(model.parameters()).device
                    # ---> FIX: Access the actual model inside the PLMEncoder <---
                    actual_bert_model = strategy.bert_model.model # Access the model attribute

                    if actual_bert_model.device != device:
                       print(f"将策略对象中的BERT模型从 {actual_bert_model.device} 移动到 {device}")
                       actual_bert_model = actual_bert_model.to(device)
                       # 更新 PLMEncoder 实例内部的模型引用和设备属性
                       strategy.bert_model.model = actual_bert_model
                       strategy.bert_model.device = device # Assuming PLMEncoder has a device attribute (it does)
                    actual_bert_model.eval()
                    print(f"已确认或移动策略对象中的BERT模型到设备: {actual_bert_model.device}")

            except Exception as e:
                print(f"处理或加载BERT模型时出错: {e}，RL 训练可能失败")
                return # 如果BERT加载失败，则无法继续
            
            # --- 修改：创建并初始化搜索空间 ---
            print("为RL环境构建训练集历史...")
            # 创建搜索空间
            entity_search_space = {}
            for sample, _ in tqdm(train_data, desc="构建训练历史"):
                entity, relation, targets, time = sample
                
                # 确保嵌套字典结构存在
                if entity not in entity_search_space:
                    entity_search_space[entity] = {}
                if time not in entity_search_space[entity]:
                    entity_search_space[entity][time] = {}
                if relation not in entity_search_space[entity][time]:
                    entity_search_space[entity][time][relation] = []
                
                # 添加目标到搜索空间
                # 注意：这里我们不使用update_history函数，因为我们只是初始化搜索空间
                entity_search_space[entity][time][relation] += targets
            
            print(f"训练集历史构建完成. 搜索空间大小: {len(entity_search_space)}")
            # --- 结束修改 ---
            
            # 确保使用全局ID映射，保证验证集和测试集ID一致性
            if not strategy.global_entity_ids or not strategy.global_relation_ids:
                try:
                    from utils import load_all_entity_relation_mappings
                    
                    print("加载全局ID映射，确保训练和测试使用相同的ID...")
                    entity_ids, relation_ids = load_all_entity_relation_mappings(args)
                    
                    if entity_ids and relation_ids:
                        strategy.global_entity_ids = entity_ids
                        strategy.global_relation_ids = relation_ids
                        print(f"成功加载全局ID映射: {len(entity_ids)}个实体, {len(relation_ids)}个关系")
                    else:
                        print("无法加载全局ID映射，将从数据创建映射")
                        # 同时使用训练集和验证集创建映射，以覆盖更多ID
                        strategy._create_global_id_mappings([train_data, valid_data])
                except Exception as e:
                    print(f"加载/创建全局ID映射时出错: {e}，将只使用训练和验证数据")
                    strategy._create_global_id_mappings([train_data, valid_data])
            
            # 初始化环境，传递主LLM模型、分词器和训练历史
            print("初始化 TKGEnvironment...")
            # 初始化环境，传递主LLM模型、分词器和训练历史
            environment = TKGEnvironment(
                interaction_data=valid_data, # 使用验证集进行交互
                # ---> FIX: Pass the raw model and tokenizer <---
                state_model=strategy.bert_model.model,  # 使用原始BERT模型
                state_tokenizer=strategy.bert_tokenizer, # 假设分词器存储在 strategy.bert_tokenizer
                reward_model=model, # <--- 使用主LLM计算奖励
                reward_tokenizer=tokenizer, # <--- 使用主LLM的分词器
                train_history=entity_search_space, # <--- 修改：传递搜索空间
                args=args,
                state_repr=["query_features", "context_features", "interaction_features", "diversity_features", "curr_step"],
                max_steps=args.active_samples,
                reward_scale=10.0,
                predef_entity_ids=strategy.global_entity_ids,
                predef_relation_ids=strategy.global_relation_ids,
                state_model_type="encoder" # 指定状态模型类型为encoder (BERT)
            )
            
            if environment is None:
                print("警告: 无法初始化RL环境，训练已跳过")
                return
            
            # --- 添加离线RL训练模式 ---
            # 根据模式设置额外参数
            agent_extra_params = {}
            if args.offline_rl:
                agent_extra_params["cql_loss_weight"] = args.cql_weight  # 设置CQL权重
                agent_extra_params["offline_steps"] = args.offline_train_steps  # 设置离线训练步数
                agent_extra_params["load_transitions"] = args.offline_data_path  # 设置离线数据路径
            
            # 初始化代理，传递额外参数
            agent = None # 初始化 agent 为 None
            init_exception = None # 用于存储可能的异常

            try:
                # 调用 _initialize_agent (这个方法会设置 strategy.agent)
                strategy._initialize_agent(environment, args, **agent_extra_params)
                # 从 strategy 对象获取 agent 实例 (可能是 None)
                agent = strategy.agent
            except Exception as e:
                print(f"DEBUG: 在调用 _initialize_agent 时捕获到异常: {e}")
                import traceback
                traceback.print_exc()
                init_exception = e # 保存异常
                agent = None # 确保初始化失败时 agent 为 None
            
            if agent is None:
                print("警告: 无法初始化RL代理，训练已跳过")
                # ---> 在这里添加详细的调试打印 <---
                if init_exception:
                    print(f"DEBUG: 代理初始化失败，因为发生了异常: {init_exception}")
                else:
                    print("DEBUG: 代理初始化失败，但在此处未捕获到特定异常。请检查 RLStrategy._initialize_agent 内部的日志。")
                # ---> 调试打印结束 <---
                return
                
            # 设置策略的环境和代理 (如果 agent 不是 None)
            strategy.environment = environment
            
            # 设置环境模式为训练模式
            if hasattr(environment, "set_mode"):
                environment.set_mode("train")
            
            # 只有在线训练模式才需要设置初始查询
            if not args.offline_rl:
                # 设置初始查询（第一个验证集样本）
                if len(valid_data) > 0:
                    first_sample = valid_data[0]
                    sample, direction = first_sample
                    print(f"设置初始查询: {sample[0]}, {sample[1]}, 方向: {direction}")
                    environment.update_query(first_sample)
            
            # --- 根据模式选择不同的训练流程 ---
            if args.offline_rl:
                # 离线训练模式
                print(f"开始离线RL训练，使用数据: {args.offline_data_path}")
                print(f"训练步数: {args.offline_train_steps}, CQL权重: {args.cql_weight}")
                
                # 直接调用train方法，使用离线数据进行训练
                agent.train(steps=args.offline_train_steps)
                
                print("离线RL训练完成！")
            else:
                # 在线训练模式 - 原有逻辑
                print(f"开始训练RL策略，总步数: {agent.train_steps}")
                
                # ===== 修改开始：按时间戳处理 =====
                
                # 获取所有时间戳
                timestamps = sorted(list(set([sample[3] for sample, _ in valid_data])))
                print(f"训练将按顺序处理 {len(timestamps)} 个时间戳")
                
                # 对每个时间戳单独训练
                for ts_idx, timestamp in enumerate(timestamps):
                    # 获取当前时间戳的样本
                    current_samples = [(sample, direction) for sample, direction in valid_data if sample[3] == timestamp]
                    
                    if not current_samples:
                        print(f"时间戳 {timestamp} 没有样本，跳过")
                        continue
                        
                    print(f"处理时间戳 {timestamp} ({ts_idx+1}/{len(timestamps)}), 样本数: {len(current_samples)}")
                    
                    # 为每个时间戳重置经验回放缓冲区
                    # 根据样本数调整缓冲区大小，确保至少能容纳所有可能的样本乘以一个因子
                    buffer_size = min(10000, max(1000, len(current_samples) * 5))
                    agent.reset_replay_memory(buffer_size)
                    
                    # 更新环境的当前样本
                    indices = list(range(len(current_samples)))
                    environment.update_candidates(current_samples, indices)
                    
                    # 为每个时间戳训练固定步数
                    local_train_steps = min(500, max(100, len(current_samples) // 2))
                    print(f"为时间戳 {timestamp} 训练 {local_train_steps} 步")
                    
                    # 调用训练，但使用冻结动作空间
                    try:
                        # 明确说明使用冻结动作空间策略
                        print(f"使用冻结动作空间为时间戳 {timestamp} 训练 {local_train_steps} 步")
                        # 调用agent.train()方法，显式传递steps参数
                        agent.train(steps=local_train_steps)
                        
                    except Exception as e:
                        print(f"时间戳 {timestamp} 训练期间出错: {e}")
                        continue
            
            # 统一的保存和清理逻辑
            try:
                # 保存最佳模型
                best_path = os.path.join(args.output_dir, "ckpts", "best_rl_model.pt")
                os.makedirs(os.path.dirname(best_path), exist_ok=True)
                torch.save(agent.policy_net.state_dict(), best_path)
                print(f"最终模型已保存到 {best_path}")
            except Exception as e:
                print(f"保存模型时出错: {e}")
            
        except Exception as e:
            print(f"初始化RL环境/代理时出错: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("RL策略已初始化，跳过重新初始化")
    
    # 设置环境和代理为评估模式 (离线训练后仍需设置)
    if hasattr(strategy, "environment") and strategy.environment is not None:
        strategy.environment.set_mode("val")
    if hasattr(strategy, "agent") and strategy.agent is not None:
        strategy.agent.mode = "val"


def create_global_id_mappings(args):
    """
    创建全局ID映射并保存到文件
    
    Args:
        args: 运行参数
        
    Returns:
        entity_ids: 实体ID映射
        relation_ids: 关系ID映射
    """
    try:
        from utils import load_all_entity_relation_mappings, save_global_id_mappings
        # 直接从数据集创建映射
        print("创建全局ID映射...")
        global_entity_ids, global_relation_ids = load_all_entity_relation_mappings(args)
        if global_entity_ids and global_relation_ids:
            # 可选：保存映射供其他程序使用
            save_global_id_mappings(global_entity_ids, global_relation_ids)
            print(f"已创建全局ID映射: {len(global_entity_ids)}个实体, {len(global_relation_ids)}个关系")
        else:
            print("警告: 创建全局ID映射失败")
            global_entity_ids, global_relation_ids = {}, {}
    except Exception as e:
        print(f"处理全局ID映射时出错: {e}. 将使用空映射。")
        global_entity_ids, global_relation_ids = {}, {}


# 修改 select_samples_for_query 函数，使其能够正确使用 RL 策略

def select_samples_for_query(
    query_sample,
    current_time_samples,
    active_learning_strategy,
    num_samples_to_select,
    model,
    tokenizer,
    history,
    args
):
    """
    为特定查询选择主动学习样本
    
    Args:
        query_sample: 当前需要预测的样本 (sample_tuple, direction)
        current_time_samples: 当前时间戳下的所有样本
        active_learning_strategy: 主动学习策略实例
        num_samples_to_select: 需要选择的样本数量
        model: 用于评估的模型
        tokenizer: 模型的分词器
        history: 历史搜索空间
        args: 运行参数
        
    Returns:
        list: 选中的样本列表，格式为 [(sample_tuple, direction), ...]
    """
    # 如果没有启用主动学习或样本数量不足，返回空列表
    if not args.active_learning or active_learning_strategy is None:
        return []
    
    # 构建候选样本池（排除当前查询样本）
    candidate_samples = []
    candidate_indices = []
    
    # 创建当前查询的唯一标识
    query_id = f"{query_sample[0][0]}_{query_sample[0][1]}_{query_sample[0][2][0]}_{query_sample[0][3]}_{query_sample[1]}"
    
    # 遍历当前时间戳的所有样本，将非当前查询的样本添加到候选池
    for i, sample in enumerate(current_time_samples):
        sample_id = f"{sample[0][0]}_{sample[0][1]}_{sample[0][2][0]}_{sample[0][3]}_{sample[1]}"
        if sample_id != query_id:  # 排除当前查询样本自身
            candidate_samples.append(sample)
            candidate_indices.append(i)
    
    # 如果候选样本数量小于需要选择的数量，直接返回所有候选样本
    if len(candidate_samples) <= num_samples_to_select:
        return candidate_samples
    
    # 使用当前查询作为参考，从候选池中选择最有价值的样本
    try:
        if active_learning_strategy.name == "random":
            # 随机选择指定数量的样本
            selected_indices, _ = active_learning_strategy.select_samples(
                candidates=candidate_samples,
                num_samples=num_samples_to_select,
                model=model,
                tokenizer=tokenizer,
                history=history,
                args=args
            )
            # 将索引转换为样本
            return [candidate_samples[i] for i in selected_indices]
        elif active_learning_strategy.name == "rl":
            # 更新：使用 RLStrategy 的 select_samples 方法
            print(f"使用 RL 策略为查询 {query_id} 选择样本")
            
            # 1. 更新主动学习策略的当前查询信息
            if hasattr(active_learning_strategy, 'update_query'):
                active_learning_strategy.update_query(query_sample)
            
            # 2. 更新主动学习策略的历史信息
            if hasattr(active_learning_strategy, 'update_historical_context'):
                active_learning_strategy.update_historical_context(history)
                
            # 3. 调用主动学习策略的选择方法
            # 注意：此处select_samples返回(selected_indices, unselected_indices)
            selected_indices, _ = active_learning_strategy.select_samples(
                candidates=candidate_samples,
                num_samples=num_samples_to_select,
                model=model,
                tokenizer=tokenizer,
                history=history,
                args=args
            )
            
            # 4. 将选中的索引转换为样本
            selected_samples = [candidate_samples[i] for i in selected_indices]
            print(f"RL 策略已为查询 {query_id} 选择 {len(selected_samples)} 个样本")
            
            return selected_samples
        else:
            # 其他策略：也使用select_samples接口
            selected_indices, _ = active_learning_strategy.select_samples(
                candidates=candidate_samples,
                num_samples=num_samples_to_select,
                model=model,
                tokenizer=tokenizer,
                history=history,
                args=args
            )
            # 将索引转换为样本
            return [candidate_samples[i] for i in selected_indices]
    except Exception as e:
        print(f"为查询 {query_id} 选择样本时出错: {e}")
        import traceback
        traceback.print_exc()
        # 发生错误时返回随机样本作为备选方案
        print(f"回退到随机选择 {min(num_samples_to_select, len(candidate_samples))} 个样本")
        try:
            selected_indices = random.sample(range(len(candidate_samples)), min(num_samples_to_select, len(candidate_samples)))
            return [candidate_samples[i] for i in selected_indices]
        except:
            # 如果随机选择也失败，返回空列表
            return []


if __name__ == "__main__":
    # 获取命令行参数
    args = get_args()
    
    # 首先创建全局ID映射 - 确保所有模块使用一致的ID映射
    # 提前创建，因为后面加载训练/验证/测试数据都需要它
    global_entity_ids, global_relation_ids = {}, {}
    # if args.active_learning and args.active_strategy == "rl": # 无论如何都创建，保证ID一致性
    try:
        from utils import load_all_entity_relation_mappings, save_global_id_mappings
        # 直接从数据集创建映射
        print("创建全局ID映射...")
        global_entity_ids, global_relation_ids = load_all_entity_relation_mappings(args)
        if global_entity_ids and global_relation_ids:
            # 可选：保存映射供其他程序使用
            save_global_id_mappings(global_entity_ids, global_relation_ids)
            print(f"已创建全局ID映射: {len(global_entity_ids)}个实体, {len(global_relation_ids)}个关系")
        else:
            print("警告: 创建全局ID映射失败")
            global_entity_ids, global_relation_ids = {}, {}
    except Exception as e:
        print(f"处理全局ID映射时出错: {e}. 将使用空映射。")
        global_entity_ids, global_relation_ids = {}, {}

    # --- 根据模式加载不同数据 ---
    train_data_full = None
    valid_data_full = None
    test_data_full = None
    head_search_space = None # Initialize search spaces
    tail_search_space = None

    # --- 修改: 始终加载训练集和验证集, 因为标准预测需要两者构建历史 ---
    print("加载训练集数据...")
    train_data_full, _, _ = load_data(args, mode="train", global_entity_ids=global_entity_ids, global_relation_ids=global_relation_ids)
    print(f"训练集: {len(train_data_full)} 条")

    print("加载验证集数据...")
    valid_data_full, _, _ = load_data(args, mode="valid", global_entity_ids=global_entity_ids, global_relation_ids=global_relation_ids)
    print(f"验证集: {len(valid_data_full)} 条")
    # --- 结束修改 ---


    if args.active_learning and args.active_strategy == "rl" and args.rl_train or args.offline_rl:
        # RL训练模式: 验证集用于交互, 训练集用于历史 (已在 train_rl_strategy 处理)
        print("RL训练模式: 使用验证集进行交互，训练集构建历史 (在策略函数内处理)")
        # 测试集在此模式下不需要加载，也不需要调整 top_k
    else:
        # 标准预测模式 (包括使用RL进行预测)
        print("标准预测模式: 加载测试集数据...")
        test_data_full, head_search_space, tail_search_space = load_data(args, mode="test", global_entity_ids=global_entity_ids, global_relation_ids=global_relation_ids)
        print(f"测试集: {len(test_data_full)} 条")
        adjust_top_k(test_data_full, args) # 在测试集上调整 top_k


    # 加载预训练模型的分词器和模型 (主 LLM)
    print(f"加载主 LLM 模型: {args.model}")
    # tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.tokenizer_revision)
    # 使用绝对路径加载 Qwen 模型
    tokenizer = AutoTokenizer.from_pretrained("/data/shangyuan/models/DeepSeek-R1-Distill-Qwen-1.5B", revision="main")
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # 动态选择GPU
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    if args.gpu == -2:
        try:
            import subprocess
            import re
            output = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'])
            memory_free = [int(x) for x in output.decode('utf-8').strip().split('\n')]
            available_gpu = memory_free.index(max(memory_free))
            device = torch.device(f"cuda:{available_gpu}")
            print(f"自动选择 GPU: {available_gpu} (可用内存: {max(memory_free)} MiB)")
        except Exception as e:
            print(f"无法自动选择 GPU: {e}. 使用默认设置 ({device}).")
    
    # model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.tokenizer_revision, torch_dtype=torch.bfloat16 if args.fp16 else torch.float32)
    # 使用绝对路径加载 Qwen 模型
    model = AutoModelForCausalLM.from_pretrained(
        "/data/shangyuan/models/DeepSeek-R1-Distill-Qwen-1.5B", 
        revision="main", 
        torch_dtype=torch.bfloat16 if args.fp16 else torch.float32
    )

    model = model.to(device)
    model.eval() # 设置为评估模式
    print(f"主 LLM 模型已加载到设备: {device}")

    # --- 根据模式执行不同流程 ---
    if args.active_learning and args.active_strategy == "rl" and args.rl_train or args.offline_rl:
        # --- 调用 RL 训练函数 ---
        print("开始 RL 策略训练流程...")
        # 传递 train_data_full 用于构建历史 (仅训练集), valid_data_full 用于交互
        train_rl_strategy(train_data_full, valid_data_full, model, tokenizer, args)
        print("RL 策略训练结束.")
    else:
        # --- 标准预测流程 (包括使用已训练好的RL策略进行预测) ---
        print("开始标准预测流程...")

        # --- 修改: 创建并初始化搜索空间 ---
        print("构建初始历史 (使用 train.txt + valid.txt)...")
        # 创建搜索空间
        entity_search_space = {}
        
        # 添加训练集数据
        for sample, _ in tqdm(train_data_full, desc="添加训练数据到历史"):
            entity, relation, targets, time = sample
            
            # 确保嵌套字典结构存在
            if entity not in entity_search_space:
                entity_search_space[entity] = {}
            if time not in entity_search_space[entity]:
                entity_search_space[entity][time] = {}
            if relation not in entity_search_space[entity][time]:
                entity_search_space[entity][time][relation] = []
            
            # 添加目标到搜索空间
            entity_search_space[entity][time][relation] += targets
        
        # 添加验证集数据
        for sample, _ in tqdm(valid_data_full, desc="添加验证数据到历史"):
            entity, relation, targets, time = sample
            
            # 确保嵌套字典结构存在
            if entity not in entity_search_space:
                entity_search_space[entity] = {}
            if time not in entity_search_space[entity]:
                entity_search_space[entity][time] = {}
            if relation not in entity_search_space[entity][time]:
                entity_search_space[entity][time][relation] = []
            
            # 添加目标到搜索空间
            entity_search_space[entity][time][relation] += targets
            
        print(f"初始历史构建完成 (包含train+valid). 搜索空间大小: {len(entity_search_space)}")
        # --- 结束修改 ---

        # 初始化评估指标
        metric = HitsMetric(args.top_k)
        results = {} # 存储预测结果

        # --- 主动学习策略初始化 (如果启用) ---
        active_learning_strategy = None
        if args.active_learning:
            print(f"初始化主动学习策略: {args.active_strategy}")
            active_learning_strategy = get_strategy(args.active_strategy, config_path=args.rl_config if args.active_strategy == 'rl' else None)
            # 如果是RL策略，需要加载训练好的模型，并设置必要的组件
            if args.active_strategy == 'rl':
                 # 移除原有的模型加载代码，现在由RLStrategy内部在select_samples中处理
                 print(f"RL策略 '{args.active_strategy}' 已初始化。模型将在首次选择样本时加载。")
                 pass # 保留if块结构，但内部逻辑已移到RLStrategy

        # 按时间戳处理测试数据
        # --- 新增：将测试数据列表转换为按时间戳分组的字典 ---
        print("按时间戳重新组织测试数据...")
        test_data_dict = {}
        if isinstance(test_data_full, list): # 检查是否确实是列表
            for sample_tuple in test_data_full:
                # 假设 test_data_full 中的每个元素是 (sample, direction, targets)
                # 其中 sample 是 (head, rel, target_list, time)
                if len(sample_tuple) >= 1 and len(sample_tuple[0]) >= 4:
                     timestamp = sample_tuple[0][3] # 获取时间戳
                     if timestamp not in test_data_dict:
                         test_data_dict[timestamp] = []
                     test_data_dict[timestamp].append(sample_tuple)
                else:
                     print(f"警告: 测试数据格式不符合预期，跳过: {sample_tuple}")
            test_data_full = test_data_dict # 用转换后的字典覆盖原来的列表
            print(f"测试数据已转换为字典格式，包含 {len(test_data_full)} 个时间戳。")
        elif not isinstance(test_data_full, dict):
            print(f"错误: test_data_full 既不是列表也不是字典，无法处理。类型: {type(test_data_full)}")
            # 可能需要在此处退出或引发错误
        # --- 转换结束 ---
        
        timestamps = sorted(list(test_data_full.keys()))

        for ts_idx, ts in enumerate(tqdm(timestamps, desc="按时间戳预测")):
            print(f"\n=== 处理时间戳 {ts} ({ts_idx+1}/{len(timestamps)}) ===")
            current_time_samples = test_data_full[ts]
            print(f"当前时间戳有 {len(current_time_samples)} 个样本")
            
            # 对当前时间戳的每个样本进行预测
            for item_idx, item in enumerate(tqdm(current_time_samples, desc=f"预测 TS {ts}", leave=False)):
                sample_tuple, direction = item
                sample = sample_tuple
                targets = sample[2]  # 目标列表
                
                # 为当前查询选择主动学习样本（这是新增的核心步骤）
                if args.active_learning and active_learning_strategy is not None and args.active_samples > 0:
                    print(f"\n--- 为查询 {item_idx+1}/{len(current_time_samples)} 选择辅助样本 ---")
                    print(f"查询: {sample[0]}, {sample[1]}, 时间戳: {sample[3]}")
                    
                    # 调用新的函数为当前查询选择样本
                    active_samples_for_query = select_samples_for_query(
                        query_sample=item,
                        current_time_samples=current_time_samples,
                        active_learning_strategy=active_learning_strategy,
                        num_samples_to_select=args.active_samples,
                        model=model,
                        tokenizer=tokenizer,
                        history=entity_search_space,
                        args=args
                    )
                    
                    print(f"已为当前查询选择 {len(active_samples_for_query)} 个辅助样本")
                else:
                    active_samples_for_query = []
                
                # 准备模型输入 (使用新的 prepare_input 签名，传递主动选择的样本)
                input_str, candidates, _ = prepare_input(
                    sample, 
                    entity_search_space, 
                    args, 
                    return_prompt=True,
                    active_samples=active_samples_for_query  # 新增参数，传递为当前查询选择的样本
                )

                # --- DEBUG: 打印 predict 函数的输入 ---\
                print(f"\n--- DEBUG: PREDICT INPUT (TS: {ts}, Sample: {sample[:2]}...) ---")
                print(f"DEBUG_INPUT_STR: {input_str}") # Keep this for context
                # +++ Add logging for full prompt +++
                print(f"DEBUG_FULL_PROMPT:\n{input_str}") 
                # +++ Add logging for candidates +++
                print(f"DEBUG_CANDIDATES_LIST (len={len(candidates)}): {candidates}") 
                print(f"DEBUG_CANDIDATES (first 10): {candidates[:10]}") # Keep this for brevity in main log
                # --- END DEBUG ---\

                # 获取搜索空间
                search_space = head_search_space.get((sample[1], sample[2][0], sample[3]), {}) if direction == "head" else tail_search_space.get((sample[0], sample[1], sample[3]), {})

                # 进行预测
                # 重命名 predict 的输出以明确其为元组列表
                ranked_candidates_tuples = predict(tokenizer, model, input_str, args)

                # --- 新增：将 (索引, 分数) 元组列表转换为实体字符串列表 ---
                ranked_entity_strings = []
                for idx, score in ranked_candidates_tuples:
                    # 检查索引是否在 candidates 列表的有效范围内
                    if 0 <= idx < len(candidates):
                        ranked_entity_strings.append(candidates[idx])
                    else:
                        # （可选）如果索引无效，可以记录一个警告
                        print(f"警告: 预测返回的索引 {idx} 超出候选列表范围 (长度 {len(candidates)})。")
                # --- 转换结束 ---

                # 更新评估指标 - 使用转换后的字符串列表
                example = {
                    "predictions": ranked_entity_strings, # <-- 使用正确的字符串列表
                    "targets": targets
                }
                update_metric(example, metric, args)
                
                # 存储结果时，可以选择存储字符串列表或原始元组列表
                # 存储字符串列表可能在输出文件中更易读
                results[f"{sample[0]}_{sample[1]}_{sample[2][0]}_{sample[3]}_{direction}"] = ranked_entity_strings 

            # --- 更新历史 (在线评估步骤) ---
            # 使用当前时间戳 ts 的所有真实样本更新历史
            for item in current_time_samples:
                 # 修改：支持 2 元素元组格式
                 sample_tuple, direction = item
                 # 从 sample_tuple 中提取数据：[entity, relation, targets_list, timestamp]
                 fact_to_update = (sample_tuple[0], sample_tuple[1], sample_tuple[2][0], sample_tuple[3])
                 
                 # 确保嵌套字典结构存在
                 if fact_to_update[0] not in entity_search_space:
                     entity_search_space[fact_to_update[0]] = {}
                 if fact_to_update[3] not in entity_search_space[fact_to_update[0]]:
                     entity_search_space[fact_to_update[0]][fact_to_update[3]] = {}
                 if fact_to_update[1] not in entity_search_space[fact_to_update[0]][fact_to_update[3]]:
                     entity_search_space[fact_to_update[0]][fact_to_update[3]][fact_to_update[1]] = []
                 
                 # 将事实添加到搜索空间
                 if fact_to_update[2] not in entity_search_space[fact_to_update[0]][fact_to_update[3]][fact_to_update[1]]:
                     entity_search_space[fact_to_update[0]][fact_to_update[3]][fact_to_update[1]].append(fact_to_update[2])


        # --- 预测结束，写入结果 ---
        output_filename_part = get_filename(args) # 获取不包含目录的文件名部分
        print(f"\n预测完成. 最终指标:")
        metric.print_metric()

        # 确保输出目录存在
        os.makedirs(args.output_dir, exist_ok=True)

        # 构建完整输出路径
        final_output_path = os.path.join(args.output_dir, output_filename_part)

        # 写入结果（取消注释）
        write_results(final_output_path, results, metric.get_metric(), args)
        print(f"详细结果已写入: {final_output_path}")

    print("脚本执行完毕.")
