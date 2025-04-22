import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, logging as tf_logging
import os

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
            
            # 加载BERT模型用于特征提取
            try:
                if not hasattr(strategy, 'bert_model') or strategy.bert_model is None or \
                   not hasattr(strategy, 'bert_tokenizer') or strategy.bert_tokenizer is None:
                    from transformers import AutoModel, AutoTokenizer
                    
                print(f"为RL训练加载BERT模型: {args.bert_model}")
                # 确保 bert_tokenizer 和 bert_model 使用 args.bert_model 加载
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
            except Exception as e:
                print(f"加载BERT模型时出错: {e}，RL 训练可能失败")
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
            environment = TKGEnvironment(
                interaction_data=valid_data, # 使用验证集进行交互
                state_model=strategy.bert_model,  # 使用BERT提取状态特征
                state_tokenizer=strategy.bert_tokenizer,
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
            agent = strategy._initialize_agent(environment, args, **agent_extra_params)
                
            if agent is None:
                print("警告: 无法初始化RL代理，训练已跳过")
                return
                
            # 设置策略的环境和代理
            strategy.environment = environment
            strategy.agent = agent
            
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
                 # --- 修改: 确保RL策略的环境也使用 train+valid 构建的初始历史 ---
                 # (或者，如果RL环境内部需要重新构建，应确保它也使用train+valid)
                 # 这里的实现是共享外部构建好的 history
                 if hasattr(active_learning_strategy, 'load_model') and hasattr(active_learning_strategy, 'agent') and active_learning_strategy.agent is not None:
                     try:
                         print("加载已训练的 RL 代理模型...")
                         active_learning_strategy.load_model()
                         # 设置必要的模型 (BERT for state, Qwen for reward/simulation)
                         if not hasattr(active_learning_strategy, 'bert_model') or not hasattr(active_learning_strategy, 'bert_tokenizer'):
                              from transformers import AutoModel, AutoTokenizer
                              print(f"为RL预测加载BERT模型: {args.bert_model}")
                              active_learning_strategy.bert_tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
                              active_learning_strategy.bert_model = AutoModel.from_pretrained(args.bert_model).to(device).eval()
                         # 确保环境引用了正确的模型和历史
                         if hasattr(active_learning_strategy, 'environment'):
                              if hasattr(active_learning_strategy.environment, 'reward_model'):
                                   active_learning_strategy.environment.reward_model = model
                                   active_learning_strategy.environment.reward_tokenizer = tokenizer
                              # 关键: 传递包含 train+valid 的历史给环境
                              active_learning_strategy.environment.train_history = entity_search_space
                              # 设置环境为评估模式
                              if hasattr(active_learning_strategy.environment, "set_mode"):
                                 active_learning_strategy.environment.set_mode("val")
                         else:
                              print("警告: RL策略缺少 'environment' 属性，无法设置历史和模式。")

                         print("RL 代理模型加载完成.")
                     except Exception as e:
                         print(f"加载或设置RL代理时出错: {e}. 主动学习可能无法正常工作.")
                         active_learning_strategy = None
                 else:
                     print("警告: RL 策略已选择，但无法找到加载模型的方法或代理未初始化。")
                     active_learning_strategy = None

        # 按时间戳处理测试数据
        timestamps = sorted(list(test_data_full.keys()))

        for ts_idx, ts in enumerate(tqdm(timestamps, desc="按时间戳预测")):
            current_time_samples = get_current_time_samples(test_data_full[ts])
            samples_to_predict = current_time_samples
            active_samples_info = []

            # --- 主动学习样本选择 ---
            if args.active_learning and active_learning_strategy is not None and len(current_time_samples) > args.active_samples:
                print(f"\n时间戳 {ts}: 执行主动学习选择 ({args.active_strategy})...")
                try:
                    # 传递当前的 history (包含 train+valid + T<ts 的 test)
                    selected_samples_indices, samples_to_predict_indices = active_learning_strategy.select_samples(
                        candidates=current_time_samples,
                        num_samples=args.active_samples,
                        model=model,
                        tokenizer=tokenizer,
                        history=entity_search_space, # 修改：传递搜索空间
                        args=args,
                    )
                    # ... (获取选中样本的逻辑不变) ...
                    selected_samples_for_context = []
                    for idx in selected_samples_indices:
                        sample, direction, targets = current_time_samples[idx]
                        formatted_sample = (sample[0], sample[1], targets[0], sample[3])
                        selected_samples_for_context.append( (formatted_sample, direction) )

                    active_samples_info = selected_samples_for_context
                    samples_to_predict = [current_time_samples[i] for i in samples_to_predict_indices]
                    print(f"主动学习完成: 选中 {len(active_samples_info)} 个样本, 剩余 {len(samples_to_predict)} 个待预测.")
                except Exception as e:
                    print(f"主动学习选择过程中出错: {e}. 跳过主动学习环节.")
                    active_samples_info = []
                    samples_to_predict = current_time_samples


            # --- 对剩余样本进行预测 ---
            for sample, direction, targets in tqdm(samples_to_predict, desc=f"预测 TS {ts}", leave=False):
                # 准备模型输入 (使用新签名)
                input_str, candidates, _ = prepare_input(sample, entity_search_space, args, return_prompt=True)

                # 获取搜索空间
                search_space = head_search_space.get((sample[1], sample[2][0], sample[3]), {}) if direction == "head" else tail_search_space.get((sample[0], sample[1], sample[3]), {})

                # 进行预测
                ranked_candidates = predict(tokenizer, model, input_str, args)

                # 更新评估指标 - 这里需要保持原有参数顺序
                update_metric(metric, ranked_candidates, targets)
                results[f"{sample[0]}_{sample[1]}_{sample[2][0]}_{sample[3]}_{direction}"] = ranked_candidates

            # --- 更新历史 (在线评估步骤) ---
            # 使用当前时间戳 ts 的所有真实样本更新历史
            for sample, direction, targets in current_time_samples:
                 fact_to_update = (sample[0], sample[1], targets[0], sample[3])
                 
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
        output_filename = get_filename(args)
        print(f"\n预测完成. 最终指标:")
        metric.print_metric()
        # write_results(output_filename, results, metric.get_metric(), args)
        print(f"详细结果保存在 {output_filename} (写入功能已注释)") # 避免意外写入

    print("脚本执行完毕.")
