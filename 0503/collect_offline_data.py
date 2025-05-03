# collect_offline_data.py

import argparse
import logging
import os
import random
from collections import deque # 如果需要在 rollout 内部处理状态序列
from os.path import join
from typing import List, Tuple

import torch
from tqdm import tqdm

# --- 1. 导入必要的模块 ---
from utils import load_data, get_filename, prepare_input, update_history, get_entity_relation_mappings
from src.rl.tkg_environment import TKGEnvironment
from src.rl.agents.replay import NamedTransition
from src.rl.plm_encoder import PLMEncoder
from src.rl.agents.random_agent import RandomAgent
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 2. 参数解析 ---
def parse_arguments():
    """
    解析命令行参数。
    复制或 adaptar run_hf.py 中的参数解析逻辑。
    确保包含以下必要参数:
    --dataset: 数据集名称 (例如 ICEWS18)
    --model: 主 LLM 模型路径 (如果奖励计算需要)
    --bert_model: BERT 模型路径 (用于状态编码)
    --history_len: 历史长度
    --history_type: 历史类型
    --history_direction: 历史方向
    --output_dir: 保存收集数据的目录
    --gpu: 使用的 GPU ID
    --seed: 随机种子
    --num_episodes_per_query: 每个查询收集多少个 episode (新参数)
    # 添加其他 TKGEnvironment 或历史构建可能需要的参数
    """
    parser = argparse.ArgumentParser()
    # --- 基本参数 ---
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., ICEWS18)")
    parser.add_argument("--model", type=str, help="Path to the main LLM model (if needed for reward)")
    parser.add_argument("--bert_model", type=str, required=True, help="Path to the BERT model for state encoding")
    parser.add_argument("--history_len", type=int, default=5, help="History length for context")
    parser.add_argument("--history_type", type=str, default="pair", help="Type of history")
    parser.add_argument("--history_direction", type=str, default="uni", help="Direction of history")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save collected transitions")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU ID to use (-1 for CPU)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # --- 随机数据收集相关参数 ---
    parser.add_argument("--num_episodes_per_query", type=int, default=5, 
                        help="Number of random episodes to collect per query (new parameter)")
    parser.add_argument("--max_rollout_steps", type=int, default=5, 
                        help="Max steps within a single rollout/episode") 
    parser.add_argument("--active_samples", type=int, default=5, 
                        help="Maximum number of samples to select for each query")
    parser.add_argument("--max_queries_per_timestamp", type=int, default=1000, 
                        help="Maximum number of queries to process per timestamp (limit for very large datasets)")
    
    # --- 其他从 run_hf.py 复制的必要参数 ---
    parser.add_argument("--text_style", action="store_true", help="Use text style for prompts")
    parser.add_argument("--label", action="store_true", help="Include labels in prompts")
    parser.add_argument("--no_time", action="store_true", help="Exclude time from prompts")
    parser.add_argument("--shuffle_history", action="store_true", help="Shuffle history before sampling")
    parser.add_argument("--no_entity", action="store_true", help="Exclude entity IDs from prompts")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 for model")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    # --- LLM生成相关参数 ---
    parser.add_argument("--max_length", type=int, default=1, help="Maximum length for text generation")
    parser.add_argument("--top_k", type=int, default=100, help="Top-k parameter for generation")
    parser.add_argument("--dec_cand", type=int, default=5, help="Number of candidates to decode")
    parser.add_argument("--do_sample", action="store_true", help="Whether to use sampling for generation")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for generation")
    
    # --- 环境参数 ---
    parser.add_argument("--state_repr", nargs='+', 
                        default=["query_features", "context_features", "interaction_features", "diversity_features", "curr_step"], 
                        help="State representation components")
    parser.add_argument("--reward_scale", type=float, default=10.0, help="Reward scaling factor")

    args = parser.parse_args()
    return args

# --- 新增：针对特定查询选择样本的辅助函数 ---
def select_samples_for_query(
    query_sample,
    current_time_samples,
    num_samples_to_select
):
    """
    为特定查询选择样本（随机选择）
    
    Args:
        query_sample: 当前需要预测的样本 (sample_tuple, direction)
        current_time_samples: 当前时间戳下的所有样本
        num_samples_to_select: 需要选择的样本数量
        
    Returns:
        list: 选中的样本列表，格式为 [(sample_tuple, direction), ...]
    """
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
        return candidate_samples, candidate_indices
    
    # 随机选择指定数量的样本
    selected_indices = random.sample(range(len(candidate_samples)), num_samples_to_select)
    return [candidate_samples[i] for i in selected_indices], [candidate_indices[i] for i in selected_indices]

# --- 3. 为特定查询执行随机 Rollout 函数 ---
def random_rollout(env: TKGEnvironment, query_sample, max_steps: int) -> List[NamedTransition]:
    """
    在一个环境中为特定查询执行一次完全随机的 rollout (episode)。

    Args:
        env: 已初始化并设置好当前时间戳上下文的 TKGEnvironment 实例。
        query_sample: 当前需要预测的查询样本 (sample_tuple, direction)。
        max_steps: 单个 rollout 的最大步数。

    Returns:
        本次 rollout 中收集到的 NamedTransition 对象列表。
    """
    collected_transitions = []
    try:
        # 设置当前查询
        env.update_query(query_sample)
        
        # 重置环境，但保持候选样本和查询不变
        state = env.reset(keep_candidates=True) 
        
        # 确保状态是张量而非元组
        if isinstance(state, tuple):
            state = state[0]  # 如果是元组，取第一个元素作为状态张量
        
        if state is None:
             logger.warning("Environment reset returned None state. Skipping rollout.")
             return []

        terminal = False
        past_states = [state] # 用于存储状态序列 (如果环境状态是序列式的)
        action_indices = []
        action_spaces = []
        rewards_log = [] # 记录原始奖励值

        steps_in_episode = 0
        while not terminal and steps_in_episode < max_steps:
            current_states = torch.stack(past_states) # 获取当前状态表示 (可能是序列)

            # 获取当前状态下的动作空间 (包含特征、实体ID等)
            action_space_tuple = env.action_space()
            if action_space_tuple is None:
                logger.warning("Action space is None. Terminating rollout early.")
                terminal = True # 无法选择动作，提前终止
                break

            # 从元组中提取动作特征或整个空间大小
            if isinstance(action_space_tuple, tuple) and len(action_space_tuple) == 3:
                 action_features, entity_ids, relation_ids = action_space_tuple
                 num_actions = action_features.shape[0]
            elif isinstance(action_space_tuple, torch.Tensor): # 兼容旧格式
                 action_features = action_space_tuple
                 num_actions = action_features.shape[0]
            else:
                 logger.error(f"Unexpected action space format: {type(action_space_tuple)}. Terminating.")
                 terminal = True
                 break

            if num_actions == 0:
                logger.warning("No actions available. Terminating rollout early.")
                terminal = True # 没有可选动作
                break

            # --- 关键: 完全随机选择动作 ---
            random_action_idx = random.randrange(num_actions)

            # 与环境交互
            # env.step 需要返回 next_state, reward, terminal
            # reward 应该是基于 MRR 等指标计算的真实奖励
            next_state_result = env.step(random_action_idx)
            
            # 处理可能的tuple和单值返回
            if isinstance(next_state_result, tuple):
                if len(next_state_result) == 3:
                    # 典型的(next_state, reward, terminal)格式
                    next_state, reward, terminal = next_state_result
                elif len(next_state_result) == 4:
                    # 可能是(next_state, reward, terminal, info)格式
                    next_state, reward, terminal, _ = next_state_result
                else:
                    logger.warning(f"Unexpected step return format with {len(next_state_result)} elements. Using defaults.")
                    next_state, reward, terminal = next_state_result[0], 0.0, True
            else:
                # 非元组返回，可能环境有问题
                logger.warning(f"Step returned non-tuple value: {type(next_state_result)}. Terminating.")
                next_state, reward, terminal = None, 0.0, True
            
            # 确保next_state是张量而非元组
            if isinstance(next_state, tuple):
                next_state = next_state[0]  # 如果是元组，取第一个元素作为状态张量

            # 记录信息用于构建 Transition
            rewards_log.append(reward)
            action_indices.append(random_action_idx)
            action_spaces.append(action_space_tuple) # 保存完整的动作空间

            if not terminal and next_state is not None:
                past_states.append(next_state)
            else:
                terminal = True  # 如果next_state为None，强制结束

            steps_in_episode += 1
            if terminal:
                 logger.debug(f"Episode finished after {steps_in_episode} steps. Terminal state reached.")
            elif steps_in_episode >= max_steps:
                 logger.debug(f"Episode reached max steps ({max_steps}). Terminating.")
                 terminal = True # 强制终止

        # --- 构建 Transitions ---
        # 处理非终止转换
        num_rewards = len(rewards_log)
        for i in range(num_rewards - 1):
            states_t = torch.stack(past_states[: i + 1]) # 截至 t 的状态序列
            action_idx_t = action_indices[i]
            action_space_t = action_spaces[i]
            next_states_t = torch.stack(past_states[: i + 2]) # 截至 t+1 的状态序列
            next_action_space_t = action_spaces[i + 1]
            reward_t = torch.tensor(rewards_log[i]) # 使用记录的奖励

            # 提取实体/关系 ID (如果可用且需要)
            action_entity_id = None
            action_relation_id = None
            if isinstance(action_space_t, tuple) and len(action_space_t) == 3:
                 _, entity_ids, relation_ids = action_space_t
                 if entity_ids and action_idx_t < len(entity_ids): action_entity_id = entity_ids[action_idx_t]
                 if relation_ids and action_idx_t < len(relation_ids): action_relation_id = relation_ids[action_idx_t]

            # 创建带有查询信息的转换对象
            t = NamedTransition(
                states_t,
                action_idx_t,
                action_space_t,
                next_states_t,
                next_action_space_t,
                reward_t,
                action_entity_id,
                action_relation_id,
                query_info=query_sample  # 添加查询信息，确保知道此转换是针对哪个查询的
            )
            collected_transitions.append(t)

        # 处理最后一个（可能是终止）转换
        if num_rewards > 0:
            states_last = torch.stack(past_states)
            action_idx_last = action_indices[-1]
            action_space_last = action_spaces[-1]
            reward_last = torch.tensor(rewards_log[-1])

            # 提取实体/关系 ID
            action_entity_id_last = None
            action_relation_id_last = None
            if isinstance(action_space_last, tuple) and len(action_space_last) == 3:
                 _, entity_ids, relation_ids = action_space_last
                 if entity_ids and action_idx_last < len(entity_ids): action_entity_id_last = entity_ids[action_idx_last]
                 if relation_ids and action_idx_last < len(relation_ids): action_relation_id_last = relation_ids[action_idx_last]

            # 终止状态没有 next_state 和 next_action_space
            t_last = NamedTransition(
                states_last,
                action_idx_last,
                action_space_last,
                None, # next_state is None for terminal
                None, # next_action_space is None for terminal
                reward_last,
                action_entity_id_last,
                action_relation_id_last,
                query_info=query_sample  # 添加查询信息
            )
            collected_transitions.append(t_last)

    except Exception as e:
        logger.error(f"Error during random rollout for query {query_sample[0][:2]}: {e}", exc_info=True)

    return collected_transitions

# --- 4. 主函数 ---
def main():
    args = parse_arguments()

    # 设置随机种子
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 设置设备
    if args.gpu == -2 and torch.cuda.is_available():
        # 特殊情况：gpu=-2 时自动选择最合适的 CUDA 设备
        device = torch.device("cuda")  # 使用默认 CUDA 设备
        torch.cuda.manual_seed_all(args.seed)
        logger.info(f"Using automatically selected CUDA device: {torch.cuda.get_device_name(device)}")
    elif args.gpu >= 0 and torch.cuda.is_available():
        # 常规情况：使用指定的 GPU
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.manual_seed_all(args.seed)
        logger.info(f"Using specified CUDA device: {torch.cuda.get_device_name(device)}")
    else:
        # 无法使用 GPU 的情况
        device = torch.device("cpu")
        logger.info("Using CPU device")

    # --- 加载数据 ---
    logger.info(f"Loading data for dataset: {args.dataset}")
    # 加载训练数据和验证数据
    # 训练数据用于构建历史上下文/搜索空间
    # 验证数据用于交互和收集数据
    try:
        train_data, entity_search_space, _ = load_data(args, mode="train")
        valid_data, _, _ = load_data(args, mode="valid")
        # 创建全局实体和关系ID映射
        entity_ids, relation_ids = get_entity_relation_mappings(train_data + valid_data)
        
        logger.info(f"Loaded {len(train_data)} train samples, {len(valid_data)} valid samples.")
        logger.info(f"Found {len(entity_ids)} entities and {len(relation_ids)} relations.")
    except Exception as e:
        logger.error(f"Failed to load data: {e}", exc_info=True)
        return

    # --- 初始化模型和分词器 ---
    logger.info(f"Loading BERT model from {args.bert_model}")
    bert_tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    bert_model = AutoModel.from_pretrained(args.bert_model).to(device)
    plm_encoder = PLMEncoder(bert_model, bert_tokenizer, model_type="encoder")
    
    # 如果提供了LLM模型路径，则加载LLM模型（用于奖励计算）
    reward_model = None
    reward_tokenizer = None
    if args.model:
        try:
            logger.info(f"Loading LLM model from {args.model} for reward calculation")
            from transformers import AutoModelForCausalLM
            reward_tokenizer = AutoTokenizer.from_pretrained(args.model)
            
            # 修改模型加载方式，强制使用单个GPU
            if args.gpu >= 0 and torch.cuda.is_available():
                # 指定单个GPU时，直接加载到该GPU上
                reward_model = AutoModelForCausalLM.from_pretrained(
                    args.model, 
                    torch_dtype=torch.float16 if args.fp16 else torch.float32,
                    device_map=None  # 禁用自动分配
                ).to(device)  # 明确移动到指定设备
            elif args.gpu == -2 and torch.cuda.is_available():
                # 自动选择GPU时，也只用一个GPU
                reward_model = AutoModelForCausalLM.from_pretrained(
                    args.model, 
                    torch_dtype=torch.float16 if args.fp16 else torch.float32,
                    device_map=None  # 禁用自动分配
                ).to(device)  # 明确移动到之前选择的设备
            else:
                # CPU模式
                reward_model = AutoModelForCausalLM.from_pretrained(
                    args.model, 
                    torch_dtype=torch.float16 if args.fp16 else torch.float32,
                    device_map=None  # 禁用自动分配
                ).to(device)  # 明确移动到CPU
                
            logger.info(f"Successfully loaded LLM model for reward calculation on {device}")
        except Exception as e:
            logger.error(f"Failed to load LLM model: {e}", exc_info=True)
            logger.warning("Will continue without LLM model, but rewards may not be calculated properly")
    
    # --- 确保输出目录存在 ---
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_dir = join(args.output_dir, "ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # --- 初始化环境 ---
    logger.info("Initializing TKG environment")
    environment = TKGEnvironment(
        interaction_data=valid_data,  # 使用验证集进行交互
        state_model=bert_model,      # 使用BERT提取状态特征
        state_tokenizer=bert_tokenizer,
        reward_model=reward_model,    # 使用LLM计算奖励
        reward_tokenizer=reward_tokenizer,
        train_history=entity_search_space,  # 传递搜索空间用于构建历史上下文
        args=args,
        state_repr=args.state_repr,
        max_steps=args.active_samples,
        reward_scale=args.reward_scale,
        predef_entity_ids=entity_ids,
        predef_relation_ids=relation_ids,
        state_model_type="encoder"  # 指定状态模型类型为encoder (BERT)
    )
    
    # --- 初始化随机代理 ---
    logger.info("Initializing random agent")
    random_agent = RandomAgent(
        env=environment,
        output_dir=args.output_dir,
        train_steps=1,  # 此值不重要，我们会手动控制训练步数
        save_every=1    # 确保每次rollout后保存
    )
    
    # --- 设置环境模式为训练模式 ---
    environment.set_mode("train")
    
    # --- 获取所有时间戳 ---
    timestamps = sorted(list(set([sample[3] for sample, _ in valid_data])))
    logger.info(f"Found {len(timestamps)} unique timestamps in validation data")
    
    # --- 按时间戳和查询收集数据 ---
    all_transitions = []
    total_steps = 0  # 用于跟踪总步数
    try:
        for ts_idx, timestamp in enumerate(tqdm(timestamps, desc="Processing timestamps")):
            # 获取当前时间戳的样本
            current_samples = [(sample, direction) for sample, direction in valid_data if sample[3] == timestamp]
            
            if not current_samples:
                logger.warning(f"No samples found for timestamp {timestamp}, skipping")
                continue
                
            # 设置样本数量限制（针对非常大的数据集）
            num_queries = min(len(current_samples), args.max_queries_per_timestamp)
            logger.info(f"Processing timestamp {timestamp} ({ts_idx+1}/{len(timestamps)}), processing {num_queries}/{len(current_samples)} queries")
            
            # 更新环境的当前时间戳样本
            indices = list(range(len(current_samples)))
            environment.update_candidates(current_samples, indices)
            
            # --- 核心修改：针对每个查询收集样本 ---
            timestamp_transitions = []
            
            # 对当前时间戳的每个查询样本进行处理
            for query_idx, query_sample in enumerate(tqdm(current_samples[:num_queries], desc=f"Processing queries for timestamp {timestamp}")):
                query_entity, query_relation = query_sample[0][0], query_sample[0][1]
                logger.info(f"Processing query {query_idx+1}/{num_queries}: {query_entity}, {query_relation}")
                
                # 为当前查询收集多个 episodes
                query_transitions = []
                for ep_idx in range(args.num_episodes_per_query):
                    # 执行一次针对当前查询的随机rollout
                    transitions = random_rollout(environment, query_sample, args.max_rollout_steps)
                    if transitions:
                        query_transitions.extend(transitions)
                        logger.debug(f"Collected {len(transitions)} transitions in episode {ep_idx+1} for query {query_idx+1}")
                    else:
                        logger.warning(f"No transitions collected in episode {ep_idx+1} for query {query_idx+1}")
                
                # 添加到时间戳的收集中
                timestamp_transitions.extend(query_transitions)
            
            # 添加到总体收集中
            all_transitions.extend(timestamp_transitions)
            total_steps += 1  # 每个时间戳算作一步
            
    except Exception as e:
        logger.error(f"Error during data collection: {e}", exc_info=True)
        # 如果发生错误，尽量保存已收集的数据
        if all_transitions:
            emergency_path = join(ckpt_dir, "transitions_emergency.ckpt")
            torch.save(all_transitions, emergency_path)
            logger.info(f"Saved {len(all_transitions)} transitions to emergency file {emergency_path}")
    
    # --- 保存所有收集的数据 ---
    if all_transitions:
        # 保存最终的聚合数据
        final_path = join(ckpt_dir, "transitions_all.ckpt")
        torch.save(all_transitions, final_path)
        logger.info(f"Successfully collected and saved {len(all_transitions)} transitions to {final_path}")
        
        # 创建随机代理的transitions，然后保存，这样可以直接被DQNAgent加载
        random_agent.transitions = all_transitions
        random_agent.curr_step = total_steps
        random_agent.save_checkpoints()
        logger.info(f"Saved transitions using RandomAgent format at step {random_agent.curr_step}")
    else:
        logger.error("No transitions were collected!")
    
    logger.info("Data collection completed")

if __name__ == "__main__":
    main()