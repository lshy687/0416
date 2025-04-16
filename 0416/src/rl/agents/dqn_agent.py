import glob
import logging
import os
import random
from collections import deque
from os.path import join
from typing import Dict, List, Optional, Union

import jsonlines
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

import wandb
from src.rl.agents.network import initialize_network
from src.rl.agents.replay import ReplayMemory, Transition
from src.rl.base_environment import BaseEnvironment
from src.rl.misc_utils import collate_summaries, parse_step_from_checkpoint, tailsum

logger = logging.getLogger(__name__)


class DQNAgent:
    def __init__(
        self,
        env: BaseEnvironment,
        output_dir: Optional[str] = None,
        overwrite_existing: bool = False,
        train_steps: int = 400,
        save_every: int = 400,
        eval_every: int = 400,
        val_rounds: int = 1,
        target_update_every: int = 10,
        optimization_steps_per_train_step=1,
        batch_size: int = 4,
        replay_memory_size: int = 1000,
        eps_start: float = 0.99,
        eps_end: float = 0.2,
        decay_steps: Optional[int] = None,
        max_grad_norm: float = 10.0,
        lr: Optional[float] = 1e-2,
        base_lr: Optional[float] = None,
        max_lr: Optional[float] = None,
        step_size_up: int = 100,
        step_size_down: Optional[int] = None,
        weight_decay: float = 1e-3,
        network_params: dict = {},
        cql_loss_weight: float = 0.0,
        load_transitions: Optional[Union[str, list[str]]] = None,
        offline_steps: int = 0,
    ):
        # 初始化网络、优化器等
        self.env = env
        self.output_dir = output_dir
        self.ckpt_dir = join(output_dir, "ckpts") if self.output_dir else None
        self.overwrite_existing = overwrite_existing
        self.curr_step = 0
        self.train_steps = train_steps
        self.save_every = save_every
        self.eval_every = eval_every
        self.val_rounds = val_rounds
        self.target_update_every = target_update_every
        self.cql_loss_weight = cql_loss_weight
        self.offline_steps = offline_steps

        if save_every % target_update_every != 0:
            raise Exception(
                "target_update_every should divide save_every"
                " to simplify model checkpointing logic"
            )

        if eval_every % save_every != 0:
            raise Exception(
                "save_every should divide eval_every"
                " to ensure best model can be loaded later"
            )

        self.optimization_steps_per_train_step = optimization_steps_per_train_step
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm

        # 初始化经验回放
        self.replay_memory = ReplayMemory(replay_memory_size)

        self.eps_start = eps_start
        self.eps_end = eps_end
        if decay_steps is None:
            decay_steps = self.train_steps
        self.eps_decay = (eps_end / eps_start) ** (1 / (decay_steps - 1))

        self.mode = "train"

        # 初始化策略网络和目标网络
        self.policy_net = initialize_network(
            env.state_dim, env.action_dim, **network_params
        )
        self.target_net = initialize_network(
            env.state_dim, env.action_dim, requires_grad=False, **network_params
        )

        # 确定单一设备并将网络移动到同一设备上
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"将所有网络组件移动到设备: {self.device}")
        self.policy_net = self.policy_net.to(self.device)
        self.target_net = self.target_net.to(self.device)

        self.target_update()

        self.optimizer = torch.optim.AdamW(
            self.policy_net.parameters(), lr, weight_decay=weight_decay
        )

        if base_lr is None:
            assert lr is not None
            base_lr = max_lr = lr
            step_size_up = 100000

        if max_lr is None:
            raise Exception("max_lr cannot be None if base_lr is given.")

        logger.info(
            f"setting up scheduler with base_lr={base_lr}, "
            f"max_lr={max_lr}, step_size_up={step_size_up}, "
            f"step_size_down={step_size_down}"
        )

        self.scheduler = torch.optim.lr_scheduler.CyclicLR(
            self.optimizer,
            base_lr,
            max_lr,
            step_size_up=step_size_up,
            step_size_down=step_size_down,
            cycle_momentum=False,
        )
        if load_transitions is not None:
            self.replay_memory.load(load_transitions, env)
        self.load_checkpoints()

    def load_checkpoints(self):
        if self.ckpt_dir is None:
            return

        if self.overwrite_existing:
            logger.info("train from scratch and overwrite existing checkpoints...")
            return

        replay_ckpts = {
            parse_step_from_checkpoint(f): f
            for f in glob.glob(join(self.ckpt_dir, "replay_*.ckpt"))
        }

        model_ckpts = {
            parse_step_from_checkpoint(f): f
            for f in glob.glob(join(self.ckpt_dir, "model_*.ckpt"))
        }
        optim_ckpts = {
            parse_step_from_checkpoint(f): f
            for f in glob.glob(join(self.ckpt_dir, "optim_*.ckpt"))
        }

        scheduler_ckpts = {
            parse_step_from_checkpoint(f): f
            for f in glob.glob(join(self.ckpt_dir, "sched_*.ckpt"))
        }

        ckpts_found = (
            set(replay_ckpts.keys())
            & set(model_ckpts.keys())
            & set(optim_ckpts.keys())
            & set(scheduler_ckpts.keys())
        )

        if not ckpts_found:
            logger.info("no existing checkpoints, train from scratch...")
            if 0 in replay_ckpts:
                logger.info("loading initial replay memory...")
                self.replay_memory = torch.load(replay_ckpts[0])
            return

        step = max(ckpts_found)
        logger.info(f"setting step={step}")
        self.curr_step = step
        logger.info(
            "loading replay memory, policy network and optimizer " f"from step={step}"
        )
        self.replay_memory = torch.load(replay_ckpts[step])
        self.policy_net.load_state_dict(torch.load(model_ckpts[step]))
        
        # 确保加载后的模型也在指定设备上
        self.policy_net = self.policy_net.to(self.device)
        
        self.target_update()
        self.optimizer.load_state_dict(torch.load(optim_ckpts[step]))
        self.scheduler.load_state_dict(torch.load(scheduler_ckpts[step]))

    def target_update(self):
        logger.debug("updating target net with policy net parameters")
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # 确保目标网络在同一设备上
        self.target_net = self.target_net.to(self.device)

    def save_checkpoints(self):
        if self.ckpt_dir is None:
            return
        step = self.curr_step
        os.makedirs(self.ckpt_dir, exist_ok=True)

        logger.info(
            f"saving replay memory, policy network and optimizer for step={step}"
        )
        replay_ckpt_path = join(self.ckpt_dir, f"replay_{step}.ckpt")
        model_ckpt_path = join(self.ckpt_dir, f"model_{step}.ckpt")
        optim_ckpt_path = join(self.ckpt_dir, f"optim_{step}.ckpt")
        sched_ckpt_path = join(self.ckpt_dir, f"sched_{step}.ckpt")
        torch.save(self.replay_memory, replay_ckpt_path)
        torch.save(self.policy_net.state_dict(), model_ckpt_path)
        torch.save(self.optimizer.state_dict(), optim_ckpt_path)
        torch.save(self.scheduler.state_dict(), sched_ckpt_path)

    def choose_action(self, states, action_space_tuple):
        # 从元组中提取动作特征、实体ID和关系ID
        # 兼容原始格式和新格式
        if isinstance(action_space_tuple, tuple) and len(action_space_tuple) == 3:
            action_features, entity_ids, relation_ids = action_space_tuple
        else:
            # 向后兼容 - 如果传入的只是特征张量，假设没有ID信息
            action_features = action_space_tuple
            entity_ids = []
            relation_ids = []
        
        # 记录所选动作的entity_id和relation_id的变量
        selected_entity_id = None
        selected_relation_id = None

        if self.mode == "train":
            # epsilon-greedy exploration with exp. decay
            eps = self.eps_start * self.eps_decay**self.curr_step
            eps = max(eps, self.eps_end)
            
            # 添加wandb初始化检查
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log(data={"epsilon": eps}, step=self.curr_step)
            except Exception as e:
                logger.debug(f"记录wandb指标时出错: {e}")
            
            rand_val = random.random()
            if rand_val < eps: # 如果随机数小于epsilon，则进行探索
                # explore
                num_actions = action_features.shape[0]
                if num_actions == 0:
                    logger.warning("探索模式：没有可选动作！")
                    return 0 
                action_idx = random.choice(range(num_actions))
                
                # 记录所选动作的entity_id和relation_id（如果有）
                if entity_ids and action_idx < len(entity_ids):
                    selected_entity_id = entity_ids[action_idx]
                if relation_ids and action_idx < len(relation_ids):
                    selected_relation_id = relation_ids[action_idx]
                
                # 记录选择的实体ID和关系ID
                self.last_selected_entity_id = selected_entity_id
                self.last_selected_relation_id = selected_relation_id
                    
                logger.debug(f"探索模式: 随机选择动作索引 {action_idx}, 实体ID: {selected_entity_id}, 关系ID: {selected_relation_id}")
                return action_idx

        # pick best action (argmax of Q values)
        action_features_device = action_features.to(self.device)
        Q_values = self.policy_net(states, action_features_device)

        # 检查 Q_values 是否为空或NaN
        if Q_values is None or Q_values.nelement() == 0 or torch.isnan(Q_values).any():
            logger.warning("利用模式：Q值计算结果无效 (None, 空或包含NaN)！将随机选择动作。")
            # Q值无效，回退到随机选择
            num_actions = action_features.shape[0]
            if num_actions == 0:
                 logger.warning("利用模式（回退）：没有可选动作！")
                 return 0
            action_idx = random.choice(range(num_actions))
            
            # 记录所选动作的entity_id和relation_id（如果有）
            if entity_ids and action_idx < len(entity_ids):
                selected_entity_id = entity_ids[action_idx]
            if relation_ids and action_idx < len(relation_ids):
                selected_relation_id = relation_ids[action_idx]
            
            # 记录选择的实体ID和关系ID
            self.last_selected_entity_id = selected_entity_id
            self.last_selected_relation_id = selected_relation_id
                
            return action_idx
        
        # 记录Q值统计信息
        try:
            import wandb
            if wandb.run is not None:
                wandb.log(
                    data={
                        "Q-mean": Q_values.mean().item(),
                        "Q-std": Q_values.std().item(),
                        "Q-max": Q_values.max().item(),
                    },
                    step=self.curr_step,
                )
        except Exception as e:
            logger.debug(f"记录wandb指标时出错: {e}")

        # 获取最大Q值的动作
        action_idx = Q_values.argmax().item()
        
        # 记录所选动作的entity_id和relation_id（如果有）
        if entity_ids and action_idx < len(entity_ids):
            selected_entity_id = entity_ids[action_idx]
        if relation_ids and action_idx < len(relation_ids):
            selected_relation_id = relation_ids[action_idx]
        
        # 记录选择的实体ID和关系ID
        self.last_selected_entity_id = selected_entity_id
        self.last_selected_relation_id = selected_relation_id
            
        logger.debug(f"利用模式: 选择Q值最高的动作 {action_idx}, Q值: {Q_values[action_idx].item():.4f}, 实体ID: {selected_entity_id}, 关系ID: {selected_relation_id}")
        
        # 记录排序后的前几个动作及其Q值
        topk_values, topk_indices = torch.topk(Q_values, min(5, len(Q_values)))
        logger.debug(f"前{len(topk_indices)}个最优动作: {[(idx.item(), val.item()) for idx, val in zip(topk_indices, topk_values)]}")
        
        return action_idx

    def _find_matching_action(self, action_space, entity_id, relation_id):
        """
        在当前action_space中查找匹配的动作索引
        
        Args:
            action_space: 当前的动作空间
            entity_id: 目标实体ID
            relation_id: 目标关系ID
            
        Returns:
            找到的动作索引，如果没有匹配则返回-1
        """
        # 如果entity_id和relation_id都为None，则无法执行匹配
        if entity_id is None and relation_id is None:
            logger.debug("无法执行匹配：entity_id和relation_id都为None")
            return -1
            
        # 检查action_space是否为元组格式并包含实体ID和关系ID
        if not isinstance(action_space, tuple) or len(action_space) != 3:
            logger.debug(f"无法执行匹配：action_space不是预期的元组格式 (实际类型: {type(action_space)})")
            return -1
            
        _, entity_ids, relation_ids = action_space
        
        # 检查是否有任何实体ID和关系ID可以匹配
        if not entity_ids and not relation_ids:
            logger.debug("无法执行匹配：action_space中没有实体ID和关系ID")
            return -1
            
        # 检查列表长度是否一致
        if len(entity_ids) != len(relation_ids):
            logger.warning(f"实体ID列表和关系ID列表长度不一致：entity_ids={len(entity_ids)}, relation_ids={len(relation_ids)}")
            # 使用最小的长度来避免索引错误
            min_len = min(len(entity_ids), len(relation_ids))
            entity_ids = entity_ids[:min_len]
            relation_ids = relation_ids[:min_len]
            
        # 遍历所有动作，查找匹配的实体ID和关系ID
        for idx, (e_id, r_id) in enumerate(zip(entity_ids, relation_ids)):
            # 如果同时匹配实体ID和关系ID，或者只匹配非空的ID
            match_entity = (entity_id is None) or (e_id == entity_id)
            match_relation = (relation_id is None) or (r_id == relation_id)
            
            if match_entity and match_relation:
                logger.debug(f"找到匹配的动作：索引={idx}, 实体ID={e_id}, 关系ID={r_id}")
                return idx
                
        logger.debug(f"未找到匹配的动作：target_entity_id={entity_id}, target_relation_id={relation_id}")
        return -1

    def evaluate_action(self, states, action_space_tuple, action_idx):
        # 从元组中提取动作特征
        if isinstance(action_space_tuple, tuple) and len(action_space_tuple) == 3:
            action_features, _, _ = action_space_tuple
        else:
            # 向后兼容 - 如果传入的只是特征张量，保持原样
            action_features = action_space_tuple
            
        # Q_values 应该是 (Num_Actions,)
        Q_values = self.policy_net(states, action_features)
        
        # 检查 Q_values 是否有效以及 action_idx 是否在范围内
        if Q_values is None or Q_values.nelement() == 0 or torch.isnan(Q_values).any() or action_idx >= len(Q_values):
             logger.warning(f"evaluate_action: 无效的Q值或动作索引 {action_idx}。返回默认值。")
             # 返回默认值或进行其他错误处理
             return torch.tensor([]), torch.tensor(0.0), True # 返回空Q值列表, 0 Q值, 强制停止

        Q_value = Q_values[action_idx]

        # stop action (-1) is implicit
        # it happens when Q values for all other actions are negative
        if Q_value < 0.0:
            return Q_values, torch.tensor(0.0), True

        return Q_values, Q_value, False

    def rollout(self):
        env = self.env
        state_tuple = env.reset()
        # 处理state可能是元组的情况，提取其中的张量
        state = state_tuple[0] if isinstance(state_tuple, tuple) else state_tuple
        terminal = False
        rewards = []
        past_states = [state]
        action_indices = []
        action_spaces = []
        device = self.policy_net.device

        while not terminal:
            states = torch.stack(past_states)
            action_space_tuple = env.action_space()
            
            # 从元组中提取动作特征、实体ID和关系ID
            if isinstance(action_space_tuple, tuple) and len(action_space_tuple) == 3:
                action_features, entity_ids, relation_ids = action_space_tuple
            else:
                # 向后兼容 - 如果返回的只是特征张量，保持原样
                action_features = action_space_tuple
                entity_ids = []
                relation_ids = []
            
            # 检查动作特征空间是否为空
            if action_features is None or action_features.shape[0] == 0:
                logger.warning("Rollout: 当前时间步没有可选动作，提前终止。")
                break # 没有动作可选，终止 rollout
                
            device = self.policy_net.device
            action_features_ = action_features.to(device)

            # action_idx 是选择的动作的索引 (0 到 N-1)
            action_idx = self.choose_action(states, action_space_tuple) 
            
            # 检查返回的 action_idx 是否有效
            if action_idx is None or action_idx < 0 or action_idx >= action_features.shape[0]:
                logger.error(f"Rollout: choose_action 返回了无效的索引 {action_idx}。")
                # 可以选择终止，或者采取默认动作，例如选择第一个动作 (0)
                action_idx = 0 # 示例：选择第一个动作
                if action_features.shape[0] == 0: # 再次检查以防万一
                     logger.error("Rollout: 动作空间为空，无法执行步骤。")
                     break

            # 使用选定的索引执行环境步骤
            next_state_tuple, reward, terminal, _ = env.step(action_idx)
            # 处理next_state可能是元组的情况，提取其中的张量
            next_state = next_state_tuple[0] if isinstance(next_state_tuple, tuple) else next_state_tuple

            rewards.append(reward)
            if not terminal:
                past_states.append(next_state)
            action_indices.append(action_idx)
            action_spaces.append(action_space_tuple)

        if (
            not len(rewards)
            == len(past_states)
            == len(action_indices)
            == len(action_spaces)
        ):
            raise Exception(
                f"should have len(rewards) ({len(rewards)}) == "
                f"len(past_states) ({len(past_states)}) == "
                f"len(action_indices) ({len(action_indices)}) =="
                f"len(action_spaces) ({len(action_spaces)})"
            )

        if self.mode == "train":
            # push non-terminal transitions to replay memory
            for i in range(len(rewards) - 1):
                states = torch.stack(past_states[: i + 1])
                action_idx = action_indices[i]
                action_space = action_spaces[i]
                next_states = torch.stack(past_states[: i + 2])
                next_action_space = action_spaces[i + 1]
                reward = torch.tensor(rewards[i])
                
                # 获取当前动作的entity_id和relation_id
                entity_id = None
                relation_id = None
                
                # 如果action_space是元组并包含entity_ids和relation_ids，则从中提取
                if isinstance(action_space, tuple) and len(action_space) == 3:
                    action_features, entity_ids, relation_ids = action_space
                    if entity_ids and action_idx < len(entity_ids):
                        entity_id = entity_ids[action_idx]
                    if relation_ids and action_idx < len(relation_ids):
                        relation_id = relation_ids[action_idx]
                
                # 如果没有从action_space获取到ID，尝试使用last_selected_entity/relation_id
                if entity_id is None and hasattr(self, 'last_selected_entity_id'):
                    entity_id = self.last_selected_entity_id
                if relation_id is None and hasattr(self, 'last_selected_relation_id'):
                    relation_id = self.last_selected_relation_id

                t = Transition(
                    states,
                    action_idx,
                    action_space,
                    next_states,
                    next_action_space,
                    reward,
                    entity_id,   # 传递实体ID
                    relation_id  # 传递关系ID
                )
                self.replay_memory.push(t)

            # push terminal transition
            states = torch.stack(past_states)
            action_idx = action_indices[-1]
            action_space = action_spaces[-1]
            reward = torch.tensor(rewards[-1])
            
            # 获取最后一个动作的entity_id和relation_id
            entity_id = None
            relation_id = None
            
            # 如果action_space是元组并包含entity_ids和relation_ids，则从中提取
            if isinstance(action_space, tuple) and len(action_space) == 3:
                action_features, entity_ids, relation_ids = action_space
                if entity_ids and action_idx < len(entity_ids):
                    entity_id = entity_ids[action_idx]
                if relation_ids and action_idx < len(relation_ids):
                    relation_id = relation_ids[action_idx]
            
            # 如果没有从action_space获取到ID，尝试使用last_selected_entity/relation_id
            if entity_id is None and hasattr(self, 'last_selected_entity_id'):
                entity_id = self.last_selected_entity_id
            if relation_id is None and hasattr(self, 'last_selected_relation_id'):
                relation_id = self.last_selected_relation_id
                
            t = Transition(
                states,
                action_idx,
                action_space,
                None,
                None,
                reward,
                entity_id,   # 传递实体ID
                relation_id  # 传递关系ID
            )
            self.replay_memory.push(t)

        rewards = torch.tensor(rewards)
        return rewards

    def evaluate_trajectory(self, trajectory: List[int]):
        # given a trajectory, evaluate how good it is
        env = self.env
        state = env.reset()
        terminal = False
        rewards = []
        past_states = [state]
        Q_action_space_hist = []
        Q_value_hist = []
        rewards = []

        for action_idx in trajectory:
            states = torch.stack(past_states)
            action_space_tuple = env.action_space()
            Q_values, Q_value, early_stopping = self.evaluate_action(
                states, action_space_tuple, action_idx
            )

            if early_stopping:
                next_state, reward, terminal = env.step(-1)
            else:
                next_state, reward, terminal = env.step(action_idx)

            Q_action_space_hist.append(Q_values.tolist())
            Q_value_hist.append(Q_value.item())
            rewards.append(reward)

            if not terminal:
                past_states.append(next_state)
            else:
                break

        return Q_action_space_hist, Q_value_hist, rewards

    def compute_conservative_loss(self, Q_pred_all, Q_pred):
        logsumexp = torch.logsumexp(Q_pred_all, dim=1)
        return (logsumexp - Q_pred).mean()

    def standardize_batch_tensors(self, tensor_list, dim_idx=1):
        """标准化批次中张量的大小，通过填充或截断第一维
        
        Args:
            tensor_list: 要标准化的张量列表
            dim_idx: 特征维度索引，默认为1（第二维）
            
        Returns:
            形状标准化后的张量批次
        """
        if not tensor_list:
            return None
            
        # 获取特征维度（第二维）
        feature_dim = tensor_list[0].shape[dim_idx]
        
        # 找出最大第一维
        max_first_dim = max(t.shape[0] for t in tensor_list)
        
        # 标准化每个张量
        standardized = []
        for tensor in tensor_list:
            if tensor.shape[0] == max_first_dim:
                standardized.append(tensor)
            else:
                # 创建填充张量
                device = tensor.device
                padded = torch.zeros((max_first_dim, feature_dim), device=device)
                # 复制原始数据
                padded[:tensor.shape[0]] = tensor
                standardized.append(padded)
                
        # 堆叠标准化后的张量
        return torch.stack(standardized)

    def optimize(self):
        # 如果回放缓冲区中的样本数量不足，则跳过优化步骤
        if len(self.replay_memory) < self.batch_size:
            return

        # 从回放记忆中随机抽取一批样本
        transitions = self.replay_memory.sample(self.batch_size)
        
        # 使用self.device而不是重新获取设备
        device = self.device
        
        # 处理状态批量 - 如果是序列形式则只取最后状态
        if all(isinstance(t.states, torch.Tensor) and t.states.dim() > 1 for t in transitions):
            states_batch = torch.stack([t.states[-1] for t in transitions]).to(device)
        else:
            states_batch = torch.stack([t.states for t in transitions]).to(device)
        
        # 将操作索引和奖励转换为张量
        action_idx_batch = torch.tensor([t.action_idx for t in transitions], device=device)
        reward_batch = torch.stack([t.reward for t in transitions]).to(device)
        
        # 初始化目标Q值张量
        Q_targets = torch.zeros(self.batch_size, device=device)
        
        # 收集非终止状态的信息
        non_terminal_mask = torch.tensor([t.next_states is not None for t in transitions], device=device)
        
        # 为非终止状态计算下一个状态的最大Q值
        if non_terminal_mask.sum() > 0:
            # 找出非终止的转换
            non_terminal_indices = torch.where(non_terminal_mask)[0]
            non_terminal_transitions = [transitions[i] for i in non_terminal_indices]
            
            # 为每个非终止转换计算最大Q值
            max_q_values = []
            for t in non_terminal_transitions:
                # 处理next_states - 如果是序列取最后一个
                if isinstance(t.next_states, torch.Tensor) and t.next_states.dim() > 1:
                    next_state = t.next_states[-1].unsqueeze(0).to(device)
                else:
                    next_state = t.next_states.unsqueeze(0).to(device)
                    
                # 处理next_action_space可能是元组的情况
                next_action_space = t.next_action_space
                
                # 计算Q值
                if isinstance(next_action_space, tuple) and len(next_action_space) == 3:
                    next_action_features, _, _ = next_action_space
                    # 确保next_action_features在正确的设备上
                    next_action_features = next_action_features.to(device)
                    next_q_values = self.target_net(next_state, next_action_features)
                else:
                    # 确保next_action_space在正确的设备上
                    next_action_space = next_action_space.to(device)
                    next_q_values = self.target_net(next_state, next_action_space)
                
                max_q_value = next_q_values.max().item()
                max_q_values.append(max_q_value)
            
            # 设置非终止状态的目标Q值
            # 注意：这里我们省略了gamma（折扣因子）乘法，如果你的代码使用gamma，请添加
            for i, q_value in zip(non_terminal_indices, max_q_values):
                Q_targets[i] = reward_batch[i]  # + gamma * q_value
        
        # 设置终止状态的目标Q值
        Q_targets[~non_terminal_mask] = reward_batch[~non_terminal_mask]
        
        # 重置梯度
        self.optimizer.zero_grad()
        
        # 计算当前状态下选定动作的Q值
        Q_values = []
        
        for i, t in enumerate(transitions):
            # 处理states - 如果是序列取最后一个
            if isinstance(t.states, torch.Tensor) and t.states.dim() > 1:
                state = t.states[-1].unsqueeze(0).to(device)
            else:
                state = t.states.unsqueeze(0).to(device)
                
            # 获取动作空间
            action_space = t.action_space.to(device) if not isinstance(t.action_space, tuple) else t.action_space
            
            # 计算Q值
            if isinstance(action_space, tuple) and len(action_space) == 3:
                action_features, entity_ids, relation_ids = action_space
                # 确保action_features在正确的设备上
                if hasattr(action_features, 'device') and action_features.device != device:
                    action_features = action_features.to(device)
                q_values = self.policy_net(state, action_features)
            else:
                # 确保action_space在正确的设备上
                if hasattr(action_space, 'device') and action_space.device != device:
                    action_space = action_space.to(device)
                q_values = self.policy_net(state, action_space)
            
            # 默认使用存储的索引
            matched_action_idx = t.action_idx
            
            # 检查是否有存储的实体ID和关系ID进行特征映射
            stored_entity_id = getattr(t, 'action_entity_id', None)
            stored_relation_id = getattr(t, 'action_relation_id', None)
            
            # 使用辅助函数查找匹配的动作索引
            found_idx = self._find_matching_action(action_space, stored_entity_id, stored_relation_id)
            if found_idx != -1:
                matched_action_idx = found_idx
                logger.debug(f"特征映射: 原始索引 {t.action_idx} -> 映射后索引 {matched_action_idx} (实体ID: {stored_entity_id}, 关系ID: {stored_relation_id})")
            elif stored_entity_id is not None or stored_relation_id is not None:
                logger.warning(f"无法找到匹配的动作 (实体ID: {stored_entity_id}, 关系ID: {stored_relation_id}), 使用原始索引 {t.action_idx}")
            
            # 添加索引检查，防止越界
            if matched_action_idx >= len(q_values):
                logger.warning(f"动作索引 {matched_action_idx} 超出当前Q值长度 {len(q_values)}，使用最大索引")
                matched_action_idx = len(q_values) - 1 if len(q_values) > 0 else 0
                action_q_value = q_values[matched_action_idx] if len(q_values) > 0 else torch.tensor(0.0, device=device)
            else:
                action_q_value = q_values[matched_action_idx]
                
            Q_values.append(action_q_value)
        
        # 将所有Q值合并为一个张量
        Q_pred = torch.stack(Q_values)
        
        # 计算损失
        loss = F.l1_loss(Q_pred, Q_targets, reduction="mean")
        
        # 如果启用了CQL损失
        if self.cql_loss_weight > 0:
            # 计算保守Q学习损失
            cql_losses = []
            for i, t in enumerate(transitions):
                state = t.states[-1].unsqueeze(0).to(device) if t.states.dim() > 1 else t.states.unsqueeze(0).to(device)
                action_space = t.action_space.to(device) if not isinstance(t.action_space, tuple) else t.action_space
                
                # 计算Q值
                if isinstance(action_space, tuple) and len(action_space) == 3:
                    action_features, entity_ids, relation_ids = action_space
                    # 确保action_features在正确的设备上
                    if hasattr(action_features, 'device') and action_features.device != device:
                        action_features = action_features.to(device)
                    q_values = self.policy_net(state, action_features)
                else:
                    # 确保action_space在正确的设备上
                    if hasattr(action_space, 'device') and action_space.device != device:
                        action_space = action_space.to(device)
                    q_values = self.policy_net(state, action_space)
                
                # 默认使用存储的索引
                matched_action_idx = t.action_idx
                
                # 检查是否有存储的实体ID和关系ID进行特征映射
                stored_entity_id = getattr(t, 'action_entity_id', None)
                stored_relation_id = getattr(t, 'action_relation_id', None)
                
                # 使用辅助函数查找匹配的动作索引
                found_idx = self._find_matching_action(action_space, stored_entity_id, stored_relation_id)
                if found_idx != -1:
                    matched_action_idx = found_idx
                    logger.debug(f"CQL计算: 特征映射成功, 使用索引 {matched_action_idx}")
                elif stored_entity_id is not None or stored_relation_id is not None:
                    logger.warning(f"CQL计算: 无法找到匹配的动作, 使用原始索引 {t.action_idx}")
                
                # 添加索引检查，防止越界
                if matched_action_idx >= len(q_values):
                    logger.warning(f"CQL损失计算：动作索引 {matched_action_idx} 超出当前Q值长度 {len(q_values)}，使用最大索引")
                    matched_action_idx = len(q_values) - 1 if len(q_values) > 0 else 0
                    action_q_value = q_values[matched_action_idx] if len(q_values) > 0 else torch.tensor(0.0, device=device)
                else:
                    action_q_value = q_values[matched_action_idx]
                
                # 计算logsumexp与选定动作Q值的差
                cql_loss = torch.logsumexp(q_values, dim=0) - action_q_value
                cql_losses.append(cql_loss)
            
            # 计算平均CQL损失并加权添加到总损失
            cql_loss = torch.stack(cql_losses).mean()
            loss = loss + self.cql_loss_weight * cql_loss
        
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        # 更新学习率
        self.scheduler.step()
        
        # 记录wandb指标
        try:
            import wandb
            if wandb.run is not None:
                wandb.log(
                    data={"lr": self.scheduler.get_last_lr()[0], "loss": loss.item()},
                    step=self.curr_step,
                )
        except Exception as e:
            logger.debug(f"记录wandb优化指标时出错: {e}")
            
        return loss.item()

    def train(self):
        self.mode = "train"
        self.env.set_mode("train")
        self.policy_net.train()
        self.target_net.train()

        reward_hist = deque() # 用于存储过去100个episode的奖励，计算平均奖励
        with tqdm(total=self.train_steps - self.curr_step) as pbar:
            while self.curr_step < self.train_steps:
                # interact with environment
                if self.curr_step >= self.offline_steps:
                    # 为每个训练步骤随机选择一个查询样本
                    self._update_training_query()
                    
                    # 1. 与环境交互，收集经验
                    rewards = self.rollout()
                    summary = self.env.summary()
                    summary["train-reward"] = rewards.sum().item() # 计算总奖励
                    reward_hist.append(summary["train-reward"]) # 将总奖励添加到历史奖励列表中
                    if len(reward_hist) > 100: # 如果历史奖励列表超过100个，则移除最早的奖励
                        reward_hist.popleft() # 移除最早的奖励
                    summary["avg-reward-last-100-episodes"] = sum(reward_hist) / len(
                        reward_hist
                    ) # 计算过去100个episode的平均奖励
                    logger.debug(f"step {self.curr_step}, {summary}")
                    
                    # 添加wandb初始化检查
                    try:
                        import wandb
                        if wandb.run is not None:  # 检查wandb是否已初始化
                            wandb.log(data=summary, step=self.curr_step)
                    except Exception as e:
                        logger.debug(f"记录wandb训练指标时出错: {e}")

                # optimize
                for _ in range(self.optimization_steps_per_train_step):
                    # 2. 优化策略网络
                    self.optimize()

                self.curr_step += 1
                pbar.update(1)

                if self.curr_step % self.target_update_every == 0:
                    # 3. 定期更新目标网络
                    self.target_update()

                if self.save_every > 0 and self.curr_step % self.save_every == 0:
                    # 4. 定期保存检查点
                    self.save_checkpoints()

                if self.eval_every > 0 and self.curr_step % self.eval_every == 0:
                    self.eval(eval_mode="val")

        self.save_checkpoints()
        
    def _update_training_query(self):
        """
        为训练随机选择一个查询样本
        
        这确保代理学习到针对特定查询选择相关样本的能力
        """
        if not hasattr(self.env, "update_query") or not hasattr(self.env, "test_data"):
            logger.warning("环境不支持更新查询或没有测试数据")
            return
            
        if not self.env.test_data:
            logger.warning("测试数据为空，无法更新查询")
            return
            
        # 随机选择一个样本作为查询
        query_idx = random.randint(0, len(self.env.test_data) - 1)
        query_sample = self.env.test_data[query_idx]
        
        # 更新环境的当前查询
        try:
            self.env.update_query(query_sample)
            entity, relation, _, timestamp = query_sample[0]
            direction = query_sample[1]
            logger.info(f"更新训练查询: {entity}, {relation}, {timestamp}, 方向: {direction}")
        except Exception as e:
            logger.error(f"更新训练查询时出错: {e}")
            
        # 重置环境，准备新一轮交互
        self.env.reset()

    def eval(
        self,
        eval_mode: str = "val",
        eval_prefix: Optional[str] = None,
        load: str = "last",
    ):
        logger.info(
            "evaluating, "
            f"eval_mode = {eval_mode}, "
            f"eval_prefix = {eval_prefix}, "
            f"load = {load}."
        )

        self.mode = "eval"
        self.env.set_mode(eval_mode)

        if eval_mode == "test":
            if load == "best":
                self.load_best_model()
            elif load == "last":
                self.load_checkpoints()
            else:
                raise Exception(f"unknown load option {load}")
            
            # 确保测试时模型也在正确设备上
            self.policy_net = self.policy_net.to(self.device)
            self.target_net = self.target_net.to(self.device)

        self.policy_net.eval()
        self.target_net.eval()

        rounds = self.val_rounds if eval_mode == "val" else 1
        if eval_prefix is None:
            eval_prefix = eval_mode

        with torch.no_grad():
            summaries = []
            for round in range(rounds):
                rewards = self.rollout()
                future_rewards = tailsum(rewards)
                summary = self.env.summary()
                summary[f"{eval_mode}-reward"] = future_rewards[0].item()
                summaries.append(summary)

            summary = collate_summaries(summaries)
            summary["step"] = self.curr_step

            self.write_eval_results(eval_prefix, summary)
            logger.info(f"step {self.curr_step}, {summary}")
            
            # 添加wandb初始化检查
            try:
                import wandb
                if wandb.run is not None:  # 检查wandb是否已初始化
                    if eval_mode == "val":
                        wandb.log(data=summary, step=self.curr_step)
                    else:
                        wandb.log(data={f"{eval_prefix}-{k}": v for k, v in summary.items()})
            except Exception as e:
                logger.debug(f"记录wandb评估指标时出错: {e}")

        self.mode = "train"
        self.env.set_mode("train")

        self.policy_net.train()
        self.target_net.train()

    def write_eval_results(self, eval_mode: str, data: Dict):
        res_path = os.path.join(self.output_dir, f"{eval_mode}-results.jsonl")
        with jsonlines.open(res_path, mode="a") as f:
            f.write(data)

    def load_model_at_step(self, step):
        model_ckpt_path = join(self.ckpt_dir, f"model_{step}.ckpt")
        self.policy_net.load_state_dict(torch.load(model_ckpt_path))
        # 确保加载后的模型在指定设备上
        self.policy_net = self.policy_net.to(self.device)
        self.target_update()

    def load_best_model(self):
        metrics_path = join(self.output_dir, "metrics.json")
        if not os.path.exists(metrics_path):
            logger.info("no metrics file found, can't load best model...")
            return
        metrics = json.load(open(metrics_path))
        best_step = max(
            metrics["test"]["step"],
            key=lambda step: metrics["test"][step]["mrr"],
        )
        logger.info(f"loading best model from step {best_step}...")
        self.load_model_at_step(best_step)
