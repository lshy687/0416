import glob
import logging
import os
import random
from os.path import join
from typing import Optional, Tuple, Any

import torch
from tqdm.auto import tqdm

from .replay import NamedTransition
from ..base_environment import BaseEnvironment
from ..misc_utils import parse_step_from_checkpoint

logger = logging.getLogger(__name__)


class RandomAgent:
    def __init__(
        self,
        env: BaseEnvironment,
        output_dir: Optional[str] = None,
        train_steps: int = 1000,
        save_every: int = 200,
    ):
        self.env = env
        self.train_steps = train_steps
        self.save_every = save_every
        self.transitions = []
        self.ckpt_dir = join(output_dir, "ckpts") if output_dir else None
        self.curr_step = 0
        self.load_checkpoints()

    def load_checkpoints(self):
        if self.ckpt_dir is None:
            return

        replay_ckpts = {
            parse_step_from_checkpoint(f): f
            for f in glob.glob(join(self.ckpt_dir, "transitions_*.ckpt"))
        }

        ckpts_found = set(replay_ckpts.keys())

        if not ckpts_found:
            logger.info("no existing checkpoints, train from scratch...")
            if 0 in replay_ckpts:
                logger.info("loading initial replay memory...")
                self.transitions = torch.load(replay_ckpts[0])
            return

        step = max(ckpts_found)
        logger.info(f"setting step={step}")
        self.curr_step = step
        logger.info(f"loading transitions from step={step}")
        self.transitions = torch.load(replay_ckpts[step])

    def save_checkpoints(self):
        if self.ckpt_dir is None:
            return
        step = self.curr_step
        os.makedirs(self.ckpt_dir, exist_ok=True)

        logger.info(f"saving transitions from for step={step}")
        t_ckpt_path = join(self.ckpt_dir, f"transitions_{step}.ckpt")
        torch.save(self.transitions, t_ckpt_path)

    def rollout(self, query_info=None):
        env = self.env
        state = env.reset()
        terminal = False
        rewards = []
        past_states = [state]
        action_indices = []
        action_spaces = []

        while not terminal:
            # 获取当前可用的动作空间信息
            action_space_tuple = env.action_space()
            if action_space_tuple is None:
                logger.warning("环境返回了空的 action_space，无法选择动作，终止 rollout。")
                break

            # 假设 action_space_tuple 是 (features, entity_ids, relation_ids)
            # 我们需要特征的数量来确定有多少可用动作
            action_features = None
            if isinstance(action_space_tuple, tuple) and len(action_space_tuple) > 0:
                action_features = action_space_tuple[0]

            num_available_actions = 0
            if torch.is_tensor(action_features) and action_features.dim() >= 1:
                num_available_actions = action_features.shape[0]
            elif isinstance(action_features, list):  # 兼容列表形式
                num_available_actions = len(action_features)

            if num_available_actions == 0:
                logger.warning("动作空间为空，无法选择动作，终止 rollout。")
                break

            # 从可用动作中随机选择一个相对索引
            action_idx = random.randrange(num_available_actions)

            # 尝试获取这个相对索引对应的原始索引（如果需要的话）
            try:
                # 检查环境是否有获取可用样本及其原始索引的方法
                if hasattr(env, 'get_available_samples'):
                    _, current_available_indices, _ = env.get_available_samples()
                    if action_idx >= len(current_available_indices):
                        logger.error(f"随机选择的相对索引 {action_idx} 超出可用原始索引列表范围 (长度 {len(current_available_indices)})！")
                        break
                    # 如果环境需要原始索引，用这个
                    original_action_idx = current_available_indices[action_idx]
                    next_state, reward, terminal = env.step(original_action_idx)
                else:
                    # 如果环境能直接处理相对索引
                    next_state, reward, terminal = env.step(action_idx)
                    original_action_idx = action_idx  # 记录用于存储
            except Exception as e:
                logger.error(f"在执行步骤时出错: {e}")
                break

            rewards.append(reward)
            if not terminal:
                past_states.append(next_state)
            action_indices.append(original_action_idx)
            action_spaces.append(action_space_tuple)

        for i in range(len(rewards) - 1):
            states = past_states[: i + 1]
            action_idx = action_indices[i]
            action_space = action_spaces[i]
            next_states = past_states[: i + 2]
            next_action_space = action_spaces[i + 1]
            reward = torch.tensor(rewards[i])

            t = NamedTransition(
                states,
                action_idx,
                action_space,
                next_states,
                next_action_space,
                reward,
                query_info=query_info
            )
            self.transitions.append(t)

        # push terminal transition
        if action_indices:  # 确保至少有一次成功的动作
            states = past_states
            action_idx = action_indices[-1]
            action_space = action_spaces[-1]
            reward = torch.tensor(rewards[-1])
            t = NamedTransition(
                states, 
                action_idx, 
                action_space, 
                None, 
                None, 
                reward, 
                query_info=query_info
            )
            self.transitions.append(t)

    def train(self, query_info=None):
        self.env.set_mode("train")
        assert self.env.named

        with tqdm(total=self.train_steps - self.curr_step) as pbar:
            while self.curr_step < self.train_steps:
                self.rollout(query_info=query_info)

                self.curr_step += 1
                pbar.update(1)

                if self.save_every > 0 and self.curr_step % self.save_every == 0:
                    self.save_checkpoints()
