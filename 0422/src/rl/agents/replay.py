import logging
import random
from collections import deque
from typing import Union

import torch

from src.rl.base_environment import BaseEnvironment

logger = logging.getLogger(__name__)


class Transition:
    def __init__(
        self, states, action_idx, action_space, next_states, next_action_space, reward,
        action_entity_id=None, action_relation_id=None
    ):
        self.states = states
        self.action_idx = action_idx
        self.action_space = action_space
        self.next_states = next_states
        self.next_action_space = next_action_space
        self.reward = reward
        self.action_entity_id = action_entity_id
        self.action_relation_id = action_relation_id


class NamedTransition(Transition):
    def __init__(self, *args):
        super(NamedTransition, self).__init__(*args)


class ReplayMemory(object):
    def __init__(self, capacity: int):
        self.memory = deque([], maxlen=capacity)

    def push(self, t: Transition):
        # 处理 action_space 可能是元组的情况
        if isinstance(t.action_space, tuple) and len(t.action_space) == 3:
            action_features = t.action_space[0]  # 直接获取第一个元素（特征张量）
            if action_features.shape[0] == 1:
                logger.info("skip push: action_features.shape[0] == 1")
                return
        # 原始逻辑 - 直接检查张量
        elif hasattr(t.action_space, 'shape') and t.action_space.shape[0] == 1:
            logger.info("skip push: action_space.shape[0] == 1")
            return
            
        # 处理 next_action_space 可能是元组的情况
        if t.next_action_space is not None:
            if isinstance(t.next_action_space, tuple) and len(t.next_action_space) == 3:
                next_action_features = t.next_action_space[0]  # 直接获取第一个元素（特征张量）
                if next_action_features.shape[0] == 1:
                    logger.info("skip push: next_action_features.shape[0] == 1")
                    return
            # 原始逻辑 - 直接检查张量
            elif hasattr(t.next_action_space, 'shape') and t.next_action_space.shape[0] == 1:
                logger.info("skip push: next_action_space.shape[0] == 1")
                return
                
        self.memory.append(t)

    def sample(self, k=1):
        return random.sample(self.memory, k=k)

    def __len__(self):
        return len(self.memory)

    def load(
        self,
        path: Union[str, list[str]],
        env: BaseEnvironment,
    ):
        """
        从文件加载转换
        """
        if not path:
            return

        assert isinstance(env, BaseEnvironment)
        logger.info(f"loading transitions from path {path} with env {env}")
        transitions = torch.load(path)
        
        # 处理转换记录
        if isinstance(transitions, list):
            for t in transitions:
                try:
                    self.push(t)
                except Exception as e:
                    logger.warning(f"Error pushing transition: {e}")
            
            logger.info(f"Loaded {len(transitions)} transitions")
            return
            
        # 处理具有任务信息的转换词典
        for task, task_trans in transitions.items():
            for t in task_trans:
                try:
                    self.push(t)
                except Exception as e:
                    logger.warning(f"Error pushing transition from task {task}: {e}")
        
        logger.info(f"Loaded transitions from {len(transitions)} tasks")
