import logging
import math
import random
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from .misc_utils import normalized_entropy, tensor_stats

logger = logging.getLogger(__name__)


class BaseEnvironment(ABC):
    """基础环境类，作为所有强化学习环境的接口"""
    
    @abstractmethod
    def reset(self):
        """重置环境，返回初始状态"""
        pass
        
    @property
    @abstractmethod
    def state(self) -> torch.Tensor:
        """返回当前状态"""
        pass
        
    def set_mode(self, mode: str):
        """设置环境模式（训练、验证或测试）"""
        self.mode = mode
        
    @property
    @abstractmethod
    def state_dim(self) -> int:
        """返回状态空间维度"""
        pass
        
    @property
    @abstractmethod
    def action_dim(self) -> int:
        """返回动作空间维度"""
        pass
        
    @abstractmethod
    def action_count(self) -> int:
        """返回当前有效动作数量"""
        pass
        
    @abstractmethod
    def action_space(self):
        """返回当前有效动作空间"""
        pass
        
    @abstractmethod
    def step(self, idx: int):
        """执行一步动作，返回下一状态、奖励、是否终止和附加信息"""
        pass
        
    @abstractmethod
    def summary(self) -> Dict:
        """返回环境摘要"""
        pass


# 其他原始environment.py中的实现类...

class RandomProjection:
    def __init__(self, do_reduce: bool, in_features: int, out_features: int):
        self.do_reduce = do_reduce
        if do_reduce:
            logger.info("initializing random projection matrix")
            rng = torch.Generator().manual_seed(42)
            self.proj_mat = torch.normal(
                mean=0.0,
                std=1.0 / math.sqrt(out_features),
                size=(in_features, out_features),
                generator=rng,
            )

    def __call__(self, X: torch.Tensor):
        if self.do_reduce:
            return X @ self.proj_mat
        return X


class FewShotEnvironment(BaseEnvironment):
    # ... 原始实现代码
    pass


class GPT3Environment(BaseEnvironment):
    # ... 原始实现代码
    pass


class MultiDatasetEnvironment(BaseEnvironment):
    # ... 原始实现代码
    pass


class ToyEnvironment(BaseEnvironment):
    # ... 原始实现代码
    pass


class ToyRecurrentEnvironment(BaseEnvironment):
    # ... 原始实现代码
    pass 