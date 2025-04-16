import logging
import math
import random
import hashlib
from typing import Dict, List, Optional, Union, Tuple

import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from model_utils import predict
from utils import prepare_input, HitsMetric, get_entities, get_relations, get_entity_relation_mappings, update_history  # 只导入真正需要的函数
from .base_environment import BaseEnvironment
from .plm_encoder import PLMEncoder  # 导入PLM编码器

logger = logging.getLogger(__name__)


class TKGEnvironment(BaseEnvironment):
    """
    用于时序知识图谱预测的强化学习环境
    
    状态：当前时间戳下的可用样本特征，融合了查询信息
    动作：选择某个样本进行专家标注
    奖励：基于模型性能变化的奖励（使用MRR指标）
    """
    
    def __init__(
        self,
        interaction_data,
        state_model,
        state_tokenizer,
        reward_model,
        reward_tokenizer,
        train_history,
        args,
        state_repr=None,
        max_steps=5,
        reward_scale=1.0,
        predef_entity_ids=None,
        predef_relation_ids=None,
        state_model_type="encoder",
    ):
        """
        初始化TKG环境
        
        Args:
            interaction_data: 用于环境交互的数据集 (通常是验证集)。
                              格式: [([h, r, [t1, t2...], ts], direction), ...]
            state_model: 用于提取状态特征的模型 (如 BERT)。
            state_tokenizer: state_model 的分词器。
            reward_model: 用于预测并计算奖励的模型 (主 LLM, Qwen)。
            reward_tokenizer: reward_model 的分词器。
            train_history: 从训练集构建的历史数据结构（搜索空间字典）。
            args: 运行参数
            state_repr: 状态表示方式，可包含以下选项：
                - "query_features": 查询语义特征，提供待预测样本的语义表示
                - "context_features": 已选示例集合的BERT编码，表示当前上下文的整体语义
                - "interaction_features": 上下文与查询之间的交互特征
                - "diversity_features": 当前已选示例的多样性度量，表示语义空间中的分散程度
                - "curr_step": 当前步数比例，表示已选择样本占最大可选数量的比例
            max_steps: 每个时间戳最大选择样本数
            reward_scale: 奖励缩放因子
            predef_entity_ids: 预定义的实体ID映射
            predef_relation_ids: 预定义的关系ID映射
            state_model_type: 状态模型的类型 ('encoder' or 'decoder')
        """
        super().__init__()
        self.interaction_data = interaction_data
        self.state_model = state_model
        self.state_tokenizer = state_tokenizer
        self.reward_model = reward_model
        self.reward_tokenizer = reward_tokenizer
        self.train_history = train_history  # 这是搜索空间字典，不是之前的历史记录结构
        self.args = args
        self.state_repr = state_repr if state_repr else ["query_features"]
        self.max_steps = max_steps
        self.reward_scale = reward_scale
        
        # 设置特征配置
        self.feature_config = {
            "semantic_dim": 32,
            "interaction_method": "attention",
            "diversity_method": "variance"
        }
        
        # 环境模式：训练、验证或测试
        self.mode = "train"
        
        # 初始化PLM编码器
        self.encoder = PLMEncoder(state_model, state_tokenizer, model_type=state_model_type)
        
        # 提取所有唯一的时间戳
        self.timestamps = sorted(list(set([x[0][3] for x in interaction_data])))
        
        # 当前环境状态
        self.current_timestamp_idx = 0
        self.current_timestamp = None
        self.current_samples = []
        self.current_indices = []
        self.selected_samples = []
        self.selected_indices = []
        self.mrr_history = []  # 存储MRR历史
        self.steps_taken = 0
        self.current_query = None  # 当前查询
        
        # 创建实体和关系的ID映射
        self.entity_ids = predef_entity_ids if predef_entity_ids is not None else {}
        self.relation_ids = predef_relation_ids if predef_relation_ids is not None else {}
        self.create_id_mappings()
        
        # 初始化随机投影矩阵 (用于特征降维)
        self.init_random_projection()
        
        # 初始化特征维度
        self.feature_dim = self._calculate_feature_dim()
        
        # 缓存已计算的特征
        self.feature_cache = {}
        self.semantic_cache = {}  # 语义特征缓存
        self.context_cache = {}   # 上下文特征缓存
        self.diversity_cache = {} # 多样性特征缓存
        
        self.historical_samples = []  # Store historical context
        
        self.reset()
    
    def create_id_mappings(self):
        """创建实体和关系的ID映射"""
        # 对所有样本中的实体和关系创建唯一ID
        for (sample, _) in self.interaction_data:
            entity, relation, targets, _ = sample
            
            if entity not in self.entity_ids:
                self.entity_ids[entity] = len(self.entity_ids)
                
            if relation not in self.relation_ids:
                self.relation_ids[relation] = len(self.relation_ids)
                
            for target in targets:
                if target not in self.entity_ids:
                    self.entity_ids[target] = len(self.entity_ids)
    
    def _calculate_feature_dim(self):
        """计算特征维度"""
        # 基础特征维度
        base_dim = 64
        
        # 根据状态表示类型计算总维度
        total_dim = 0
            
        # 查询特征
        if "query_features" in self.state_repr:
            # 查询的语义特征维度，使用配置的降维维度
            query_dim = self.feature_config.get("semantic_dim", 32)  # 降维后的维度
            total_dim += query_dim
            
        # 当前步骤信息
        if "curr_step" in self.state_repr:
            total_dim += 1  # 仅包含当前步数比例
            
        # 上下文特征
        if "context_features" in self.state_repr:
            # 上下文特征维度与查询特征保持一致
            context_dim = self.feature_config.get("semantic_dim", 32)  # 降维后的维度
            total_dim += context_dim
            
        # 交互特征
        if "interaction_features" in self.state_repr:
            if self.feature_config["interaction_method"] == "attention":
                total_dim += 16  # 注意力评分和统计特征
            else:
                total_dim += 4   # 余弦相似度统计特征
                
        # 多样性特征
        if "diversity_features" in self.state_repr:
            total_dim += 8  # 多样性统计特征
            
        return max(total_dim, base_dim)  # 确保至少有基础维度
    
    @property
    def state_dim(self) -> int:
        """返回状态空间维度"""
        return self.feature_dim
    
    @property
    def action_dim(self) -> int:
        """返回动作空间维度"""
        # 动作特征维度包括语义特征维度+相似度特征+独特性特征
        self.action_feature_dim = self.feature_config.get("semantic_dim", 32) + 3  # 3 = similarity_to_context + similarity_to_query + uniqueness
        return self.action_feature_dim
    
    @property
    def state(self) -> torch.Tensor:
        """返回当前状态"""
        # 当前状态表示为特征列表
        features = []
        
        # 查询特征
        if "query_features" in self.state_repr and self.current_query:
            query_sample, query_direction = self.current_query
            query_embedding = self._extract_semantic_features(query_sample, query_direction)
            if query_embedding is not None:
                features.append(query_embedding)
            else:
                # 如果无法获取查询嵌入，使用零向量
                features.append(torch.zeros(self.feature_config.get("semantic_dim", 32)))
        
        # 当前步骤信息
        if "curr_step" in self.state_repr:
            step_ratio = torch.tensor([self.steps_taken / self.max_steps])
            features.append(step_ratio)
        
        # 上下文特征
        if "context_features" in self.state_repr:
            context_embedding = self._extract_context_features()
            if context_embedding is not None:
                features.append(context_embedding)
            else:
                features.append(torch.zeros(self.feature_config.get("semantic_dim", 32)))
        
        # 交互特征
        if "interaction_features" in self.state_repr:
            interaction_features = self._extract_interaction_features()
            if interaction_features is not None:
                features.append(interaction_features)
            else:
                # 默认交互特征维度
                interaction_dim = 16 if self.feature_config["interaction_method"] == "attention" else 4
                features.append(torch.zeros(interaction_dim))
        
        # 多样性特征
        if "diversity_features" in self.state_repr:
            diversity_features = self._calculate_diversity_feature()
            if diversity_features is not None:
                features.append(diversity_features)
            else:
                features.append(torch.zeros(8))  # 默认多样性特征维度
        
        # 如果没有任何特征，返回全零向量
        if not features:
            return torch.zeros(self.feature_dim)
        
        # 合并所有特征并确保维度匹配
        state_vector = torch.cat(features)
        
        # 如果实际特征维度与预期不同，填充或截断
        if state_vector.shape[0] != self.feature_dim:
            if state_vector.shape[0] < self.feature_dim:
                padding = torch.zeros(self.feature_dim - state_vector.shape[0])
                state_vector = torch.cat([state_vector, padding])
            else:
                state_vector = state_vector[:self.feature_dim]
        
        return state_vector

    def get_available_samples(self):
        """获取当前时间戳下可用的样本"""
        if self.current_timestamp_idx >= len(self.timestamps):
            return [], [], None
        
        timestamp = self.timestamps[self.current_timestamp_idx]
        samples = []
        indices = []
        
        for i, (sample, direction) in enumerate(self.interaction_data):
            if sample[3] == timestamp and i not in self.selected_indices:
                samples.append((sample, direction))
                indices.append(i)
        
        return samples, indices, timestamp
    
    def evaluate_mrr(self, active_selected_samples=None):
        """
        评估在给定历史上下文和主动选择的样本下的MRR。

        Args:
            active_selected_samples: 由RL Agent主动选择的样本列表。

        Returns:
            (MRR值, 总样本数, 倒数排名总和)
        """
        if active_selected_samples is None:
            active_selected_samples = []
            
        metric = HitsMetric()

        if self.current_query is None:
            logger.debug("evaluate_mrr: 没有当前查询，返回0.0")
            return 0.0, 0, 0

        query_sample, query_direction = self.current_query
        entity, relation, targets, timestamp = query_sample

        logger.debug(f"evaluate_mrr: 评估查询 {entity}, {relation} at {timestamp} (dir: {query_direction})")
        logger.debug(f"evaluate_mrr: 历史样本数 {len(self.historical_samples)}, 主动选择样本数 {len(active_selected_samples)}")

        try:
            # 准备包含历史上下文和主动选择样本的提示
            prompt = self._prepare_prompt_combined(query_sample, query_direction, self.historical_samples, active_selected_samples)

            # 获取搜索空间
            if query_direction == "head":
                search_space = {}  # 头实体搜索空间
            else:
                search_space = {}  # 尾实体搜索空间

            # 使用模型进行预测
            predictions = predict(self.reward_tokenizer, self.reward_model, prompt, self.args)

            if not predictions:
                logger.debug("evaluate_mrr: 模型未返回预测结果，返回0.0")
                return 0.0, 0, 0

            logger.debug(f"evaluate_mrr: 预测结果数量 {len(predictions)}, 前5: {predictions[:5]}")
            logger.debug(f"evaluate_mrr: 正确目标 {targets}")

            # 计算排名
            rank = float('inf')
            correct_targets_set = set(targets) # Use set for faster lookup
            for i, (pred_entity, _) in enumerate(predictions, 1):
                if pred_entity in correct_targets_set:
                    rank = i
                    logger.debug(f"evaluate_mrr: 找到正确答案 '{pred_entity}' @ rank {rank}")
                    break

            # 更新指标
            metric.total += 1
            if rank != float('inf'):
                metric.update(rank)
                logger.debug(f"evaluate_mrr: 更新指标 rank={rank}, reciprocal={1.0/rank:.4f}")
            else:
                max_rank = len(predictions) + 1 # 如果没找到，使用最大排名+1
                metric.update(max_rank)
                logger.debug(f"evaluate_mrr: 未找到正确答案，更新指标 rank={max_rank}, reciprocal={1.0/max_rank:.4f}")

        except Exception as e:
            logger.error(f"evaluate_mrr: 预测或处理样本时出错: {e}", exc_info=True)
            # 出错时返回上一步的MRR或0
            return self.mrr_history[-1] if self.mrr_history else 0.0, 0, 0

        # 计算最终MRR
        if metric.total == 0:
            logger.debug("evaluate_mrr: 总样本数为0，返回0.0")
            return 0.0, 0, 0

        mrr = metric.mrr_sum / metric.total
        logger.debug(f"evaluate_mrr: 计算得到 MRR={mrr:.4f}, Total={metric.total}, SumRR={metric.mrr_sum:.4f}")
        return mrr, metric.total, metric.mrr_sum
    
    def _prepare_prompt_combined(self, query_sample, query_direction, historical_samples, active_selected_samples):
        """
        准备包含历史上下文和主动选择样本的提示。

        Args:
            query_sample: 当前查询样本元组
            query_direction: 预测方向
            historical_samples: 历史上下文样本列表
            active_selected_samples: RL Agent选择的当前时间步样本列表

        Returns:
            最终的提示字符串
        """
        # 1. 获取基础查询格式（不含任何上下文）
        try:
            # 修改调用方式，使用新的prepare_input签名
            base_query_prompt, _, _ = prepare_input(
                query_sample,  # 传递样本元组
                self.train_history,  # 传递搜索空间
                self.args,
                return_prompt=True
            )
        except Exception as e:
             logger.error(f"调用 prepare_input 获取基础提示时出错: {e}", exc_info=True)
             return "" # 返回空字符串表示失败

        # 2. 格式化历史样本
        historical_context_str = self._format_samples_for_prompt(historical_samples, self.args, context_type="historical")

        # 3. 整合基础查询和历史上下文 (历史在前，查询在后)
        prompt_with_history = historical_context_str + base_query_prompt

        # 4. 格式化主动选择的样本 (使用 active_learning 中的函数)
        try:
            # 延迟导入以处理可能的循环依赖或加载顺序问题
            from active_learning import integrate_active_samples
            final_prompt = integrate_active_samples(prompt_with_history, active_selected_samples, self.args)
        except ImportError:
             logger.error("无法导入 active_learning.integrate_active_samples")
             final_prompt = prompt_with_history # 回退
        except Exception as e:
            logger.error(f"调用 integrate_active_samples 添加主动样本时出错: {e}", exc_info=True)
            final_prompt = prompt_with_history # 回退

        # logger.debug(f"_prepare_prompt_combined: 生成的最终提示:\n{final_prompt}") # 可能太长，暂时注释掉
        return final_prompt

    def _format_samples_for_prompt(self, samples, args, context_type="historical"):
        """将样本列表格式化为字符串，以便添加到提示中。主要用于历史样本。"""
        prompt_part = ""
        if not samples:
            return prompt_part

        # 仅为历史样本添加标题
        title = ""
        if context_type == "historical":
            # 加一个换行符与前面内容隔开，再加标题，再加换行符与样本列表隔开
            title = "\n历史相关事件:\n"
        # 主动学习样本的标题和格式化由 integrate_active_samples 负责

        prompt_part += title

        # 遍历样本并格式化
        formatted_samples = []
        for item in samples:
            try:
                # 尝试解包，兼容 [(sample_tuple, direction), ...] 和 [sample_tuple, ...]
                if isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[0], tuple) and len(item[0]) == 4:
                    sample_data, _ = item # 方向信息在此处不用
                elif isinstance(item, tuple) and len(item) == 4:
                    sample_data = item
                else:
                    logger.warning(f"_format_samples_for_prompt: 未知的样本格式: {type(item)}，跳过")
                    continue

                entity, relation, targets, time = sample_data
                target = targets[0] if targets else "?" # 使用第一个目标

                # 构建单行样本字符串
                sample_str = ""
                if not args.no_time:
                    sample_str += f"{time}:"
                if args.label:
                    target_idx_str = str(targets.index(target)) if target != '?' and target in targets else '?'
                    sample_str += f"[{entity},{relation},{target_idx_str}. {target}]"
                else:
                    sample_str += f"[{entity},{relation},{target}]"
                formatted_samples.append(sample_str)
            except Exception as e:
                 logger.warning(f"_format_samples_for_prompt: 格式化样本 {item} 时出错: {e}", exc_info=False) # 减少日志噪音
                 continue # 跳过出错的样本

        # 用换行符连接所有格式化后的样本字符串
        if formatted_samples:
            prompt_part += "\n".join(formatted_samples)
            prompt_part += "\n" # 在样本列表末尾添加一个换行符，与后续内容分隔

        return prompt_part

    def reset(self):
        """重置环境"""
        # 保存当前查询信息以便在重置后仍可使用
        current_query = self.current_query
        
        self.selected_samples = []
        self.selected_indices = []
        self.mrr_history = []
        self.steps_taken = 0
        
        # 获取初始样本（如果没有设置current_samples，使用时间戳）
        if not self.current_samples:
            self.current_samples, self.current_indices, self.current_timestamp = self.get_available_samples()
        
        # 如果没有样本可用，环境已结束
        if not self.current_samples:
            return self.state, True
            
        # 确保重置后查询信息依然可用
        if current_query is not None:
            self.current_query = current_query
            logger.debug(f"重置环境后保留查询: {current_query[0][0]}, {current_query[0][1]}")
            
        # 计算初始MRR
        if self.current_query is not None:
            initial_mrr, _, _ = self.evaluate_mrr([])
            logger.debug(f"重置后初始MRR: {initial_mrr:.4f}")
            self.mrr_history.append(initial_mrr)
        else:
            self.mrr_history.append(0.0)
            logger.debug("没有查询，初始MRR设为0")
        
        # 清除所有相关缓存以确保状态隔离和内存管理
        self._clear_similarity_cache()

        return self.state, False
    
    def action_count(self):
        """可用动作数量"""
        return len(self.current_samples)
    
    def action_space(self):
        """
        动作空间，返回一个元组，包含：
        1. 动作特征张量 - 形状: (num_available_samples, action_feature_dim)
        2. 实体ID列表 - 每个动作对应的实体ID
        3. 关系ID列表 - 每个动作对应的关系ID
        
        为保持兼容性，如果调用方只期望特征张量，可以只使用返回值的第一个元素。
        """
        if not self.current_samples:
            # 如果没有可用样本，返回一个形状正确的空张量和空列表
            return torch.empty((0, self.action_feature_dim)), [], []

        action_features_list = []
        entity_ids_list = []  # 存储实体ID
        relation_ids_list = []  # 存储关系ID
        
        context_embedding = self._extract_context_features() if self.selected_samples else None
        query_embedding = None
        if self.current_query:
            query_sample, query_direction = self.current_query
            query_embedding = self._extract_semantic_features(query_sample, query_direction)

        for sample, direction in self.current_samples:
            # 提取样本的实体ID和关系ID
            entity, relation, _, _ = sample
            entity_ids_list.append(entity)  # 存储实体ID
            relation_ids_list.append(relation)  # 存储关系ID
            
            # 1. 候选示例的语义表示
            candidate_embedding = self._extract_semantic_features(sample, direction)
            if candidate_embedding is None:
                 # 如果无法获取嵌入，跳过这个动作或使用零向量？暂时用零向量
                 candidate_embedding = torch.zeros(self.feature_config.get("semantic_dim", 32))

            # 2. 与当前上下文的相似度
            similarity_to_context = self._calculate_similarity(candidate_embedding, context_embedding)

            # 3. 与查询的相似度
            similarity_to_query = self._calculate_similarity(candidate_embedding, query_embedding)

            # 4. 候选示例的独特性
            uniqueness = self._calculate_uniqueness(candidate_embedding)

            # 组合特征
            # 确保所有特征都是标量或1D张量，然后 unsqueeze(-1) 变成 (1,)
            action_feature = torch.cat([
                candidate_embedding,
                similarity_to_context.unsqueeze(0),
                similarity_to_query.unsqueeze(0),
                uniqueness.unsqueeze(0)
            ])
            action_features_list.append(action_feature)

        if not action_features_list:
             return torch.empty((0, self.action_feature_dim)), [], []

        # 将所有动作特征堆叠成一个张量
        action_features = torch.stack(action_features_list)
        
        # 返回元组：(特征张量, 实体ID列表, 关系ID列表)
        return action_features, entity_ids_list, relation_ids_list
    
    def step(self, action_idx):
        """
        执行一步动作（选择一个样本）
        
        Args:
            action_idx: 动作索引，对应要选择的样本索引
        
        Returns:
            (下一状态, 奖励, 是否终止, 信息)
        """
        if action_idx < 0 or action_idx >= len(self.current_samples):
            raise ValueError(f"无效的动作索引: {action_idx}, 应该在 [0, {len(self.current_samples)-1}] 范围内")
        
        # 选择样本
        selected_sample = self.current_samples[action_idx]
        selected_idx = self.current_indices[action_idx]
        
        sample, direction = selected_sample
        entity, relation, _, timestamp = sample
        logger.debug(f"选择样本: {entity}, {relation}, {timestamp}, 方向: {direction}, 索引: {action_idx}")
        
        # 添加到已选择列表
        self.selected_samples.append(selected_sample)
        self.selected_indices.append(selected_idx)
        
        # 清除依赖于selected_samples的缓存
        self.context_cache = {}
        self.diversity_cache = {}
        # feature_cache 可能包含交互特征，也依赖于上下文和查询，在此一并清除以确保安全
        self.feature_cache = {}

        # 从当前可用样本中移除
        self.current_samples.pop(action_idx)
        self.current_indices.pop(action_idx)
        
        # 增加步数
        self.steps_taken += 1
        logger.debug(f"步数: {self.steps_taken}/{self.max_steps}")
        
        # 计算新的MRR
        new_mrr, total, reciprocal_sum = self.evaluate_mrr()
        
        # 计算奖励（MRR提升）
        prev_mrr = self.mrr_history[-1] if self.mrr_history else 0.0
        base_reward = (new_mrr - prev_mrr) * self.reward_scale
        reward = base_reward
        logger.debug(f"基础奖励(MRR提升): {base_reward:.4f}, 从{prev_mrr:.4f}到{new_mrr:.4f}")
        
        # 如果奖励为负，我们可以适当减少惩罚强度
        if reward < 0:
            old_reward = reward
            reward = reward * 0.5  # 减轻负奖励的影响
            logger.debug(f"负奖励减轻: {old_reward:.4f} -> {reward:.4f}")
            
        # 如果是最后一个可选样本，给予额外奖励（鼓励选择完所有样本）
        if not self.current_samples and self.steps_taken < self.max_steps:
            complete_bonus = 0.1 * self.reward_scale
            reward += complete_bonus
            logger.debug(f"完成所有样本奖励: +{complete_bonus:.4f}")
            
        self.mrr_history.append(new_mrr)
        logger.debug(f"总奖励: {reward:.4f}")
        
        # 检查是否达到最大步数或没有可用样本
        done = self.steps_taken >= self.max_steps or not self.current_samples
        if done:
            logger.debug(f"环境结束: 达到最大步数={self.steps_taken >= self.max_steps}, 没有样本={not self.current_samples}")
        
        # 如果当前时间戳的样本已处理完，移到下一个时间戳
        if done and self.current_timestamp_idx < len(self.timestamps) - 1:
            self.current_timestamp_idx += 1
            next_samples, next_indices, next_timestamp = self.get_available_samples()
            
            if next_samples:
                self.current_samples = next_samples
                self.current_indices = next_indices
                self.current_timestamp = next_timestamp
                self.selected_samples = []
                self.selected_indices = []
                self.steps_taken = 0
                
                # 计算新时间戳的初始MRR
                if self.current_query is not None:
                    initial_mrr, _, _ = self.evaluate_mrr([])
                    self.mrr_history = [initial_mrr]
                else:
                    self.mrr_history = [0.0]
                
                done = False
        
        # 额外信息
        info = {
            "mrr": new_mrr,
            "total": total,
            "reciprocal_sum": reciprocal_sum,
            "mrr_history": self.mrr_history,
            "selected_count": len(self.selected_samples),
            "remaining_count": len(self.current_samples),
            "timestamp": self.current_timestamp,
            "reward": reward,
        }
        
        return self.state, reward, done, info
        # 返回当前状态，奖励，是否终止，信息；是强化学习的标准接口
    
    def summary(self):
        """返回环境摘要信息"""
        return {
            f"{self.mode}/最终MRR": self.mrr_history[-1] if self.mrr_history else 0.0,
            f"{self.mode}/MRR变化": self.mrr_history,
            f"{self.mode}/选择的样本数": len(self.selected_samples),
            f"{self.mode}/当前时间戳": self.current_timestamp,
            f"{self.mode}/时间戳索引": self.current_timestamp_idx,
            f"{self.mode}/总时间戳数": len(self.timestamps),
        }

    def update_query(self, query):
        """
        更新当前查询
        
        Args:
            query: 新的查询 (sample, direction)
        """
        self.current_query = query
        # 清除与查询相关的缓存 (保留，因为查询嵌入需要更新)
        self.semantic_cache = {} # 查询或上下文变化时，语义缓存可能失效
        self.context_cache = {} # 上下文变化时需要清除
        self.diversity_cache = {} # 上下文变化时需要清除
        self.feature_cache = {} # 交互特征依赖查询和上下文

    def update_candidates(self, samples, indices):
        """
        更新候选样本和索引
        
        Args:
            samples: 候选样本列表
            indices: 候选样本索引列表
        """
        self.current_samples = samples
        self.current_indices = indices
        # 清除与样本相关的缓存 (保留，因为动作空间需要重新计算)
        # 注意：这里不清缓存可能导致action_space使用旧数据，最好清除
        self.semantic_cache = {} # 候选样本变化，其语义特征需重算
        # context, diversity, interaction 不直接依赖候选，但 agent 可能需要最新状态
        # self.context_cache = {}
        # self.diversity_cache = {}
        # self.feature_cache = {}

    def _clear_similarity_cache(self):
        """清除（现在是所有相关）缓存"""
        # 当更新查询或样本时清除相关缓存
        self.semantic_cache = {}
        self.context_cache = {}
        self.diversity_cache = {}
        self.feature_cache = {} # 包含交互特征等

    def _extract_sample_text(self, sample, direction, simplified=False):
        """
        从样本中提取文本表示
        
        Args:
            sample: (entity, relation, targets, timestamp) 元组
            direction: "head" 或 "tail" 表示预测方向
            simplified: 是否使用简化表示（默认False）
            
        Returns:
            样本的文本表示
        """
        entity, relation, _, timestamp = sample
        
        if simplified:
            # 简化表示，更接近于三元组形式但保留方向信息
            if direction == "head":
                # 预测头实体
                text = f"[?] {relation} ? {timestamp}"
            else:
                # 预测尾实体
                text = f"{entity} {relation} [?] {timestamp}"
                
            if self.args.no_time:
                if direction == "head":
                    text = f"[?] {relation} ?"
                else:
                    text = f"{entity} {relation} [?]"
            return text
        
        # 原始自然语言问题格式
        if direction == "head":
            # 预测头实体
            text = f"{relation}的主体在{timestamp}时刻是什么?"
        else:
            # 预测尾实体
            text = f"{entity}在{timestamp}时刻的{relation}是什么?"
            
        if self.args.no_time:
            # 如果不使用时间
            if direction == "head":
                text = f"{relation}的主体是什么?"
            else:
                text = f"{entity}的{relation}是什么?"
                
        return text
        
    def _extract_semantic_features(self, sample, direction, use_cache=True):
        """
        提取样本的语义特征并使用随机投影降维
        
        Args:
            sample: (entity, relation, targets, timestamp) 元组
            direction: "head" 或 "tail" 表示预测方向
            use_cache: 是否使用缓存
            
        Returns:
            样本的降维后语义特征向量
        """
        # 创建缓存键，包含降维信息
        dim = self.feature_config.get("semantic_dim", 32)
        cache_key = f"sem_{sample[0]}_{sample[1]}_{direction}_{sample[3]}_{dim}"
        
        # 如果特征已缓存，直接返回
        if use_cache and cache_key in self.semantic_cache:
            return self.semantic_cache[cache_key]
            
        # 提取文本表示
        text = self._extract_sample_text(sample, direction)
        
        # 使用PLM编码器获取语义表示
        embedding = self.encoder.encode(text, use_cache=use_cache)
        
        # 应用随机投影降维
        projected_embedding = self._apply_random_projection(embedding)
        
        # 缓存结果
        if use_cache:
            self.semantic_cache[cache_key] = projected_embedding
            
        return projected_embedding
        
    def _extract_context_features(self, use_cache=True):
        """
        提取当前上下文特征（已选择样本的集合表示）并使用随机投影降维
        
        Args:
            use_cache: 是否使用缓存
            
        Returns:
            上下文特征向量
        """
        dim = self.feature_config.get("semantic_dim", 32)
        if not self.selected_samples:
            return torch.zeros(dim)  # 返回降维后维度的零向量
            
        # 创建缓存键，包含降维信息
        samples_str = "_".join([f"{s[0][0]}_{s[0][1]}_{s[1]}" for s in self.selected_samples])
        cache_key = f"ctx_{samples_str}_{dim}"
        
        # 如果特征已缓存，直接返回
        if use_cache and cache_key in self.context_cache:
            return self.context_cache[cache_key]
            
        # 提取所有已选择样本的文本表示
        sample_texts = []
        for sample, direction in self.selected_samples:
            text = self._extract_sample_text(sample, direction)
            sample_texts.append(text)
            
        # 如果没有样本，返回零向量
        if not sample_texts:
            return torch.zeros(dim)
            
        # 合并所有文本
        combined_text = " ".join(sample_texts)
        
        # 使用PLM编码器获取语义表示
        embedding = self.encoder.encode(combined_text, use_cache=use_cache)
        
        # 应用随机投影降维
        projected_embedding = self._apply_random_projection(embedding)
        
        # 缓存结果
        if use_cache:
            self.context_cache[cache_key] = projected_embedding
            
        return projected_embedding
        
    def _extract_interaction_features(self, use_cache=True):
        """
        提取上下文与查询之间的交互特征
        
        Args:
            use_cache: 是否使用缓存
            
        Returns:
            交互特征向量
        """
        if not self.selected_samples or self.current_query is None:
            method = self.feature_config["interaction_method"]
            return torch.zeros(16 if method == "attention" else 4)
            
        # 创建缓存键
        dim = self.feature_config.get("semantic_dim", 32)
        query_sample, query_direction = self.current_query
        query_key = f"{query_sample[0]}_{query_sample[1]}_{query_direction}"
        samples_str = "_".join([f"{s[0][0]}_{s[0][1]}_{s[1]}" for s in self.selected_samples])
        cache_key = f"inter_{query_key}_{samples_str}_{self.feature_config['interaction_method']}_{dim}"
        
        # 如果特征已缓存，直接返回
        if use_cache and cache_key in self.feature_cache:
            return self.feature_cache[cache_key]
            
        # 获取查询嵌入
        query_embedding = self._extract_semantic_features(query_sample, query_direction)
        
        # 获取每个已选择样本的嵌入
        sample_embeddings = []
        for sample, direction in self.selected_samples:
            embedding = self._extract_semantic_features(sample, direction)
            sample_embeddings.append(embedding)
            
        # 如果没有已选择的样本，返回零向量
        if not sample_embeddings:
            method = self.feature_config["interaction_method"]
            return torch.zeros(16 if method == "attention" else 4)
            
        # 计算交互特征
        if self.feature_config["interaction_method"] == "attention":
            # 使用注意力机制计算交互
            sample_embeddings_tensor = torch.stack(sample_embeddings)
            # 计算查询与每个样本的注意力分数
            attention_scores = F.softmax(
                torch.matmul(query_embedding.unsqueeze(0), sample_embeddings_tensor.T) / 
                torch.sqrt(torch.tensor(query_embedding.shape[0], dtype=torch.float32)),
                dim=1
            ).squeeze(0)
            
            # 使用注意力分数加权平均样本嵌入
            weighted_embedding = torch.matmul(attention_scores, sample_embeddings_tensor)
            
            # 计算统计特征
            max_score = attention_scores.max()
            min_score = attention_scores.min()
            mean_score = attention_scores.mean()
            std_score = attention_scores.std()
            
            # 构建特征向量
            stats_features = torch.tensor([max_score, min_score, mean_score, std_score])
            
            # 为了适应降维后的嵌入维度，将权重嵌入归一化并调整最终特征向量大小
            normalized_embedding = weighted_embedding / (weighted_embedding.norm() + 1e-8)
            
            # 调整为固定大小的特征向量 (16维)
            feature_dim = 16
            stats_dim = 4
            embedding_dim = feature_dim - stats_dim  # 为统计特征预留空间
            
            if normalized_embedding.shape[0] > embedding_dim:
                # 如果嵌入太大，取前几个维度
                truncated_embedding = normalized_embedding[:embedding_dim]
                feature_vector = torch.cat([truncated_embedding, stats_features])
            else:
                # 如果嵌入太小，填充零
                padding = torch.zeros(embedding_dim - normalized_embedding.shape[0])
                feature_vector = torch.cat([normalized_embedding, padding, stats_features])
                
        else:  # 使用余弦相似度
            # 计算查询与每个样本的余弦相似度
            similarities = torch.tensor([
                F.cosine_similarity(query_embedding.unsqueeze(0), embedding.unsqueeze(0))[0]
                for embedding in sample_embeddings
            ])
            
            # 计算统计特征
            max_sim = similarities.max()
            min_sim = similarities.min()
            mean_sim = similarities.mean()
            std_sim = similarities.std()
            
            # 构建特征向量
            feature_vector = torch.tensor([max_sim, min_sim, mean_sim, std_sim])
        
        # 缓存结果
        if use_cache:
            self.feature_cache[cache_key] = feature_vector
            
        return feature_vector
        
    def _calculate_diversity_feature(self, use_cache=True):
        """
        计算已选样本的多样性特征
        
        Args:
            use_cache: 是否使用缓存
            
        Returns:
            多样性特征向量
        """
        if not self.selected_samples or len(self.selected_samples) < 2:
            return torch.zeros(8)  # 样本数量不足，返回零向量
            
        # 创建缓存键
        dim = self.feature_config.get("semantic_dim", 32)
        samples_str = "_".join([f"{s[0][0]}_{s[0][1]}_{s[1]}" for s in self.selected_samples])
        cache_key = f"div_{samples_str}_{self.feature_config['diversity_method']}_{dim}"
        
        # 如果特征已缓存，直接返回
        if use_cache and cache_key in self.diversity_cache:
            return self.diversity_cache[cache_key]
            
        # 获取每个已选择样本的嵌入
        embeddings = []
        for sample, direction in self.selected_samples:
            embedding = self._extract_semantic_features(sample, direction)
            embeddings.append(embedding)
            
        # 转换为张量
        embeddings_tensor = torch.stack(embeddings)
        
        # 计算多样性
        if self.feature_config["diversity_method"] == "variance":
            # 计算方差作为多样性度量
            variance = embeddings_tensor.var(dim=0)
            mean_var = variance.mean()
            max_var = variance.max()
            min_var = variance.min()
            
            # 计算样本间的平均距离
            mean_distance = 0.0
            n_pairs = 0
            
            for i in range(len(embeddings)):
                for j in range(i+1, len(embeddings)):
                    dist = torch.norm(embeddings[i] - embeddings[j])
                    mean_distance += dist
                    n_pairs += 1
                    
            if n_pairs > 0:
                mean_distance /= n_pairs
                
            # 构建特征向量
            feature_vector = torch.tensor([
                mean_var.item(),
                max_var.item(),
                min_var.item(),
                mean_distance.item(),
                len(self.selected_samples) / self.max_steps,  # 已选样本比例
                0.0, 0.0, 0.0  # 填充
            ])
            
        else:  # 使用成对相似度
            # 计算所有样本对之间的余弦相似度
            similarities = []
            
            for i in range(len(embeddings)):
                for j in range(i+1, len(embeddings)):
                    sim = F.cosine_similarity(
                        embeddings[i].unsqueeze(0), 
                        embeddings[j].unsqueeze(0)
                    )[0]
                    similarities.append(sim)
                    
            if similarities:
                similarities_tensor = torch.tensor(similarities)
                mean_sim = similarities_tensor.mean()
                max_sim = similarities_tensor.max()
                min_sim = similarities_tensor.min()
                std_sim = similarities_tensor.std()
                
                # 构建特征向量
                feature_vector = torch.tensor([
                    1.0 - mean_sim.item(),  # 转换为多样性度量（1-相似度）
                    1.0 - min_sim.item(),   # 最大多样性
                    1.0 - max_sim.item(),   # 最小多样性
                    std_sim.item(),         # 多样性标准差
                    len(self.selected_samples) / self.max_steps,  # 已选样本比例
                    0.0, 0.0, 0.0  # 填充
                ])
            else:
                feature_vector = torch.zeros(8)
                
        # 缓存结果
        if use_cache:
            self.diversity_cache[cache_key] = feature_vector
            
        return feature_vector
    
    def set_mode(self, mode: str):
        """
        设置环境模式（训练、验证或测试）
        
        Args:
            mode: 环境模式，可以是 "train"、"val" 或 "test"
        """
        if mode not in ["train", "val", "test"]:
            raise ValueError(f"不支持的环境模式: {mode}，应为 'train'、'val' 或 'test'")
            
        self.mode = mode
        logger.info(f"环境模式已设置为: {mode}")
        
        # 重置环境状态
        self.reset()
    
    def init_random_projection(self):
        """初始化随机投影矩阵用于特征降维"""
        import torch.nn as nn
        
        # 获取目标维度
        target_dim = self.feature_config.get("semantic_dim", 32)
        # 源维度将在第一次获取特征时确定
        self.source_dim = None
        self.projection_matrix = None
        
        # 创建随机投影层
        self.random_projection = nn.Linear(384, target_dim, bias=False)  # 使用384作为初始占位符
        # 正交初始化以保持距离关系
        nn.init.orthogonal_(self.random_projection.weight)
        
    def _apply_random_projection(self, embedding):
        """应用随机投影进行降维"""
        # 如果是第一次调用，初始化投影矩阵
        if self.source_dim is None or self.projection_matrix is None:
            self.source_dim = embedding.shape[0]
            target_dim = self.feature_config.get("semantic_dim", 32)
            
            # 如果源维度变化，重新初始化投影层
            if self.random_projection.in_features != self.source_dim:
                self.random_projection = torch.nn.Linear(self.source_dim, target_dim, bias=False)
                torch.nn.init.orthogonal_(self.random_projection.weight)
            
            # 缓存投影矩阵以加速计算
            self.projection_matrix = self.random_projection.weight
            
        # 应用投影
        with torch.no_grad():
            projected = torch.matmul(embedding, self.projection_matrix.t())
            
        return projected 

    def _calculate_similarity(self, emb1, emb2):
        """计算两个嵌入向量的余弦相似度"""
        if emb1 is None or emb2 is None or emb1.shape[0] == 0 or emb2.shape[0] == 0:
            return torch.tensor(0.0) # 返回标量 0
        # 确保输入是二维的，以便 cosine_similarity 工作
        if emb1.dim() == 1:
            emb1 = emb1.unsqueeze(0)
        if emb2.dim() == 1:
            emb2 = emb2.unsqueeze(0)
        return F.cosine_similarity(emb1, emb2).squeeze()

    def _calculate_uniqueness(self, candidate_embedding):
        """计算候选示例与已选示例集合的独特性（1 - 最大相似度）"""
        if not self.selected_samples or candidate_embedding is None:
            return torch.tensor(1.0) # 如果没有已选样本，则独特性最高

        max_similarity = torch.tensor(-1.0)
        selected_embeddings = []
        for s_sample, s_direction in self.selected_samples:
             # 从缓存或计算获取已选样本的嵌入
             emb = self._extract_semantic_features(s_sample, s_direction)
             if emb is not None:
                 selected_embeddings.append(emb)

        if not selected_embeddings:
             return torch.tensor(1.0) # 如果无法获取已选样本嵌入，独特性为1

        selected_embeddings_tensor = torch.stack(selected_embeddings)

        # 计算候选嵌入与所有已选嵌入的相似度
        similarities = F.cosine_similarity(candidate_embedding.unsqueeze(0), selected_embeddings_tensor)
        max_similarity = similarities.max()

        # 独特性 = 1 - 最大相似度
        uniqueness = 1.0 - max_similarity
        return uniqueness 

    def set_historical_context(self, historical_samples):
        """
        设置或更新用于评估的历史上下文样本。
        
        Args:
            historical_samples: 从历史中检索到的样本列表
        """
        self.historical_samples = historical_samples or []
        logger.info(f"环境历史上下文已更新，样本数: {len(self.historical_samples)}")

    def _prepare_prompt_combined(self, query_sample, query_direction, historical_samples, active_selected_samples):
        """
        准备包含历史上下文和主动选择样本的提示。

        Args:
            query_sample: 当前查询样本元组
            query_direction: 预测方向
            historical_samples: 历史上下文样本列表
            active_selected_samples: RL Agent选择的当前时间步样本列表

        Returns:
            最终的提示字符串
        """
        # 1. 获取基础查询格式（不含任何上下文）
        try:
            # 修改调用方式，使用新的prepare_input签名
            base_query_prompt, _, _ = prepare_input(
                query_sample,  # 传递样本元组
                self.train_history,  # 传递搜索空间
                self.args,
                return_prompt=True
            )
        except Exception as e:
             logger.error(f"调用 prepare_input 获取基础提示时出错: {e}", exc_info=True)
             return "" # 返回空字符串表示失败

        # 2. 格式化历史样本
        historical_context_str = self._format_samples_for_prompt(historical_samples, self.args, context_type="historical")

        # 3. 整合基础查询和历史上下文 (历史在前，查询在后)
        prompt_with_history = historical_context_str + base_query_prompt

        # 4. 格式化主动选择的样本 (使用 active_learning 中的函数)
        try:
            # 延迟导入以处理可能的循环依赖或加载顺序问题
            from active_learning import integrate_active_samples
            final_prompt = integrate_active_samples(prompt_with_history, active_selected_samples, self.args)
        except ImportError:
             logger.error("无法导入 active_learning.integrate_active_samples")
             final_prompt = prompt_with_history # 回退
        except Exception as e:
            logger.error(f"调用 integrate_active_samples 添加主动样本时出错: {e}", exc_info=True)
            final_prompt = prompt_with_history # 回退

        # logger.debug(f"_prepare_prompt_combined: 生成的最终提示:\n{final_prompt}") # 可能太长，暂时注释掉
        return final_prompt

    def _format_samples_for_prompt(self, samples, args, context_type="historical"):
        """将样本列表格式化为字符串，以便添加到提示中。主要用于历史样本。"""
        prompt_part = ""
        if not samples:
            return prompt_part

        # 仅为历史样本添加标题
        title = ""
        if context_type == "historical":
            # 加一个换行符与前面内容隔开，再加标题，再加换行符与样本列表隔开
            title = "\n历史相关事件:\n"
        # 主动学习样本的标题和格式化由 integrate_active_samples 负责

        prompt_part += title

        # 遍历样本并格式化
        formatted_samples = []
        for item in samples:
            try:
                # 尝试解包，兼容 [(sample_tuple, direction), ...] 和 [sample_tuple, ...]
                if isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[0], tuple) and len(item[0]) == 4:
                    sample_data, _ = item # 方向信息在此处不用
                elif isinstance(item, tuple) and len(item) == 4:
                    sample_data = item
                else:
                    logger.warning(f"_format_samples_for_prompt: 未知的样本格式: {type(item)}，跳过")
                    continue

                entity, relation, targets, time = sample_data
                target = targets[0] if targets else "?" # 使用第一个目标

                # 构建单行样本字符串
                sample_str = ""
                if not args.no_time:
                    sample_str += f"{time}:"
                if args.label:
                    target_idx_str = str(targets.index(target)) if target != '?' and target in targets else '?'
                    sample_str += f"[{entity},{relation},{target_idx_str}. {target}]"
                else:
                    sample_str += f"[{entity},{relation},{target}]"
                formatted_samples.append(sample_str)
            except Exception as e:
                 logger.warning(f"_format_samples_for_prompt: 格式化样本 {item} 时出错: {e}", exc_info=False) # 减少日志噪音
                 continue # 跳过出错的样本

        # 用换行符连接所有格式化后的样本字符串
        if formatted_samples:
            prompt_part += "\n".join(formatted_samples)
            prompt_part += "\n" # 在样本列表末尾添加一个换行符，与后续内容分隔

        return prompt_part 