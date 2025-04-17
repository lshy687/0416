# 强化学习模块

本目录包含时序知识图谱上下文学习项目的强化学习模块。

## 文件结构和关系

- `base_environment.py`: 包含`BaseEnvironment`抽象基类，定义了所有强化学习环境必须实现的接口和方法。
- `tkg_environment.py`: 实现了特定于时序知识图谱(TKG)的环境`TKGEnvironment`，该类继承自`BaseEnvironment`。
- `agents/`: 包含不同的强化学习代理实现，如DQN代理。
- `misc_utils.py`: 包含一些常用工具函数。

## 类的层次结构

```
BaseEnvironment (抽象基类)
└── TKGEnvironment (时序知识图谱环境)
```

## 用法示例

```python
from src.rl.base_environment import BaseEnvironment  # 导入基类
from src.rl.tkg_environment import TKGEnvironment    # 导入时序知识图谱环境
from src.rl.agents.dqn_agent import DQNAgent         # 导入DQN代理

# 创建环境
environment = TKGEnvironment(
    test_data=train_data,                  # 训练数据
    model=model,                           # 预测模型
    tokenizer=tokenizer,                   # 分词器
    args=args,                             # 参数
    state_repr=["query_features", "context_features", "interaction_features", "diversity_features", "curr_step"],
    max_steps=args.active_samples,         # 最大步数
    reward_scale=1.0,                      # 奖励缩放因子
)

# 创建代理
agent = DQNAgent(
    env=environment,                       # 环境
    train_steps=400,                       # 训练步数
    batch_size=32,                         # 批次大小
    replay_memory_size=1000,               # 经验回放缓冲区大小
)

# 训练代理
agent.train()
```

# 环境特征表示

TKGEnvironment支持灵活的环境特征表示配置，可以通过`state_repr`参数和`feature_config`参数进行定制。

## 可用特征组件

环境特征可以包含以下组件，可以通过`state_repr`参数选择需要的组件:

1. **sample_features**: 样本基础特征，包含实体ID、关系ID、时间戳等基本信息的64维向量
2. **query_features**: 查询语义特征，使用BERT模型提取的384维语义表示
3. **similarity**: 查询-样本相似度，1维余弦相似度
4. **history**: 历史信息，包含已选样本比例、MRR历史等信息的32维向量
5. **curr_step**: 当前步骤信息，包含当前步数比例和时间戳索引的16维向量
6. **context_features**: 当前上下文特征，所有已选样本的集合BERT表示，384维向量
7. **interaction_features**: 上下文与查询的交互特征，可以是注意力评分(16维)或余弦相似度统计(4维)
8. **diversity_features**: 已选样本的多样性度量，包含语义空间分散程度的8维向量

## 特征配置选项

通过`feature_config`参数可以进一步配置特征提取方式:

```python
feature_config = {
    # 是否使用CLS作为句子表示（True使用CLS向量，False使用平均池化）
    "use_cls_pooling": True,
    
    # 多样性计算方法
    # "variance": 使用嵌入方差和样本间平均距离作为多样性度量
    # "pairwise": 使用样本间的余弦相似度作为多样性度量
    "diversity_method": "variance",
    
    # 交互特征计算方法
    # "attention": 使用注意力机制计算查询与上下文的交互，生成16维特征
    # "cosine": 使用余弦相似度统计量作为交互特征，生成4维特征
    "interaction_method": "cosine"
}
```

## 消融实验配置示例

以下是几种不同的特征组合示例，可用于消融实验:

1. **基础特征** (仅使用步骤信息)
```python
state_repr=["curr_step"]
```

2. **查询特征**
```python
state_repr=["query_features", "curr_step"]
```

3. **上下文特征**
```python
state_repr=["context_features", "curr_step"]
```

4. **交互特征**
```python
state_repr=["query_features", "interaction_features", "curr_step"]
```

5. **多样性特征**
```python
state_repr=["diversity_features", "curr_step"]
```

6. **完整特征集**
```python
state_repr=[
    "query_features", "context_features", "interaction_features", "diversity_features", "curr_step"
]
```

## 初始化示例

```python
# 创建环境，使用完整特征集和自定义配置
environment = TKGEnvironment(
    test_data=train_data,                  # 训练数据
    model=model,                           # 预测模型
    tokenizer=tokenizer,                   # 分词器
    args=args,                             # 参数
    state_repr=["query_features", "context_features", "interaction_features", "diversity_features", "curr_step"],
    feature_config={
        "use_cls_pooling": True,
        "diversity_method": "pairwise",
        "interaction_method": "attention"
    },
    max_steps=args.active_samples,         # 最大步数
    reward_scale=1.0,                      # 奖励缩放因子
)
``` 