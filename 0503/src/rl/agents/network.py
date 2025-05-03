import logging
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_sequence

logger = logging.getLogger(__name__)


def initialize_weights(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=0.01)


def initialize_network(
    state_dim: int,
    action_dim: int,
    linear: bool = False,
    hidden_dim: int = 16,
    recurrent: bool = False,
    dropout: float = 0.0,
    normalize: bool = True,
    tanh: bool = False,
    requires_grad: bool = True,
) -> nn.Module:

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if linear:
        net = LinearNetwork(
            state_dim, action_dim, normalize, device
        )
    elif recurrent:
        net = LSTMNetwork(
            state_dim,
            action_dim,
            hidden_dim,
            dropout,
            normalize,
            tanh,
            device,
        )
    else:
        net = MLPNetwork(
            state_dim,
            action_dim,
            hidden_dim,
            dropout,
            normalize,
            tanh,
            device,
        )

    initialize_weights(net)

    for p in net.parameters():
        p.requires_grad = requires_grad

    return net.to(device)


# Welford's online variance algorithm
# https://math.stackexchange.com/questions/198336/how-to-calculate-standard-deviation-with-streaming-inputs


class RunningNorm(nn.Module):
    def __init__(self, num_features: int):
        super(RunningNorm, self).__init__()
        self.register_buffer("count", torch.tensor(0))
        self.register_buffer("mean", torch.zeros(num_features))
        self.register_buffer("M2", torch.zeros(num_features))
        self.register_buffer("eps", torch.tensor(1e-5))

    def track(self, x):
        x = x.detach().reshape(-1, x.shape[-1])

        self.count = self.count + x.shape[0]
        delta = x - self.mean
        self.mean.add_(delta.sum(dim=0) / self.count)
        self.M2.add_((delta * (x - self.mean)).sum(dim=0))

    def forward(self, x):
        # track stats only when training
        if self.training:
            self.track(x)

        # use stats to normalize current batch
        if self.count < 2:
            return x

        # 添加clamp操作确保var非负
        var = self.M2 / self.count + self.eps
        var = torch.clamp(var, min=1e-8)  # 使用clamp将var限制在最小1e-8，防止负数和精确的0
        
        x_normed = (x - self.mean) / torch.sqrt(var)
        return x_normed


class LinearNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        normalize: bool,
        device: str,
    ):
        super(LinearNetwork, self).__init__()
        input_dim = state_dim + action_dim
        self.net = nn.Linear(input_dim, 1)
        self.normalize = normalize
        # self.norm = RunningNorm(state_dim + action_dim)  # 注释掉 RunningNorm
        if self.normalize:  # 只有在需要归一化时才创建 LayerNorm
            self.norm = nn.LayerNorm(state_dim + action_dim)
        self.device = device

    def forward(self, states, action_space):
        # 处理action_space可能是元组的情况
        if isinstance(action_space, tuple) and len(action_space) == 3:
            action_features = action_space[0]  # 提取特征张量
        else:
            action_features = action_space  # 原始格式，直接使用
        
        if isinstance(states, list):
            state = torch.stack([s[-1] for s in states])
        elif states.dim() == 3:
            state = states[:, -1, :]
        elif states.dim() == 2:
            state = states[-1].unsqueeze(0)
            if action_features.dim() == 2:
                action_features = action_features.unsqueeze(0)
        else:
            state = states.unsqueeze(0)
            if action_features.dim() == 2:
                action_features = action_features.unsqueeze(0)

        state = state.to(self.device)
        action_features = action_features.to(self.device)

        if action_features.shape[1] == 0:
            return torch.zeros((state.shape[0], 0), device=self.device)

        num_actions = action_features.shape[1]
        state_aligned = state.unsqueeze(1).expand(-1, num_actions, -1)
        state_action_space = torch.cat((state_aligned, action_features), dim=2)

        if self.normalize:
            # LayerNorm 直接应用于最后一维 (特征维度)
            state_action_space = self.norm(state_action_space)

        logits = self.net(state_action_space)
        final_q_values = logits.squeeze(-1)
        
        if states.dim() == 2 and action_features.dim() == 3 and action_features.shape[0] == 1:
            final_q_values = final_q_values.squeeze(0)
        elif states.dim() == 1:
            final_q_values = final_q_values.squeeze(0)

        return final_q_values


class MLPNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        dropout: float,
        normalize: bool,
        tanh: bool,
        device: str,
    ):
        super(MLPNetwork, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.normalize = normalize
        # self.norm = RunningNorm(state_dim + action_dim)  # 注释掉 RunningNorm
        if self.normalize:  # 只有在需要归一化时才创建 LayerNorm
            self.norm = nn.LayerNorm(state_dim + action_dim)
        self.tanh = tanh

        input_dim = state_dim + action_dim
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.device = device

    def forward(self, states, action_space):
        # 处理action_space可能是元组的情况
        if isinstance(action_space, tuple) and len(action_space) == 3:
            action_features = action_space[0]
        else:
            action_features = action_space
        
        if not torch.is_tensor(action_features):
            logger.error(f"MLPNetwork: action_features is not a Tensor! Type: {type(action_features)}")
            return torch.zeros((1, 0), device=self.device)
        
        is_batched_actions = action_features.dim() == 3
        batch_size = action_features.shape[0] if is_batched_actions else 1
        
        # 处理 action_features 可能为 [N,D] 或 [B,N,D]
        if action_features.dim() == 2 and not is_batched_actions:
            num_actions = action_features.shape[0]
            action_features = action_features.unsqueeze(0)  # [1, N, D]
        elif is_batched_actions:
            num_actions = action_features.shape[1]
        else:  # 维度不为 2 或 3
            logger.error(f"MLPNetwork: Unexpected action_features dim: {action_features.dim()}")
            return torch.zeros((batch_size, 0), device=self.device)  # 返回[B, 0]
        
        if num_actions == 0:
            logger.warning("MLPNetwork: Received empty action features (num_actions=0).")
            return torch.zeros((batch_size, 0), device=self.device)
        # --- action_features 处理结束, 确保为 [B, N, D] ---

        # --- 处理 states (确保能处理 1D) ---
        if not torch.is_tensor(states):
            logger.error(f"MLPNetwork: states is not a Tensor! Type: {type(states)}")
            return torch.zeros((batch_size, num_actions), device=self.device)

        # state_processed: 目标形状 [batch_size, state_dim]
        if states.dim() == 1:
            # --- 添加处理 1D 状态的逻辑 ---
            # 这是测试/评估时最可能的情况: [state_dim]
            state_processed = states.unsqueeze(0)  # 变成 [1, state_dim]
            # 如果动作是批量的 (虽然评估时通常不是), 扩展 batch 维
            if batch_size > 1 and state_processed.shape[0] == 1:
                state_processed = state_processed.expand(batch_size, -1)  # [B, state_dim]
            # --- 1D 处理结束 ---
        elif states.dim() == 2:
            # 可能是 [batch_size, state_dim] (来自 optimize) 或 [seq_len, state_dim] (也可能来自 optimize)
            if states.shape[0] == batch_size:  # 假设是 [batch_size, state_dim]
                state_processed = states
            else:  # 假设是 [seq_len, state_dim]，取最后一个时间步
                state_processed = states[-1, :].unsqueeze(0)  # [1, state_dim]
                if batch_size > 1:  # 如果动作是批量的，扩展
                    state_processed = state_processed.expand(batch_size, -1)  # [B, state_dim]
        elif states.dim() == 3:
            # 假设是 [batch_size, seq_len, state_dim] (来自 optimize)
            state_processed = states[:, -1, :]  # 取最后一个时间步 [batch_size, state_dim]
        else:
            logger.error(f"MLPNetwork: Unhandled state dimension: {states.dim()}")
            return torch.zeros((batch_size, num_actions), device=self.device)
        # --- states 处理结束, state_processed 应为 [B, state_dim] ---

        # --- 检查 batch size 匹配 ---
        if state_processed.shape[0] != action_features.shape[0]:
            logger.error(f"MLPNetwork: Mismatched batch sizes after processing. State: {state_processed.shape}, Actions: {action_features.shape}")
            return torch.zeros((action_features.shape[0], num_actions), device=self.device)

        # --- 继续后续计算 ---
        state_for_mlp = state_processed.to(self.device)  # 可以用新名字或覆盖 state
        action_features = action_features.to(self.device)

        state_aligned = state_for_mlp.unsqueeze(1).expand(-1, num_actions, -1)
        state_action_space = torch.cat((state_aligned, action_features), dim=2)

        if self.normalize:
            # LayerNorm 直接应用于最后一维 (特征维度)
            try:
                state_action_space = self.norm(state_action_space)
            except Exception as norm_err:
                logger.error(f"Error during LayerNorm: {norm_err}", exc_info=True)
                return torch.zeros((state_for_mlp.shape[0], num_actions), device=self.device)

        state_action_space = self.dropout(state_action_space)
        x = self.input_layer(state_action_space)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.hidden_layer(x)
        x = F.relu(x)
        x = self.dropout(x)
        logits = self.output_layer(x)

        if self.tanh:
            logits = torch.tanh(logits)

        final_q_values = logits.squeeze(-1)  # [B, N]

        # --- 输出形状调整 (保持不变) ---
        # 如果原始输入 states 是一维的，且输出 batch 维是 1，则去掉 batch 维
        if states.dim() == 1 and final_q_values.shape[0] == 1:
            final_q_values = final_q_values.squeeze(0)  # [N]

        return final_q_values


class LSTMNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        dropout: float,
        normalize: bool,
        tanh: bool,
        device: str,
    ):
        super(LSTMNetwork, self).__init__()

        self.normalize = normalize
        # self.norm = RunningNorm(state_dim + action_dim) # 注释掉 RunningNorm
        if self.normalize:  # 只有在需要归一化时才创建 LayerNorm
            self.norm = nn.LayerNorm(state_dim + action_dim)
        self.tanh = tanh

        self.device = device
        self.action_dim = action_dim
        self.state_dim = state_dim

        if state_dim == 0:
            logger.warning(
                "got state_dim=0, LSTM not processing any state information..."
            )

        self.lstm = nn.LSTM(state_dim, hidden_dim, batch_first=True)
        self.mlp = MLPNetwork(
            hidden_dim,
            action_dim,
            hidden_dim,
            dropout,
            normalize,
            tanh,
            device,
        )

    def forward(self, states, action_space):
        # 处理action_space可能是元组的情况
        if isinstance(action_space, tuple) and len(action_space) == 3:
            action_features = action_space[0]  # 提取特征张量
        else:
            action_features = action_space  # 原始格式，直接使用
        
        is_batched = action_features.dim() == 3
        action_features = action_features.to(self.device)
        
        # 处理不同形式的状态输入
        if isinstance(states, list):
            # 列表形式的状态序列
            batch_size = len(states)
            states = [s.to(self.device) for s in states]
            
            # 应用LayerNorm（如果启用）
            if self.normalize and hasattr(self, 'norm'):
                normalized_states = []
                for s in states:
                    # 确保输入维度正确
                    state_dim = s.shape[-1]
                    s_reshaped = s.view(-1, state_dim)
                    norm_s = self.norm(s_reshaped).view(s.shape)
                    normalized_states.append(norm_s)
                states = normalized_states
                
            # 打包序列
            packed_states = pack_sequence(states, enforce_sorted=False)
            lstm_out, _ = self.lstm(packed_states)
            
            # 解包并获取最后时间步的输出
            unpacked_out, lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
            state_repr = unpacked_out[torch.arange(batch_size), lengths - 1]
            
        elif states.dim() == 3:
            # [batch_size, seq_len, state_dim] 形式的批量序列
            batch_size = states.shape[0]
            states = states.to(self.device)
            
            # 应用LayerNorm（如果启用）
            if self.normalize and hasattr(self, 'norm'):
                batch_size, seq_len, state_dim = states.shape
                states_flat = states.reshape(-1, state_dim)
                norm_states_flat = self.norm(states_flat)
                states = norm_states_flat.reshape(batch_size, seq_len, state_dim)
                
            lstm_out, _ = self.lstm(states)
            state_repr = lstm_out[:, -1, :]  # 获取最后时间步的输出
            
        elif states.dim() == 2:
            # [seq_len, state_dim] 形式的单个序列
            states = states.to(self.device)
            states_batch = states.unsqueeze(0)  # 添加批次维度
            
            # 应用LayerNorm（如果启用）
            if self.normalize and hasattr(self, 'norm'):
                seq_len, state_dim = states.shape
                states_flat = states.reshape(-1, state_dim)
                norm_states_flat = self.norm(states_flat)
                states_batch = norm_states_flat.reshape(1, seq_len, state_dim)
                
            lstm_out, _ = self.lstm(states_batch)
            state_repr = lstm_out[0, -1, :]  # 获取批次中唯一序列的最后时间步输出
            
            # 如果action_features是2D的，将其增加一个批次维度
            if action_features.dim() == 2:
                action_features = action_features.unsqueeze(0)
                is_batched = False
                
        else:
            # 修正：补全字符串和错误信息
            raise ValueError(f"LSTMNetwork expects states with history (2D or 3D tensor, or list of tensors), but got dim={states.dim()}")
            
        # 使用MLP网络处理状态表示和动作特征
        if not is_batched:
            state_repr = state_repr.unsqueeze(0)
            
        state_repr = self.mlp(state_repr, action_features)
        
        return state_repr
