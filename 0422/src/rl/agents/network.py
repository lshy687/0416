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

        # 添加检查M2和count的打印语句
        print(f"DEBUG RunningNorm Check: count={self.count.item()}")
        print(f"DEBUG RunningNorm Check: M2 contains NaN: {torch.isnan(self.M2).any()}, min/max: {self.M2.min().item() if self.M2.numel() > 0 else 'N/A'}, {self.M2.max().item() if self.M2.numel() > 0 else 'N/A'}")

        # biased var estimator
        var = self.M2 / self.count + self.eps
        print(f"DEBUG RunningNorm: var contains NaN: {torch.isnan(var).any()}, min/max: {var.min().item() if var.numel() > 0 else 'N/A'}, {var.max().item() if var.numel() > 0 else 'N/A'}")
        
        # 添加clamp操作确保var非负
        var = torch.clamp(var, min=1e-8)  # 使用clamp将var限制在最小1e-8，防止负数和精确的0
        print(f"DEBUG RunningNorm (After Clamp): var contains NaN: {torch.isnan(var).any()}, min/max: {var.min().item() if var.numel() > 0 else 'N/A'}, {var.max().item() if var.numel() > 0 else 'N/A'}")
        
        x_normed = (x - self.mean) / torch.sqrt(var)
        print(f"DEBUG RunningNorm: x_normed contains NaN: {torch.isnan(x_normed).any()}, min/max: {x_normed.min().item() if x_normed.numel() > 0 else 'N/A'}, {x_normed.max().item() if x_normed.numel() > 0 else 'N/A'}")
        
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
            action_features = action_space[0]  # 提取特征张量
        else:
            action_features = action_space  # 原始格式，直接使用
        
        if isinstance(states, list):
            state = torch.stack([s[-1] for s in states])
        elif states.dim() == 3:
            state = states[:, 0, :]
        elif states.dim() == 2:
            assert action_features.dim() == 2
            state = states[-1].unsqueeze(0)
            action_features = action_features.unsqueeze(0)

        state = state.to(self.device)
        action_features = action_features.to(self.device)

        is_batched = action_features.dim() == 3 

        if not is_batched:
            if state.dim() > 1:
                state = state[-1]
            state = state.unsqueeze(0)
            action_features = action_features.unsqueeze(0)
            batch_size = 1
        else:
            batch_size = action_features.shape[0]
            if isinstance(states, list): 
                state = torch.stack([s[-1] for s in states])
            elif state.dim() == 3:
                state = state[:, -1, :]

        num_actions = action_features.shape[1]
        state_dim = state.shape[1]
        action_dim = action_features.shape[2]

        if num_actions == 0:
            logger.warning("MLPNetwork: action_space is empty!")
            return torch.zeros((batch_size, 0), device=self.device)

        task_offset = torch.zeros(1).to(state.device)

        state_aligned = state.unsqueeze(1).expand(-1, num_actions, -1)
        
        state_action_space = torch.cat((state_aligned, action_features), dim=2)

        if self.normalize:
            # LayerNorm 直接应用于最后一维 (特征维度)
            state_action_space = self.norm(state_action_space)
            # print(f"DEBUG Network (After LayerNorm): Contains NaN: {torch.isnan(state_action_space).any()}, min/max: {state_action_space.min().item() if state_action_space.numel() > 0 else 'N/A'}, {state_action_space.max().item() if state_action_space.numel() > 0 else 'N/A'}")
            
        state_action_space = self.dropout(state_action_space)
        x = self.input_layer(state_action_space)
        # print(f"DEBUG Network (Input Layer Output): Contains NaN: {torch.isnan(x).any()}, min/max: {x.min().item() if x.numel() > 0 else 'N/A'}, {x.max().item() if x.numel() > 0 else 'N/A'}")
        
        x = F.leaky_relu(x)
        # print(f"DEBUG Network (After ReLU 1): Contains NaN: {torch.isnan(x).any()}, min/max: {x.min().item() if x.numel() > 0 else 'N/A'}, {x.max().item() if x.numel() > 0 else 'N/A'}")
        
        x = self.dropout(x)
        x = self.hidden_layer(x)
        # print(f"DEBUG Network (Hidden Layer Output): Contains NaN: {torch.isnan(x).any()}, min/max: {x.min().item() if x.numel() > 0 else 'N/A'}, {x.max().item() if x.numel() > 0 else 'N/A'}")
        
        x = F.leaky_relu(x)
        # print(f"DEBUG Network (After ReLU 2): Contains NaN: {torch.isnan(x).any()}, min/max: {x.min().item() if x.numel() > 0 else 'N/A'}, {x.max().item() if x.numel() > 0 else 'N/A'}")
        
        logits = self.output_layer(x)
        # print(f"DEBUG Network (Logits Output): Contains NaN: {torch.isnan(logits).any()}, min/max: {logits.min().item() if logits.numel() > 0 else 'N/A'}, {logits.max().item() if logits.numel() > 0 else 'N/A'}")
        
        if self.tanh:
            logits = torch.tanh(logits)
            # print(f"DEBUG Network (After Tanh): Contains NaN: {torch.isnan(logits).any()}, min/max: {logits.min().item() if logits.numel() > 0 else 'N/A'}, {logits.max().item() if logits.numel() > 0 else 'N/A'}")
            
        final_q_values = (logits + task_offset).squeeze(-1)
        # print(f"DEBUG Network (Final Q Values): Contains NaN: {torch.isnan(final_q_values).any()}, min/max: {final_q_values.min().item() if final_q_values.numel() > 0 else 'N/A'}, {final_q_values.max().item() if final_q_values.numel() > 0 else 'N/A'}")
        
        if not is_batched:
            final_q_values = final_q_values.squeeze(0)

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
            raise ValueError("LSTMNetwork expects states with history (2D or 3D tensor, or list of tensors)")
            
        # 使用MLP网络处理状态表示和动作特征
        if not is_batched:
            state_repr = state_repr.unsqueeze(0)
            
        state_repr = self.mlp(state_repr, action_features)
        
        return state_repr
