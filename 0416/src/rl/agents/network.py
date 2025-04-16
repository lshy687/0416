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

        # biased var estimator
        var = self.M2 / self.count + self.eps
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
        self.norm = RunningNorm(state_dim + action_dim)
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
            original_shape = state_action_space.shape
            state_action_space_flat = state_action_space.view(-1, original_shape[-1])
            state_action_space_norm = self.norm(state_action_space_flat)
            state_action_space = state_action_space_norm.view(original_shape)

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
        self.norm = RunningNorm(state_dim + action_dim)
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
            original_shape = state_action_space.shape
            state_action_space_flat = state_action_space.view(-1, original_shape[-1])
            state_action_space_norm = self.norm(state_action_space_flat)
            state_action_space = state_action_space_norm.view(original_shape)

        state_action_space = self.dropout(state_action_space)
        x = self.input_layer(state_action_space)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        x = self.hidden_layer(x)
        x = F.leaky_relu(x)
        logits = self.output_layer(x)
        if self.tanh:
            logits = torch.tanh(logits)
            
        final_q_values = (logits + task_offset).squeeze(-1)
        
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
        if state_dim == 0:
            logger.warning(
                "got state_dim=0, LSTM not processing any state information..."
            )

        self.normalize = normalize
        self.norm = RunningNorm(state_dim)
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
        self.device = device

    def forward(self, states, action_space):
        # 处理action_space可能是元组的情况
        if isinstance(action_space, tuple) and len(action_space) == 3:
            action_features = action_space[0]  # 提取特征张量
        else:
            action_features = action_space  # 原始格式，直接使用
        
        is_batched = isinstance(states, list) or states.dim() == 3
        batch_size = len(states) if isinstance(states, list) else states.shape[0]
        
        if isinstance(states, list):
            states_tensor_list = [s.to(self.device) for s in states]
            if self.normalize:
                states_normed = []
                for s_tensor in states_tensor_list:
                    s_flat = s_tensor.view(-1, s_tensor.shape[-1])
                    s_norm_flat = self.norm(s_flat)
                    states_normed.append(s_norm_flat.view(s_tensor.shape))
                states_packed = pack_sequence(states_normed, enforce_sorted=False)
            else:
                states_packed = pack_sequence(states_tensor_list, enforce_sorted=False)

        elif states.dim() == 3:
            states = states.to(self.device)
            if self.normalize:
                original_shape = states.shape
                states_flat = states.view(-1, states.shape[-1])
                states_norm_flat = self.norm(states_flat)
                states = states_norm_flat.view(original_shape)
            states_packed = states
            
        elif states.dim() == 2:
            assert action_features.dim() == 2
            states = states.to(self.device)
            if self.normalize:
                states = self.norm(states) 
            states_packed = states.unsqueeze(0)
            action_features = action_features.unsqueeze(0)
        else:
            raise ValueError("LSTMNetwork expects states with history (L x D or B x L x D)")

        lstm_output, (h_n, c_n) = self.lstm(states_packed)
        
        if isinstance(states_packed, torch.nn.utils.rnn.PackedSequence):
            lstm_output_unpacked, lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_output, batch_first=True)
            last_lstm_output = lstm_output_unpacked[torch.arange(batch_size), lengths - 1]
            state_repr = last_lstm_output
        else:
            state_repr = lstm_output[:, -1, :]
            
        # 将提取的状态表示传递给MLP网络，同时传递动作特征
        mlp_output = self.mlp(state_repr, action_features)
        
        # 保留原来对特殊情况的处理
        if states.dim() == 2:
            mlp_output = mlp_output.squeeze(0)
            
        return mlp_output
