import torch
from torch import distributions, nn
import torch.nn.functional as F
from typing import *


class ConvolutionalBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            dilation: int = 1,
            n_layers: int = 2,
            activation_func: nn.Module = nn.ReLU(),
            downsample: nn.Module = nn.Identity()):
        super(ConvolutionalBlock, self).__init__()
        assert n_layers >= 1
        padding = (dilation * (kernel_size - 1.)) / 2.
        if padding == int(padding):
            padding = int(padding)
        else:
            raise ValueError(f'Padding must be an integer, but was {padding:0.2f}')
        if downsample is None:
            downsample = nn.Identity()

        layers = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                padding_mode='circular'
            ),
            nn.BatchNorm2d(out_channels),
            activation_func
        ]
        for i in range(n_layers - 1):
            layers.append(nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                padding_mode='circular'
            ))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(activation_func)
        # Remove final activation layer - to be applied after residual connection
        layers = layers[:-1]
        layers.append(downsample)
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ResidualModel(nn.Module):
    def __init__(self, conv_block_kwargs: Sequence[Dict]):
        super(ResidualModel, self).__init__()
        assert len(conv_block_kwargs) >= 1
        self.conv_blocks = nn.ModuleList()
        self.change_n_channels = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        self.activation_funcs = nn.ModuleList()
        for kwargs in conv_block_kwargs:
            self.conv_blocks.append(ConvolutionalBlock(**kwargs))
            if kwargs['in_channels'] != kwargs['out_channels']:
                self.change_n_channels.append(nn.Conv2d(kwargs['in_channels'], kwargs['out_channels'], 1))
            else:
                self.change_n_channels.append(nn.Identity())
            self.downsamplers.append(kwargs.get('downsample', nn.Identity()))
            self.activation_funcs.append(kwargs.get('activation_func', nn.ReLU()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for cb, cnc, d, a in zip(
                self.conv_blocks,
                self.change_n_channels,
                self.downsamplers,
                self.activation_funcs
        ):
            identity = x
            x = cb(x)
            x = x + d(cnc(identity))
            x = a(x)
            # print(x.shape)
        return x


class BasicActorCriticNetwork(nn.Module):
    def __init__(
            self,
            conv_block_kwargs: Sequence[Dict],
            use_adaptive_avg_pool: bool = True,
            fc_in_channels: Optional[int] = None,
            value_activation: Optional[nn.Module] = None,
            value_scale: float = 1.,
            value_shift: float = 0.
    ):
        super(BasicActorCriticNetwork, self).__init__()
        self.base = ResidualModel(conv_block_kwargs)
        if use_adaptive_avg_pool:
            self.prepare_for_fc = nn.AdaptiveAvgPool2d((1, 1))
            fc_in_channels = conv_block_kwargs[-1]['out_channels']
        else:
            self.prepare_for_fc = nn.Identity()
            if fc_in_channels is None:
                raise ValueError('If use_adaptive_avg_pool is False, fc_in_channels must be provided')
        self.actor = nn.Linear(fc_in_channels, 4)
        self.critic = nn.Linear(fc_in_channels, 1)
        self.value_activation = nn.Identity() if value_activation is None else value_activation
        self.value_scale = value_scale
        self.value_shift = value_shift

    def forward(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        base_out = self.prepare_for_fc(self.base(states))
        base_out = base_out.view(base_out.shape[0], -1)
        logits = self.actor(base_out)
        values = self.critic(base_out).squeeze(dim=-1)
        return logits, self.value_activation(values) * self.value_scale + self.value_shift

    def sample_action(self,
                      states: torch.Tensor,
                      available_actions_mask: Optional[torch.Tensor] = None,
                      train: bool = False):
        if train:
            logits, values = self.forward(states)
        else:
            with torch.no_grad():
                logits, values = self.forward(states)
        if available_actions_mask is not None:
            logits.masked_fill_(~available_actions_mask, float('-inf'))
        # In case all actions are masked, select one at random
        probs = F.softmax(torch.where(
            logits.isneginf().all(axis=-1, keepdim=True),
            torch.zeros_like(logits),
            logits
        ), dim=-1)
        m = distributions.Categorical(probs)
        sampled_actions = m.sample()
        if train:
            return sampled_actions, (logits, values)
        else:
            return sampled_actions

    def choose_best_action(self,
                           states: torch.Tensor,
                           available_actions_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        with torch.no_grad():
            logits, _ = self.forward(states)
            if available_actions_mask is not None:
                logits.masked_fill_(~available_actions_mask, float('-inf'))
            return logits.argmax(dim=-1)


class QModel(nn.Module):
    def __init__(
            self,
            conv_block_kwargs: Sequence[Dict],
            fc_in_channels: int,
            dueling_q: bool = False,
            use_adaptive_avg_pool: bool = True,
            value_activation: Optional[nn.Module] = None,
            value_scale: float = 1.,
            value_shift: float = 0.
    ):
        super(QModel, self).__init__()
        self.base = ResidualModel(conv_block_kwargs)
        self.dueling_q = dueling_q
        if use_adaptive_avg_pool:
            self.prepare_for_fc = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.prepare_for_fc = nn.Identity()
        fc_out_channels = fc_in_channels // 2
        if self.dueling_q:
            self.value = nn.Sequential(
                nn.Linear(fc_in_channels, fc_out_channels),
                nn.ReLU(),
                nn.Linear(fc_out_channels, 1)
            )
            self.advantage = nn.Sequential(
                nn.Linear(fc_in_channels, fc_out_channels),
                nn.ReLU(),
                nn.Linear(fc_out_channels, 4)
            )
        else:
            self.q = nn.Sequential(
                nn.Linear(fc_in_channels, fc_out_channels),
                nn.ReLU(),
                nn.Linear(fc_out_channels, 4)
            )
        self.value_activation = nn.Identity() if value_activation is None else value_activation
        self.value_scale = value_scale
        self.value_shift = value_shift

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        base_out = self.prepare_for_fc(self.base(states))
        base_out = base_out.view(base_out.shape[0], -1)
        if self.dueling_q:
            values = self.value(base_out)
            advantages = self.advantage(base_out)
            q_values = values + (advantages - advantages.mean(dim=-1, keepdims=True))
        else:
            q_values = self.q(base_out)
        q_values = self.value_activation(q_values) * self.value_scale + self.value_shift
        return q_values

    def get_state_value(self, states: torch.Tensor) -> torch.Tensor:
        if self.dueling_q:
            base_out = self.prepare_for_fc(self.base(states))
            base_out = base_out.view(base_out.shape[0], -1)
            return self.value(base_out)
        else:
            return self.forward(states).max(dim=-1, keepdim=True)[0]


class DeepQNetwork(nn.Module):
    def __init__(self,
                 optimizer_constructor,
                 epsilon: Optional[float] = None,
                 softmax_exploration: bool = False,
                 delayed_updates: bool = False,
                 double_q: bool = False,
                 dueling_q: bool = False,
                 tau: float = 5e-3,
                 *args, **kwargs
                 ):
        super(DeepQNetwork, self).__init__()

        self.epsilon = epsilon
        self.softmax_exploration = softmax_exploration
        self.delayed_updates = delayed_updates
        self.double_q = double_q
        self.dueling_q = dueling_q
        self.tau = tau
        assert 0. < tau < 1.

        self.q_1 = QModel(*args, dueling_q=dueling_q, **kwargs)
        self.q_1_opt = optimizer_constructor(self.q_1.parameters())
        if self.delayed_updates:
            self.target_q_1 = QModel(*args, dueling_q=dueling_q, **kwargs)
            for param, target_param in zip(self.q_1.parameters(), self.target_q_1.parameters()):
                target_param.data.copy_(param)
        if self.double_q:
            self.q_2 = QModel(*args, dueling_q=dueling_q, **kwargs)
            self.q_2_opt = optimizer_constructor(self.q_2.parameters())
            if self.delayed_updates:
                self.target_q_2 = QModel(*args, dueling_q=dueling_q, **kwargs)
                for param, target_param in zip(self.q_2.parameters(), self.target_q_2.parameters()):
                    target_param.data.copy_(param)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.q_1(states)

    def sample_action(self,
                      states: torch.Tensor,
                      epsilon: Optional[Union[float, torch.Tensor]] = None,
                      available_actions_mask: Optional[torch.Tensor] = None,
                      get_preds: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        epsilon = self.epsilon if epsilon is None else epsilon
        if epsilon is None:
            raise RuntimeError('epsilon and self.epsilon are both None')

        with torch.no_grad():
            q_vals = self.forward(states)
        if available_actions_mask is not None:
            q_vals.masked_fill_(~available_actions_mask, float('-inf'))
        if self.softmax_exploration:
            logits = q_vals
        else:
            logits = torch.where(
                q_vals.isneginf(),
                q_vals,
                torch.zeros_like(q_vals)
            )
        # In case all actions are masked, select one at random
        probs = F.softmax(torch.where(
            logits.isneginf().all(axis=-1, keepdim=True),
            torch.zeros_like(logits),
            logits * 10
        ), dim=-1)
        m = distributions.Categorical(probs)
        sampled_actions = m.sample()
        actions = torch.where(
            torch.rand(size=(q_vals.shape[0], 1), device=states.device) < epsilon,
            sampled_actions.unsqueeze(-1),
            q_vals.argmax(dim=-1, keepdim=True)
        )
        if get_preds:
            return actions, q_vals
        else:
            return actions

    def choose_best_action(self,
                           states: torch.Tensor,
                           available_actions_mask: Optional[torch.Tensor] = None,
                           get_preds: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return self.sample_action(states, 0., available_actions_mask, get_preds)

    def train_on_batch(self,
                       s_batch: torch.Tensor,
                       a_batch: torch.Tensor,
                       r_batch: torch.Tensor,
                       d_batch: torch.Tensor,
                       next_s_batch: torch.Tensor,
                       gamma: float) -> Dict:
        assert 0. < gamma <= 1.
        a_batch = a_batch.unsqueeze(-1)
        r_batch = r_batch.unsqueeze(-1)
        d_batch = d_batch.unsqueeze(-1)
        q_values = self.q_1(s_batch).gather(-1, a_batch)
        if self.double_q:
            q_values_2 = self.q_2(s_batch).gather(-1, a_batch)
        if self.delayed_updates:
            next_step_q_model = self.target_q_1
            if self.double_q:
                next_step_q_model_2 = self.target_q_2
        else:
            next_step_q_model = self.q_1
            if self.double_q:
                next_step_q_model_2 = self.q_2
        next_q_values = next_step_q_model(next_s_batch)
        if self.double_q:
            next_q_values_2 = next_step_q_model_2(next_s_batch)
            next_q_values = torch.minimum(
                next_q_values.max(dim=-1, keepdims=True)[0],
                next_q_values_2.max(dim=-1, keepdims=True)[0]
            )
        q_target = r_batch + (1 - d_batch) * gamma * next_q_values
        q_target = q_target.detach()

        loss_1 = F.smooth_l1_loss(q_values, q_target, reduction='mean')
        self.q_1_opt.zero_grad()
        loss_1.backward()
        self.q_1_opt.step()
        if self.double_q:
            loss_2 = F.smooth_l1_loss(q_values_2, q_target, reduction='mean')
            self.q_2_opt.zero_grad()
            loss_2.backward()
            self.q_2_opt.step()

        if self.delayed_updates:
            for param, target_param in zip(self.q_1.parameters(), self.target_q_1.parameters()):
                target_param.data.copy_(self.tau * param + (1. - self.tau) * target_param)
            if self.double_q:
                for param, target_param in zip(self.q_2.parameters(), self.target_q_2.parameters()):
                    target_param.data.copy_(self.tau * param + (1. - self.tau) * target_param)

        losses_dict = {
            'Q_1_loss': loss_1
        }
        if self.double_q:
            losses_dict['Q_2_loss'] = loss_2
        return losses_dict
