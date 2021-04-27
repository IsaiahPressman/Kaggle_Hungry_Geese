import torch
from torch import distributions, nn
import torch.nn.functional as F
from typing import *

from .conv_blocks import ResidualModel
from ..config import *


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


class FullConvActorCriticNetwork(nn.Module):
    def __init__(
            self,
            cross_normalize_value: bool = True,
            value_activation: Optional[nn.Module] = None,
            use_separate_action_value_heads: bool = False,
            value_scale: float = 1.,
            value_shift: float = 0.,
            **residual_model_kwargs
    ):
        super(FullConvActorCriticNetwork, self).__init__()
        self.use_separate_action_value_heads = use_separate_action_value_heads
        self.base = ResidualModel(**residual_model_kwargs)
        self.base_out_channels = residual_model_kwargs['conv_block_kwargs'][-1]['out_channels']
        if self.use_separate_action_value_heads:
            self.actor = nn.Linear(self.base_out_channels, 4 * N_PLAYERS)
            self.critic = nn.Linear(self.base_out_channels, 1 * N_PLAYERS)
        else:
            self.actor = nn.Linear(self.base_out_channels, 4)
            self.critic = nn.Linear(self.base_out_channels, 1)
        # activation = residual_model_kwargs['conv_block_kwargs'][-1].get('activation', nn.ReLU)
        # fc_channels = int(self.base_out_channels / 4)
        # self.actor = nn.Sequential(
        #     nn.Linear(self.base_out_channels, fc_channels),
        #     activation(inplace=True),
        #     nn.Linear(fc_channels, 4)
        # )
        # self.critic = nn.Sequential(
        #     nn.Linear(self.base_out_channels, fc_channels),
        #     activation(inplace=True),
        #     nn.Linear(fc_channels, 1)
        # )
        self.cross_normalize_value = cross_normalize_value
        if self.cross_normalize_value:
            if value_activation is not None:
                print('WARNING: Setting value_activation has no effect while cross_normalize_value is True')
            if value_scale != 1.:
                print('WARNING: Setting value_scale has no effect while cross_normalize_value is True')
            if value_shift != 0.:
                print('WARNING: Setting value_shift has no effect while cross_normalize_value is True')
            self.value_activation = None
            self.value_scale = None
            self.value_shift = None
        else:
            self.value_activation = nn.Identity() if value_activation is None else value_activation
            self.value_scale = value_scale
            self.value_shift = value_shift

    def forward(self,
                states: torch.Tensor,
                head_locs: torch.Tensor,
                still_alive: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        @param states: Tensor of shape (batch_size, n_channels, n_rows, n_cols) representing the board
        @param head_locs: Tensor of shape (batch_size, n_geese), containing the integer index locations of the goose
            heads whose actions/values should be returned
        @param still_alive: Booleran tensor of shape (batch_size, n_geese), where True values indicate which geese are
            still alive and taking actions
        @return (policy, value):
             policy_logits: Tensor of shape (batch_size, n_geese, 4), representing the action distribution per goose
             value: Tensor of shape (batch_size, n_geese), representing the predicted value per goose
        """
        batch_size, n_geese = head_locs.shape
        base_out = self.base(states)
        if base_out.shape[-2:] != states.shape[-2:]:
            raise RuntimeError(f'Fully convolutional networks must use padding so that the input and output sizes match'
                               f'Got input size: {states.shape} and output size: {base_out.shape}')
        # Reshape to tensor of shape (batch_size, n_channels, n_rows * n_cols)
        # Then, swap axes so that channels are last (batch_size, n_rows * n_cols, n_channels)
        base_out = base_out.reshape(base_out.shape[0], self.base_out_channels, -1).transpose(-2, -1)
        batch_indices = torch.arange(batch_size).repeat_interleave(n_geese)
        head_indices = torch.where(
            still_alive,
            head_locs,
            torch.zeros_like(head_locs)
        ).view(-1).to(dtype=torch.int64)
        # Base_out_indexed (before .view()) is a tensor of shape (batch_size * n_geese, n_channels)
        # After .view(), base_out_indexed has shape (batch_size, n_geese, n_channels)
        base_out_indexed = base_out[batch_indices, head_indices].view(batch_size, n_geese, -1)
        actor_out = self.actor(base_out_indexed)
        critic_out = self.critic(base_out_indexed)
        if self.use_separate_action_value_heads:
            goose_indices = torch.arange(n_geese).repeat(batch_size)
            actor_out = actor_out.view(batch_size, n_geese, n_geese, 4)
            critic_out = critic_out.view(batch_size, n_geese, n_geese, 1)
            actor_out = actor_out[batch_indices, goose_indices, goose_indices].view(batch_size, n_geese, 4)
            critic_out = critic_out[batch_indices, goose_indices, goose_indices].view(batch_size, n_geese, 1)
        logits = torch.where(
            still_alive.unsqueeze(-1),
            actor_out,
            torch.zeros_like(actor_out)
        )
        values = torch.where(
            still_alive.unsqueeze(-1),
            critic_out,
            torch.zeros_like(critic_out)
        ).squeeze(dim=-1)
        if self.cross_normalize_value:
            if n_geese != 4.:
                raise RuntimeError('cross_normalize_value still needs to be implemented for n_geese != 4')
            values.masked_fill_(~still_alive, float('-inf'))
            win_probs = torch.softmax(values, dim=-1)
            remaining_rewards = torch.linspace(0., 1., n_geese, dtype=states.dtype, device=states.device)
            remaining_rewards_min = remaining_rewards[-still_alive.sum(dim=-1)].unsqueeze(-1)
            remaining_rewards_var = 1. - remaining_rewards_min
            values = remaining_rewards_min + win_probs * remaining_rewards_var
            # TODO: This is a hacky solution - there should be a more elegant way to do this for any n_geese_remaining?
            values = torch.where(
                still_alive.sum(dim=-1, keepdim=True) == 4,
                values * 2.,
                values
            )
            values = torch.where(
                still_alive.sum(dim=-1, keepdim=True) == 3,
                values * 1.2,
                values
            )
            # After the value multiplication, the winning goose may have a value > 1
            # This "excess" value is redistributed evenly among the other geese
            max_vals = values.max(dim=-1, keepdim=True)[0]
            values = torch.where(
                max_vals > 1.,
                torch.where(
                    values == max_vals,
                    values - (max_vals - 1.),
                    values + (max_vals - 1.) / (still_alive.sum(dim=-1, keepdim=True) - 1.)
                ),
                values
            )
            # Rescale values from the range [0., 1] to the range [-1., 1]
            return logits, 2. * values - 1.
        else:
            return logits, self.value_activation(values) * self.value_scale + self.value_shift

    def sample_action(self,
                      states: torch.Tensor,
                      head_locs: torch.Tensor,
                      still_alive: torch.Tensor,
                      available_actions_mask: Optional[torch.Tensor] = None,
                      train: bool = False):
        if train:
            logits, values = self.forward(states, head_locs, still_alive)
        else:
            with torch.no_grad():
                logits, values = self.forward(states, head_locs, still_alive)
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
                           head_locs: torch.Tensor,
                           still_alive: torch.Tensor,
                           available_actions_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        with torch.no_grad():
            logits, _ = self.forward(states, head_locs, still_alive)
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
