from kaggle_environments import make as kaggle_make
from kaggle_environments.envs.hungry_geese.hungry_geese import Action, Configuration, row_col
from typing import *
import torch

from ..config import N_PLAYERS
from .goose_env import ObsType
from ..utils import STATE_TYPE, torch_terminal_value_func

ACTIONS_TUPLE = tuple(Action)


class TorchEnv:
    """
    A PyTorch vectorized version of goose_env, able to be run on GPU
    """
    def __init__(
            self,
            n_envs: int,
            obs_type: ObsType,
            config: Optional[Configuration] = None,
            n_geese: int = N_PLAYERS,
            device: torch.device = torch.device('cuda')
    ):
        if config is None:
            config = Configuration(kaggle_make('hungry_geese', debug=False).configuration)
        self.config = config
        self.n_rows = config.rows
        self.n_cols = config.columns
        self.max_len = int(config.max_length)
        self.n_food = config.min_food
        self.hunger_rate = config.hunger_rate
        self.episode_steps = config.episode_steps

        self.n_envs = n_envs
        self.obs_type = obs_type
        self.n_geese = n_geese
        self.device = device

        tensor_kwargs = dict(
            dtype=torch.int64,
            device=self.device
        )
        self.geese = torch.zeros((self.n_envs, self.n_geese, self.max_len, 2), **tensor_kwargs)
        self.geese_tensor = torch.zeros((self.n_envs, self.n_geese, self.n_rows, self.n_cols), **tensor_kwargs)
        self.head_ptrs = torch.zeros((self.n_envs, self.n_geese), **tensor_kwargs)
        self.tail_ptrs = torch.zeros_like(self.head_ptrs)
        self.last_actions = torch.zeros_like(self.head_ptrs)
        self.lengths = torch.ones_like(self.head_ptrs)
        self.rewards = torch.zeros_like(self.head_ptrs)
        self.alive = torch.ones((self.n_envs, self.n_geese), dtype=torch.bool, device=self.device)
        self.ate_last_turn = torch.zeros_like(self.alive)
        self.food_tensor = torch.zeros((self.n_envs, self.n_rows, self.n_cols), **tensor_kwargs)
        self.step_counters = torch.zeros((self.n_envs,), **tensor_kwargs)
        self.dones = torch.ones((self.n_envs,), dtype=torch.bool, device=self.device)
        self.obs = torch.zeros((self.n_envs, *obs_type.get_obs_spec(self.n_geese)[1:]),
                               dtype=torch.float32, device=self.device)

        self.env_idxs = torch.arange(self.n_envs, device=self.device)
        self.env_geese_idxs = self.env_idxs.repeat_interleave(self.n_geese)
        self.geese_idxs = torch.arange(self.n_geese, device=self.device).repeat(self.n_envs)
        self.env_food_idxs = self.env_idxs.repeat_interleave(self.n_food)
        self.food_idxs = torch.arange(self.n_food, device=self.device).repeat(self.n_envs)
        self.loc_to_row_col = torch.tensor(
            [row_col(i, self.n_cols) for i in range(self.n_rows * self.n_cols)],
            **tensor_kwargs
        ).view(self.n_rows * self.n_cols, 2)
        self.row_col_to_loc = torch.arange(
            self.n_rows * self.n_cols,
            device=self.device
        ).view(self.n_rows, self.n_cols)
        self.move_to_offset = torch.tensor(
            [list(a.to_row_col()) for a in Action],
            **tensor_kwargs
        )
        self.wrap_vals = torch.tensor([self.n_rows, self.n_cols], **tensor_kwargs)
        self.goose_body_idxs = torch.arange(self.max_len, device=self.device)
        self.obs_channel_idxs = {}
        self.geese_channel_idxs = None
        if self.obs_type == ObsType.COMBINED_GRADIENT_OBS_SMALL:
            player_channel_list = [
                'contains_head',
                'contains_body',
            ]
            for i, channel in enumerate(player_channel_list):
                self.obs_channel_idxs[channel] = torch.arange(
                    i,
                    self.n_geese * len(player_channel_list),
                    len(player_channel_list),
                    device=self.device
                )
            self.obs_channel_idxs.update({
                'contains_food': torch.tensor([-3]).to(device=self.device),
                'steps_since_starvation': torch.tensor([-2]).to(device=self.device),
                'current_step': torch.tensor([-1]).to(device=self.device),
            })
            self.geese_channel_idxs = torch.arange(
                self.n_geese * len(player_channel_list),
                device=self.device
            ).view(
                1,
                self.n_geese,
                len(player_channel_list)
            ).expand(
                self.n_envs,
                self.n_geese,
                len(player_channel_list)
            ).clone()
        elif self.obs_type == ObsType.COMBINED_GRADIENT_OBS_LARGE:
            player_channel_list = [
                'contains_head',
                'contains_tail',
                'contains_body',
            ]
            for i, channel in enumerate(player_channel_list):
                self.obs_channel_idxs[channel] = torch.arange(
                    i,
                    self.n_geese * len(player_channel_list),
                    len(player_channel_list),
                    device=self.device
                )
            self.obs_channel_idxs.update({
                'contains_food': torch.tensor([-3]).to(device=self.device),
                'steps_since_starvation': torch.tensor([-2]).to(device=self.device),
                'current_step': torch.tensor([-1]).to(device=self.device),
            })
            self.geese_channel_idxs = torch.arange(
                self.n_geese * len(player_channel_list),
                device=self.device
            ).view(
                1,
                self.n_geese,
                len(player_channel_list)
            ).expand(
                self.n_envs,
                self.n_geese,
                len(player_channel_list)
            ).clone()
        elif self.obs_type == ObsType.COMBINED_GRADIENT_OBS_FULL:
            player_channel_list = [
                'contains_head',
                'contains_tail',
                'last_head_loc',
                'contains_body',
            ]
            for i, channel in enumerate(player_channel_list):
                self.obs_channel_idxs[channel] = torch.arange(
                    i,
                    self.n_geese * len(player_channel_list),
                    len(player_channel_list),
                    device=self.device
                )
            self.obs_channel_idxs.update({
                'contains_food': torch.tensor([-3]).to(device=self.device),
                'steps_since_starvation': torch.tensor([-2]).to(device=self.device),
                'current_step': torch.tensor([-1]).to(device=self.device),
            })
            self.geese_channel_idxs = torch.arange(
                self.n_geese * len(player_channel_list),
                device=self.device
            ).view(
                1,
                self.n_geese,
                len(player_channel_list)
            ).expand(
                self.n_envs,
                self.n_geese,
                len(player_channel_list)
            ).clone()
        else:
            raise NotImplementedError(f'Unsupported obs_type: {self.obs_type}')

        self.reset()

    def reset(self, get_reward_and_dead: bool = False):
        self.geese[self.dones] = 0
        self.geese_tensor[self.dones] = 0
        self.head_ptrs[self.dones] = 0
        self.tail_ptrs[self.dones] = 0
        self.last_actions[self.dones] = 0
        self.lengths[self.dones] = 1
        self.rewards[self.dones] = 0
        self.alive[self.dones] = True
        self.ate_last_turn[self.dones] = False
        self.food_tensor[self.dones] = 0
        self.step_counters[self.dones] = 0
        self.obs[self.dones] = 0.

        head_locs = torch.multinomial(
            torch.ones((self.dones.sum(), self.n_rows * self.n_cols), device=self.device),
            self.n_geese
        )
        done_env_idxs = self.env_idxs[self.dones].repeat_interleave(self.n_geese)
        done_geese_env_idxs = self.geese_idxs.view(self.n_envs, self.n_geese)[self.dones].view(-1)
        done_geese_idxs = self.loc_to_row_col[head_locs]
        self.geese[self.dones, :, 0] = done_geese_idxs
        self.geese_tensor[
            done_env_idxs,
            done_geese_env_idxs,
            done_geese_idxs[:, :, 0].view(-1),
            done_geese_idxs[:, :, 1].view(-1)
        ] = 1

        food_weights = 1. - (self.all_geese_tensor + self.food_tensor).view(self.n_envs, -1)[self.dones]
        food_locs = torch.multinomial(
            food_weights,
            self.n_food
        )
        done_env_idxs = self.env_idxs[self.dones].repeat_interleave(self.n_food)
        done_food_idxs = self.loc_to_row_col[food_locs]
        self.food_tensor[done_env_idxs, done_food_idxs[:, :, 0].view(-1), done_food_idxs[:, :, 1].view(-1)] = 1

        self._initialize_obs(self.dones)
        self.dones[:] = False

        if get_reward_and_dead:
            agent_rankings = torch_terminal_value_func(self.rewards)
            agents_not_done = self.alive & ~self.dones.unsqueeze(-1)
            returned_rewards = torch.where(
                agents_not_done,
                torch.zeros_like(agent_rankings),
                agent_rankings
            )
            return self.obs, returned_rewards, ~agents_not_done
        else:
            return self.obs

    def force_reset(self, *args, **kwargs):
        self.dones[:] = True
        return self.reset(*args, **kwargs)

    @property
    def heads(self) -> torch.Tensor:
        return self.geese[
            self.env_geese_idxs,
            self.geese_idxs,
            self.head_ptrs.view(-1)
        ].view(self.n_envs, self.n_geese, 2)

    @property
    def head_locs(self) -> torch.Tensor:
        heads = self.heads.view(-1, 2)
        return self.row_col_to_loc[
            heads[:, 0],
            heads[:, 1]
        ].view(self.n_envs, self.n_geese)

    @property
    def tails(self) -> torch.Tensor:
        return self.geese[
            self.env_geese_idxs,
            self.geese_idxs,
            self.tail_ptrs.view(-1)
        ].view(self.n_envs, self.n_geese, 2)

    @property
    def all_geese_tensor(self) -> torch.Tensor:
        return self.geese_tensor.sum(dim=1)

    def get_available_action_masks(self, dtype: torch.dtype = torch.bool) -> torch.Tensor:
        action_masks = torch.ones((self.n_envs, self.n_geese, 4), dtype=dtype, device=self.device)
        action_masks.scatter_(
            -1,
            self.last_actions.unsqueeze(-1),
            torch.zeros((self.n_envs, self.n_geese, 1), dtype=dtype, device=self.device)
        )
        action_masks[self.step_counters == 0] = True
        return action_masks[:, :, [2, 3, 0, 1]]

    def get_heads_tensor(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if dtype is None:
            dtype = self.geese_tensor.dtype
        heads_tensor = torch.zeros(self.geese_tensor.shape, dtype=dtype, device=self.device)
        heads = self.heads[self.alive]
        heads_tensor[
            self.env_geese_idxs[self.alive.view(-1)],
            self.geese_idxs[self.alive.view(-1)],
            heads[:, 0],
            heads[:, 1]
        ] = 1
        return heads_tensor

    def get_tails_tensor(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if dtype is None:
            dtype = self.geese_tensor.dtype
        tails_tensor = torch.zeros(self.geese_tensor.shape, dtype=dtype, device=self.device)
        tails = self.tails[self.alive]
        tails_tensor[
            self.env_geese_idxs[self.alive.view(-1)],
            self.geese_idxs[self.alive.view(-1)],
            tails[:, 0],
            tails[:, 1]
        ] = 1
        return tails_tensor

    def _wrap(self, position_tensor: torch.Tensor) -> torch.Tensor:
        view_shape = [1] * (position_tensor.ndim - 1)
        return position_tensor % self.wrap_vals.view(*view_shape, 2)

    def _kill_geese(self, kill_goose_mask: torch.Tensor) -> NoReturn:
        self.geese_tensor[kill_goose_mask] = 0
        self.alive[kill_goose_mask] = False
        self.ate_last_turn[kill_goose_mask] = False

    def _move_geese(self, actions: torch.Tensor, update_mask: torch.Tensor) -> NoReturn:
        update_geese = self.alive & update_mask.unsqueeze(dim=-1)
        # Get new head positions
        offsets = self.move_to_offset[actions]
        new_heads = self._wrap(self.heads + offsets)
        # Check for illegal actions
        illegal_actions = (((self.last_actions - actions).abs() == 2) &
                           update_geese &
                           (self.step_counters.unsqueeze(-1) > 1))
        # Update last action
        self.last_actions[update_mask] = torch.where(
            self.alive[update_mask],
            actions[update_mask],
            torch.zeros_like(self.last_actions[update_mask])
        )
        # Kill geese that took illegal actions
        self._kill_geese(illegal_actions)
        update_geese = self.alive & update_mask.unsqueeze(dim=-1)

        # Update self.head_ptrs
        self.head_ptrs[update_geese] = (self.head_ptrs[update_geese] + 1) % self.max_len
        # Update self.geese
        updated_new_heads = new_heads[update_geese].view(-1, 2)
        updated_env_geese_idxs = self.env_geese_idxs[update_geese.view(-1)]
        updated_geese_idxs = self.geese_idxs[update_geese.view(-1)]
        self.geese[
            updated_env_geese_idxs,
            updated_geese_idxs,
            self.head_ptrs.view(-1)[update_geese.view(-1)]
        ] = updated_new_heads
        # Update self.geese_tensor by adding new heads
        self.geese_tensor[
            updated_env_geese_idxs,
            updated_geese_idxs,
            updated_new_heads[:, 0],
            updated_new_heads[:, 1]
        ] += 1

        # Check if any geese eat
        goose_eat = (self.food_tensor[
            self.env_geese_idxs,
            new_heads.view(-1, 2)[:, 0],
            new_heads.view(-1, 2)[:, 1]
        ] > 0) & update_geese.view(-1)
        self.ate_last_turn[update_geese] = goose_eat[update_geese.view(-1)]
        # Remove food where geese have eaten
        self.food_tensor[
            self.env_geese_idxs[goose_eat],
            new_heads.view(-1, 2)[goose_eat, 0],
            new_heads.view(-1, 2)[goose_eat, 1]
        ] = 0
        # Update self.geese_tensor by removing tails
        updated_tails = self.tails[update_geese].view(-1, 2)
        self.geese_tensor[
            updated_env_geese_idxs,
            updated_geese_idxs,
            updated_tails[:, 0],
            updated_tails[:, 1]
        ] -= (1 - goose_eat.to(torch.int64))[update_geese.view(-1)]
        # Update self.tail_ptrs
        grow_goose = goose_eat & (self.lengths.view(-1) < self.max_len)
        self.tail_ptrs[:] = torch.where(
            grow_goose,
            self.tail_ptrs.view(-1),
            (self.tail_ptrs.view(-1) + 1) % self.max_len
        ).view(self.n_envs, self.n_geese)
        # Update self.lengths
        self.lengths[grow_goose.view(self.n_envs, self.n_geese)] += 1

        # Check if any geese collide with themselves
        self_collision = self.geese_tensor[
            self.env_geese_idxs,
            self.geese_idxs,
            self.heads.view(-1, 2)[:, 0],
            self.heads.view(-1, 2)[:, 1]
        ] > 1
        self_collision = self_collision.view(self.n_envs, self.n_geese) & update_geese
        # Kill geese that collided with themselves
        self._kill_geese(self_collision)
        update_geese = self.alive & update_mask.unsqueeze(dim=-1)

        # Shrink geese every self.hunger_rate steps
        shrink_goose = ((self.step_counters.repeat_interleave(self.n_geese) % self.hunger_rate == 0) &
                        update_geese.view(-1))
        shrink_tails = self.tails.view(-1, 2)[shrink_goose]
        self.geese_tensor[
            self.env_geese_idxs[shrink_goose],
            self.geese_idxs[shrink_goose],
            shrink_tails[:, 0],
            shrink_tails[:, 1]
        ] -= 1
        self.tail_ptrs[:] = torch.where(
            shrink_goose,
            (self.tail_ptrs.view(-1) + 1) % self.max_len,
            self.tail_ptrs.view(-1)
        ).view(self.n_envs, self.n_geese)
        self.lengths[shrink_goose.view(self.n_envs, self.n_geese)] -= 1
        self._kill_geese((self.lengths == 0) & update_geese)
        update_geese = self.alive & update_mask.unsqueeze(dim=-1)

        # Check for collisions between geese
        collision = self.all_geese_tensor[
            self.env_geese_idxs,
            self.heads.view(-1, 2)[:, 0],
            self.heads.view(-1, 2)[:, 1]
        ] > 1
        collision = collision.view(self.n_envs, self.n_geese) & update_geese
        self._kill_geese(collision)

    def _replenish_food(self, update_mask: torch.Tensor) -> NoReturn:
        all_geese_cached = self.all_geese_tensor
        food_weights = 1. - (all_geese_cached + self.food_tensor).view(self.n_envs, -1)[update_mask]
        # Ensure that if there are no available food locations, an error is not thrown
        food_weights[food_weights.sum(dim=-1) == 0] = 1.
        food_locs = torch.multinomial(
            food_weights,
            self.n_food
        ).view(-1)
        # Make sure to only put food in environments according to the number of food missing from that env
        n_food_needed = self.n_food - self.food_tensor.view(self.n_envs, -1)[update_mask].sum(dim=-1, keepdims=True)
        spots_available = self.n_rows * self.n_cols - (all_geese_cached +
                                                       self.food_tensor
                                                       ).view(self.n_envs, -1)[update_mask].sum(dim=-1, keepdims=True)
        new_food_needed = torch.arange(self.n_food, device=self.device).unsqueeze(dim=0).expand(update_mask.sum(), -1)
        new_food_needed = (new_food_needed < torch.minimum(n_food_needed, spots_available)).view(-1)
        new_food_env_idxs = self.env_idxs[update_mask].repeat_interleave(self.n_food)[new_food_needed]
        new_food_idxs = self.loc_to_row_col[food_locs[new_food_needed]]
        self.food_tensor[new_food_env_idxs, new_food_idxs[:, 0], new_food_idxs[:, 1]] = 1

    def _check_if_done(self) -> NoReturn:
        self.dones[:] = (self.alive.sum(dim=-1) <= 1) | (self.step_counters >= self.episode_steps - 1)

    def _initialize_obs(self, new_envs_mask: torch.Tensor) -> NoReturn:
        if self.obs_type == ObsType.COMBINED_GRADIENT_OBS_SMALL:
            updated_env_geese_idxs = self.env_geese_idxs[new_envs_mask.repeat_interleave(self.n_geese)]
            updated_obs_channel_idxs = {
                key: val.unsqueeze(0).expand(
                    self.n_envs,
                    -1
                )[new_envs_mask].view(-1) for key, val in self.obs_channel_idxs.items()
            }

            self.obs[new_envs_mask] = 0.
            self.obs[
                updated_env_geese_idxs,
                updated_obs_channel_idxs['contains_head']
            ] = self.get_heads_tensor(dtype=torch.float32)[new_envs_mask].view(-1, self.n_rows, self.n_cols)
            updated_heads = self.heads[new_envs_mask].view(-1, 2)
            self.obs[
                updated_env_geese_idxs,
                updated_obs_channel_idxs['contains_body'],
                updated_heads[:, 0],
                updated_heads[:, 1]
            ] = self.lengths.to(dtype=torch.float32)[new_envs_mask].view(-1) / self.max_len
            self.obs[
                self.env_idxs[new_envs_mask],
                updated_obs_channel_idxs['contains_food']
            ] = self.food_tensor[new_envs_mask].to(dtype=torch.float32)
        elif self.obs_type == ObsType.COMBINED_GRADIENT_OBS_LARGE:
            updated_env_geese_idxs = self.env_geese_idxs[new_envs_mask.repeat_interleave(self.n_geese)]
            updated_obs_channel_idxs = {
                key: val.unsqueeze(0).expand(
                    self.n_envs,
                    -1
                )[new_envs_mask].view(-1) for key, val in self.obs_channel_idxs.items()
            }

            self.obs[new_envs_mask] = 0.
            self.obs[
                updated_env_geese_idxs,
                updated_obs_channel_idxs['contains_head']
            ] = self.get_heads_tensor(dtype=torch.float32)[new_envs_mask].view(-1, self.n_rows, self.n_cols)
            self.obs[
                updated_env_geese_idxs,
                updated_obs_channel_idxs['contains_tail']
            ] = self.get_tails_tensor(dtype=torch.float32)[new_envs_mask].view(-1, self.n_rows, self.n_cols)
            updated_heads = self.heads[new_envs_mask].view(-1, 2)
            self.obs[
                updated_env_geese_idxs,
                updated_obs_channel_idxs['contains_body'],
                updated_heads[:, 0],
                updated_heads[:, 1]
            ] = self.lengths.to(dtype=torch.float32)[new_envs_mask].view(-1) / self.max_len
            self.obs[
                self.env_idxs[new_envs_mask],
                updated_obs_channel_idxs['contains_food']
            ] = self.food_tensor[new_envs_mask].to(dtype=torch.float32)
        elif self.obs_type == ObsType.COMBINED_GRADIENT_OBS_FULL:
            updated_env_geese_idxs = self.env_geese_idxs[new_envs_mask.repeat_interleave(self.n_geese)]
            updated_obs_channel_idxs = {
                key: val.unsqueeze(0).expand(
                    self.n_envs,
                    -1
                )[new_envs_mask].view(-1) for key, val in self.obs_channel_idxs.items()
            }

            self.obs[new_envs_mask] = 0.
            self.obs[
                updated_env_geese_idxs,
                updated_obs_channel_idxs['contains_head']
            ] = self.get_heads_tensor(dtype=torch.float32)[new_envs_mask].view(-1, self.n_rows, self.n_cols)
            self.obs[
                updated_env_geese_idxs,
                updated_obs_channel_idxs['contains_tail']
            ] = self.get_tails_tensor(dtype=torch.float32)[new_envs_mask].view(-1, self.n_rows, self.n_cols)
            updated_heads = self.heads[new_envs_mask].view(-1, 2)
            self.obs[
                updated_env_geese_idxs,
                updated_obs_channel_idxs['contains_body'],
                updated_heads[:, 0],
                updated_heads[:, 1]
            ] = self.lengths.to(dtype=torch.float32)[new_envs_mask].view(-1) / self.max_len
            self.obs[
                self.env_idxs[new_envs_mask],
                updated_obs_channel_idxs['contains_food']
            ] = self.food_tensor[new_envs_mask].to(dtype=torch.float32)
        else:
            raise NotImplementedError(f'Unsupported obs_type: {self.obs_type}')

    def _update_obs(self, update_mask: torch.Tensor) -> NoReturn:
        if self.obs_type == ObsType.COMBINED_GRADIENT_OBS_SMALL:
            updated_env_geese_idxs = self.env_geese_idxs[update_mask.repeat_interleave(self.n_geese)]
            updated_obs_channel_idxs = {
                key: val.unsqueeze(0).expand(
                    self.n_envs,
                    -1
                )[update_mask].view(-1) for key, val in self.obs_channel_idxs.items()
            }
            n_goose_channels = self.geese_channel_idxs.shape[-1]

            self.obs[
                updated_env_geese_idxs,
                updated_obs_channel_idxs['contains_head']
            ] = self.get_heads_tensor(dtype=torch.float32)[update_mask].view(-1, self.n_rows, self.n_cols)
            body_decrement = torch.where(
                self.step_counters % self.hunger_rate == 0,
                2. / self.max_len,
                1. / self.max_len
            )[update_mask].unsqueeze(-1).expand(-1, self.n_geese)
            body_decrement = torch.where(
                self.ate_last_turn[update_mask],
                body_decrement - 1. / self.max_len,
                body_decrement
            ).view(-1, 1, 1)
            self.obs[
                updated_env_geese_idxs,
                updated_obs_channel_idxs['contains_body']
            ] -= body_decrement
            updated_heads = self.heads[update_mask].view(-1, 2)
            self.obs[
                updated_env_geese_idxs,
                updated_obs_channel_idxs['contains_body'],
                updated_heads[:, 0],
                updated_heads[:, 1]
            ] = self.lengths.to(dtype=torch.float32)[update_mask].view(-1) / self.max_len
            self.obs[
                updated_env_geese_idxs,
                updated_obs_channel_idxs['contains_body']
            ] *= self.geese_tensor[update_mask].view(-1, self.n_rows, self.n_cols).to(dtype=torch.float32)
            self.obs.clamp_(min=0.)
            self.obs[
                self.env_geese_idxs[(update_mask.unsqueeze(-1) & ~self.alive).view(-1)].repeat_interleave(
                    n_goose_channels),
                self.geese_channel_idxs[update_mask.unsqueeze(-1) & ~self.alive].view(-1)
            ] = 0.
            self.obs[
                self.env_idxs[update_mask],
                updated_obs_channel_idxs['contains_food']
            ] = self.food_tensor[update_mask].to(dtype=torch.float32)
            self.obs[
                self.env_idxs[update_mask],
                updated_obs_channel_idxs['steps_since_starvation']
            ] = (self.step_counters[update_mask].view(-1, 1, 1).to(torch.float32) % self.hunger_rate) / self.hunger_rate
            self.obs[
                self.env_idxs[update_mask],
                updated_obs_channel_idxs['current_step']
            ] = self.step_counters[update_mask].view(-1, 1, 1).to(torch.float32) / self.episode_steps
        elif self.obs_type == ObsType.COMBINED_GRADIENT_OBS_LARGE:
            updated_env_geese_idxs = self.env_geese_idxs[update_mask.repeat_interleave(self.n_geese)]
            updated_obs_channel_idxs = {
                key: val.unsqueeze(0).expand(
                    self.n_envs,
                    -1
                )[update_mask].view(-1) for key, val in self.obs_channel_idxs.items()
            }
            n_goose_channels = self.geese_channel_idxs.shape[-1]

            self.obs[
                updated_env_geese_idxs,
                updated_obs_channel_idxs['contains_head']
            ] = self.get_heads_tensor(dtype=torch.float32)[update_mask].view(-1, self.n_rows, self.n_cols)
            self.obs[
                updated_env_geese_idxs,
                updated_obs_channel_idxs['contains_tail']
            ] = self.get_tails_tensor(dtype=torch.float32)[update_mask].view(-1, self.n_rows, self.n_cols)
            body_decrement = torch.where(
                self.step_counters % self.hunger_rate == 0,
                2. / self.max_len,
                1. / self.max_len
            )[update_mask].unsqueeze(-1).expand(-1, self.n_geese)
            body_decrement = torch.where(
                self.ate_last_turn[update_mask],
                body_decrement - 1. / self.max_len,
                body_decrement
            ).view(-1, 1, 1)
            self.obs[
                updated_env_geese_idxs,
                updated_obs_channel_idxs['contains_body']
            ] -= body_decrement
            updated_heads = self.heads[update_mask].view(-1, 2)
            self.obs[
                updated_env_geese_idxs,
                updated_obs_channel_idxs['contains_body'],
                updated_heads[:, 0],
                updated_heads[:, 1]
            ] = self.lengths.to(dtype=torch.float32)[update_mask].view(-1) / self.max_len
            self.obs[
                updated_env_geese_idxs,
                updated_obs_channel_idxs['contains_body']
            ] *= self.geese_tensor[update_mask].view(-1, self.n_rows, self.n_cols).to(dtype=torch.float32)
            self.obs.clamp_(min=0.)
            self.obs[
                self.env_geese_idxs[
                    (update_mask.unsqueeze(-1) & ~self.alive).view(-1)
                ].repeat_interleave(n_goose_channels),
                self.geese_channel_idxs[update_mask.unsqueeze(-1) & ~self.alive].view(-1)
            ] = 0.
            self.obs[
                self.env_idxs[update_mask],
                updated_obs_channel_idxs['contains_food']
            ] = self.food_tensor[update_mask].to(dtype=torch.float32)
            self.obs[
                self.env_idxs[update_mask],
                updated_obs_channel_idxs['steps_since_starvation']
            ] = (self.step_counters[update_mask].view(-1, 1, 1).to(torch.float32) % self.hunger_rate) / self.hunger_rate
            self.obs[
                self.env_idxs[update_mask],
                updated_obs_channel_idxs['current_step']
            ] = self.step_counters[update_mask].view(-1, 1, 1).to(torch.float32) / self.episode_steps
        elif self.obs_type == ObsType.COMBINED_GRADIENT_OBS_FULL:
            updated_env_geese_idxs = self.env_geese_idxs[update_mask.repeat_interleave(self.n_geese)]
            updated_obs_channel_idxs = {
                key: val.unsqueeze(0).expand(
                    self.n_envs,
                    -1
                )[update_mask].view(-1) for key, val in self.obs_channel_idxs.items()
            }
            n_goose_channels = self.geese_channel_idxs.shape[-1]

            self.obs[
                updated_env_geese_idxs,
                updated_obs_channel_idxs['last_head_loc']
            ] = self.obs[
                updated_env_geese_idxs,
                updated_obs_channel_idxs['contains_head']
            ].clone().view(-1, self.n_rows, self.n_cols)
            self.obs[
                updated_env_geese_idxs,
                updated_obs_channel_idxs['contains_head']
            ] = self.get_heads_tensor(dtype=torch.float32)[update_mask].view(-1, self.n_rows, self.n_cols)
            self.obs[
                updated_env_geese_idxs,
                updated_obs_channel_idxs['contains_tail']
            ] = self.get_tails_tensor(dtype=torch.float32)[update_mask].view(-1, self.n_rows, self.n_cols)
            body_decrement = torch.where(
                self.step_counters % self.hunger_rate == 0,
                2. / self.max_len,
                1. / self.max_len
            )[update_mask].unsqueeze(-1).expand(-1, self.n_geese)
            body_decrement = torch.where(
                self.ate_last_turn[update_mask],
                body_decrement - 1. / self.max_len,
                body_decrement
            ).view(-1, 1, 1)
            self.obs[
                updated_env_geese_idxs,
                updated_obs_channel_idxs['contains_body']
            ] -= body_decrement
            updated_heads = self.heads[update_mask].view(-1, 2)
            self.obs[
                updated_env_geese_idxs,
                updated_obs_channel_idxs['contains_body'],
                updated_heads[:, 0],
                updated_heads[:, 1]
            ] = self.lengths.to(dtype=torch.float32)[update_mask].view(-1) / self.max_len
            self.obs[
                updated_env_geese_idxs,
                updated_obs_channel_idxs['contains_body']
            ] *= self.geese_tensor[update_mask].view(-1, self.n_rows, self.n_cols).to(dtype=torch.float32)
            self.obs.clamp_(min=0.)
            self.obs[
                self.env_geese_idxs[
                    (update_mask.unsqueeze(-1) & ~self.alive).view(-1)
                ].repeat_interleave(n_goose_channels),
                self.geese_channel_idxs[update_mask.unsqueeze(-1) & ~self.alive].view(-1)
            ] = 0.
            self.obs[
                self.env_idxs[update_mask],
                updated_obs_channel_idxs['contains_food']
            ] = self.food_tensor[update_mask].to(dtype=torch.float32)
            self.obs[
                self.env_idxs[update_mask],
                updated_obs_channel_idxs['steps_since_starvation']
            ] = (self.step_counters[update_mask].view(-1, 1, 1).to(torch.float32) % self.hunger_rate) / self.hunger_rate
            self.obs[
                self.env_idxs[update_mask],
                updated_obs_channel_idxs['current_step']
            ] = self.step_counters[update_mask].view(-1, 1, 1).to(torch.float32) / self.episode_steps
        else:
            raise NotImplementedError(f'Unsupported obs_type: {self.obs_type}')

    def _update_rewards(self, update_mask: torch.Tensor) -> NoReturn:
        update_geese = self.alive & update_mask.unsqueeze(dim=-1)
        self.rewards[:] = torch.where(
            update_geese,
            self.step_counters.unsqueeze(dim=-1) * (self.max_len + 1) + self.lengths,
            self.rewards
        )

    def step(self, actions: torch.Tensor, update_mask: Optional[torch.Tensor] = None,
             get_reward_and_dead: bool = False):
        if update_mask is None:
            update_mask = torch.ones_like(self.dones, dtype=torch.bool)
        if update_mask.shape != self.dones.shape:
            raise RuntimeError(f'update_mask should be of shape {self.dones.shape}, was {update_mask.shape}')
        if (self.dones & update_mask).any():
            raise RuntimeError(f'{self.dones.sum()} environments are finished - call env.reset() before continuing')
        if actions.shape != (self.n_envs, self.n_geese):
            raise ValueError(f'actions.shape was {actions.shape}, but should be {(self.n_envs, self.n_geese)}')
        if actions.dtype != torch.int64:
            raise TypeError(f'actions.dtype was {actions.dtype}, but should be {torch.int64}')

        self.step_counters[update_mask] += 1
        self._move_geese(actions, update_mask)
        self._replenish_food(update_mask)
        self._check_if_done()
        self._update_obs(update_mask)
        self._update_rewards(update_mask)

        if get_reward_and_dead:
            agent_rankings = torch_terminal_value_func(self.rewards)
            agents_not_done = self.alive & ~self.dones.unsqueeze(-1)
            returned_rewards = torch.where(
                agents_not_done,
                torch.zeros_like(agent_rankings),
                agent_rankings
            )
            return self.obs, returned_rewards, ~agents_not_done
        else:
            return self.obs

    def generate_obs_dicts(self, selected_envs_mask: Optional[torch.Tensor] = None) -> List[STATE_TYPE]:
        return generate_obs_dicts(
            n_envs=self.n_envs,
            n_geese=self.n_geese,
            dones=self.dones.cpu(),
            alive=self.alive.cpu(),
            last_actions=self.last_actions.cpu(),
            rewards=self.rewards.cpu(),
            step_counters=self.step_counters.cpu(),
            geese_positions=self.geese.cpu(),
            tail_ptrs=self.tail_ptrs.cpu(),
            head_ptrs=self.head_ptrs.cpu(),
            row_col_to_loc=self.row_col_to_loc.cpu(),
            food_tensor=self.food_tensor.cpu(),
            selected_envs_mask=selected_envs_mask
        )

    def render_env(self, env_idx: int, include_info: bool = True) -> str:
        out_mat = torch.zeros((self.n_rows, self.n_cols), device=self.device, dtype=torch.int64)
        for i in range(self.n_geese):
            out_mat = torch.where(
                self.geese_tensor[env_idx, i] > 0,
                i + 1,
                out_mat
            )
            if self.alive[env_idx, i]:
                head = self.heads[env_idx, i]
                out_mat[head[0], head[1]] += self.n_geese
        out_mat = torch.where(
            self.food_tensor[env_idx] > 0,
            self.n_geese * 2 + 1,
            out_mat
        )
        out_mat = out_mat.cpu()

        display_dict = {
            0: '_',
            self.n_geese * 2 + 1: '*'
        }
        display_dict.update({i + 1: chr(ord('a') + i) for i in range(self.n_geese)})
        display_dict.update({i + self.n_geese + 1: chr(ord('A') + i) for i in range(self.n_geese)})

        out_str = ''
        if include_info:
            out_str += (f'Step: {self.step_counters[env_idx].cpu().item()}\n'
                        f'Lengths: {[i.item() for i in (self.lengths[env_idx] * self.alive[env_idx]).cpu()]}\n'
                        f'Rewards: {[i.item() for i in self.rewards[env_idx].cpu()]}\n'
                        f'Last actions: {[ACTIONS_TUPLE[i.item()].name for i in self.last_actions[env_idx].cpu()]}\n\n')
        for row in range(self.n_rows):
            out_str += ' '.join(display_dict.get(cell.item(), '?') for cell in out_mat[row]) + '\n'
        return out_str

    def copy_data_to(self, target: 'TorchEnv') -> NoReturn:
        target.geese[:] = self.geese[:]
        target.geese_tensor[:] = self.geese_tensor[:]
        target.head_ptrs[:] = self.head_ptrs[:]
        target.tail_ptrs[:] = self.tail_ptrs[:]
        target.last_actions[:] = self.last_actions[:]
        target.lengths[:] = self.lengths[:]
        target.rewards[:] = self.rewards[:]
        target.alive[:] = self.alive[:]
        target.ate_last_turn[:] = self.ate_last_turn[:]
        target.food_tensor[:] = self.food_tensor[:]
        target.step_counters[:] = self.step_counters[:]
        target.dones[:] = self.dones[:]
        target.obs[:] = self.obs[:]

    def copy_data_from(self, source: 'TorchEnv') -> NoReturn:
        source.copy_data_to(self)

    def share_memory_(self) -> NoReturn:
        self.geese.share_memory_()
        self.geese_tensor.share_memory_()
        self.head_ptrs.share_memory_()
        self.tail_ptrs.share_memory_()
        self.last_actions.share_memory_()
        self.lengths.share_memory_()
        self.rewards.share_memory_()
        self.alive.share_memory_()
        self.ate_last_turn.share_memory_()
        self.food_tensor.share_memory_()
        self.step_counters.share_memory_()
        self.dones.share_memory_()
        self.obs.share_memory_()


def _get_geese_list(
        env_idx: int,
        n_geese: int,
        alive: torch.Tensor,
        geese_positions: torch.Tensor,
        tail_ptrs: torch.Tensor,
        head_ptrs: torch.Tensor,
        row_col_to_loc: torch.Tensor
) -> List[List[int]]:
    geese = []
    for goose_idx in range(n_geese):
        if alive[env_idx, goose_idx]:
            goose_vec_full = geese_positions[env_idx, goose_idx].roll(
                shifts=-tail_ptrs[env_idx, goose_idx].item(),
                dims=0
            )
            goose_vec_sliced = goose_vec_full[:(head_ptrs[env_idx, goose_idx] - tail_ptrs[env_idx, goose_idx] + 1)]
            goose_loc_list = row_col_to_loc[goose_vec_sliced[:, 0], goose_vec_sliced[:, 1]].tolist()
        else:
            goose_loc_list = []
        geese.append(goose_loc_list[::-1])
    return geese


def _get_food_list(env_idx: int, food_tensor: torch.Tensor) -> List[int]:
    return [i.item() for i in torch.nonzero(food_tensor[env_idx].view(-1))]


def generate_obs_dicts(
        n_envs: int,
        n_geese: int,
        dones: torch.Tensor,
        alive: torch.Tensor,
        last_actions: torch.Tensor,
        rewards: torch.Tensor,
        step_counters: torch.Tensor,
        geese_positions: torch.Tensor,
        tail_ptrs: torch.Tensor,
        head_ptrs: torch.Tensor,
        row_col_to_loc: torch.Tensor,
        food_tensor: torch.Tensor,
        selected_envs_mask: Optional[torch.Tensor] = None
) -> List[STATE_TYPE]:
    all_dicts = []
    if selected_envs_mask is None:
        selected_envs_mask = torch.ones(dones.shape, dtype=torch.bool)
    else:
        selected_envs_mask = selected_envs_mask.cpu()
    for env_idx in torch.arange(n_envs)[selected_envs_mask]:
        state_dict_list = []
        if selected_envs_mask[env_idx]:
            if dones[env_idx]:
                statuses = ['DONE' for _ in range(n_geese)]
            else:
                statuses = ['ACTIVE' if a else 'DONE' for a in alive[env_idx]]
            for goose_idx in range(n_geese):
                dict_g = {
                    'action': ACTIONS_TUPLE[last_actions[env_idx, goose_idx]].name,
                    'reward': rewards[env_idx, goose_idx].cpu().item(),
                    'info': {},
                    'observation': {
                        'index': goose_idx
                    },
                    'status': statuses[goose_idx]
                }
                if goose_idx == 0:
                    dict_g['observation'].update({
                        'step': step_counters[env_idx].cpu().item(),
                        'geese': _get_geese_list(
                            env_idx=env_idx,
                            n_geese=n_geese,
                            alive=alive,
                            geese_positions=geese_positions,
                            tail_ptrs=tail_ptrs,
                            head_ptrs=head_ptrs,
                            row_col_to_loc=row_col_to_loc
                        ),
                        'food': _get_food_list(env_idx, food_tensor)
                    })
                state_dict_list.append(dict_g)
        all_dicts.append(state_dict_list)
    return all_dicts
