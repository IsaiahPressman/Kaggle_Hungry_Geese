from kaggle_environments.envs.hungry_geese.hungry_geese import Action, Configuration, row_col
from typing import *
import torch

from ..config import N_PLAYERS
from .goose_env import ObsType


class TorchEnv:
    """
    A PyTorch vectorized version of goose_env, able to be run on GPU
    """
    def __init__(
            self,
            config: Configuration,
            n_envs: int,
            obs_type: ObsType,
            n_geese: int = N_PLAYERS,
            device: torch.device = torch.device('cuda'),
            debug: bool = False
    ):
        self.config = config
        self.n_rows = config.rows
        self.n_cols = config.columns
        self.max_len = config.max_length
        self.n_food = config.min_food
        self.debug = debug

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
        self.last_move = -torch.ones_like(self.head_ptrs)
        self.lengths = torch.ones_like(self.head_ptrs)
        self.rewards = torch.zeros_like(self.head_ptrs)
        self.alive = torch.ones((self.n_envs, self.n_geese), dtype=torch.bool, device=self.device)
        self.food_tensor = torch.zeros((self.n_envs, self.n_rows, self.n_cols), **tensor_kwargs)
        self.step_counters = torch.zeros((self.n_envs,), **tensor_kwargs)
        self.dones = torch.ones((self.n_envs,), dtype=torch.bool, device=self.device)

        self.env_idxs = torch.arange(self.n_envs, device=self.device)
        self.env_geese_idxs = self.env_idxs.repeat_interleave(self.n_geese)
        self.geese_idxs = torch.arange(self.n_geese, device=self.device).repeat(self.n_envs)
        self.env_food_idxs = self.env_idxs.repeat_interleave(self.n_food)
        self.food_idxs = torch.arange(self.n_food, device=self.device).repeat(self.n_envs)
        self.loc_to_row_column = torch.tensor(
            [row_col(i, self.n_cols) for i in range(self.n_rows * self.n_cols)],
            **tensor_kwargs
        ).view(self.n_rows * self.n_cols, 2)
        self.move_to_offset = torch.tensor(
            [list(a.to_row_col()) for a in Action],
            **tensor_kwargs
        )
        self.wrap_vals = torch.tensor([self.n_rows, self.n_cols], **tensor_kwargs)
        self.goose_body_idxs = torch.arange(self.max_len, device=device)

    def reset(self) -> torch.Tensor:
        self.geese[self.dones] = 0
        self.geese_tensor[self.dones] = 0
        self.head_ptrs[self.dones] = 0
        self.tail_ptrs[self.dones] = 0
        self.last_move[self.dones] = 1
        self.lengths[self.dones] = 1
        self.rewards[self.dones] = 0
        self.alive[self.dones] = 1
        self.food_tensor[self.dones] = 0
        self.step_counters[self.dones] = 0

        head_locs = torch.multinomial(
            torch.ones((self.dones.sum(), self.n_rows * self.n_cols), device=self.device),
            self.n_geese
        )
        done_env_idxs = self.env_idxs[self.dones].repeat_interleave(self.n_geese)
        done_geese_env_idxs = self.geese_idxs.view(self.n_envs, self.n_geese)[self.dones].view(-1)
        done_geese_idxs = self.loc_to_row_column[head_locs]
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
        done_food_idxs = self.loc_to_row_column[food_locs]
        self.food_tensor[done_env_idxs, done_food_idxs[:, :, 0].view(-1), done_food_idxs[:, :, 1].view(-1)] = 1

        self.dones[:] = False
        # TODO: return observation
        return torch.ones(1)

    @property
    def heads(self) -> torch.Tensor:
        return self.geese[
            self.env_geese_idxs,
            self.geese_idxs,
            self.head_ptrs.view(-1)
        ].view(self.n_envs, self.n_geese, 2)

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

    def _wrap(self, position_tensor: torch.Tensor) -> torch.Tensor:
        view_shape = [1] * (position_tensor.ndim - 1)
        return position_tensor % self.wrap_vals.view(*view_shape, 2)

    def _kill_geese(self, kill_goose_mask: torch.Tensor) -> NoReturn:
        self.geese_tensor[kill_goose_mask] = 0
        self.alive[kill_goose_mask] = False

    def _move_geese(self, moves: torch.Tensor, update_mask: torch.Tensor) -> NoReturn:
        update_geese = self.alive & update_mask.unsqueeze(dim=-1)
        # Get new head positions
        offsets = self.move_to_offset[moves]
        new_heads = self._wrap(self.heads + offsets)
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
        self.tail_ptrs = torch.where(
            grow_goose,
            self.tail_ptrs.view(-1),
            (self.tail_ptrs.view(-1) + 1) % self.max_len
        ).view(self.n_envs, self.n_geese)
        # Update self.lengths
        self.lengths[grow_goose.view(self.n_envs, self.n_geese)] += 1
        # Update last move
        self.last_move = torch.where(
            update_geese,
            moves,
            torch.zeros_like(self.last_move)
        )

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

        # Shrink geese every 40 steps
        shrink_goose = (self.step_counters.repeat_interleave(self.n_geese) % 40 == 0) & update_geese.view(-1)
        shrink_tails = self.tails.view(-1, 2)[shrink_goose]
        self.geese_tensor[
            self.env_geese_idxs[shrink_goose],
            self.geese_idxs[shrink_goose],
            shrink_tails[:, 0],
            shrink_tails[:, 1]
        ] -= 1
        self.tail_ptrs = torch.where(
            shrink_goose,
            (self.tail_ptrs.view(-1) + 1) % self.max_len,
            self.tail_ptrs.view(-1)
        ).view(self.n_envs, self.n_geese)
        self.lengths[shrink_goose.view(self.n_envs, self.n_geese)] -= 1
        self._kill_geese((self.lengths == 0) & update_geese)

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
        food_locs = torch.multinomial(
            food_weights,
            self.n_food
        ).view(-1)
        n_food_needed = self.n_food - self.food_tensor.view(self.n_envs, -1)[update_mask].sum(dim=-1, keepdims=True)
        spots_available = self.n_rows * self.n_cols - (all_geese_cached +
                                                       self.food_tensor
                                                       ).view(self.n_envs, -1)[update_mask].sum(dim=-1, keepdims=True)
        new_food_needed = torch.arange(self.n_food, device=self.device).unsqueeze(dim=0).expand(update_mask.sum(), -1)
        new_food_needed = (new_food_needed < torch.minimum(n_food_needed, spots_available)).view(-1)
        new_food_env_idxs = self.env_idxs[update_mask].repeat_interleave(self.n_food)[new_food_needed]
        new_food_idxs = self.loc_to_row_column[food_locs[new_food_needed]]
        self.food_tensor[new_food_env_idxs, new_food_idxs[:, 0], new_food_idxs[:, 1]] = 1

    def _check_if_done(self) -> NoReturn:
        self.dones = self.alive.sum(dim=-1) <= 1

    def _update_obs(self, update_mask: torch.Tensor) -> NoReturn:
        if self.obs_type == ObsType.COMBINED_GRADIENT_OBS_SMALL:
            pass
        elif self.obs_type == ObsType.COMBINED_GRADIENT_OBS_LARGE:
            pass
        else:
            raise ValueError(f'Unsupported obs_type: {self.obs_type}')

    def _update_rewards(self, update_mask: torch.Tensor) -> NoReturn:
        update_geese = self.alive & update_mask.unsqueeze(dim=-1)
        self.rewards = torch.where(
            update_geese,
            self.step_counters.unsqueeze(dim=-1) * (self.max_len + 1) + self.lengths,
            self.rewards
        )

    def step(self, actions: torch.Tensor, update_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
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

        self.step_counters += 1
        self._move_geese(actions, update_mask)
        self._replenish_food(update_mask)
        self._check_if_done()
        self._update_obs(update_mask)
        self._update_rewards(update_mask)

        # TODO: return observation
        return torch.ones(1)

    def generate_obs_dicts(self) -> List[List[Dict]]:
        raise NotImplementedError()

    def render_env(self, env_idx: int, include_info: bool = False) -> str:
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
                        f'Rewards: {[i.item() for i in self.rewards[env_idx].cpu()]}\n\n')
        for row in range(self.n_rows):
            out_str += ' '.join(display_dict.get(cell.item(), '?') for cell in out_mat[row]) + '\n'
        return out_str

    def copy_data_to(self, target: 'TorchEnv') -> NoReturn:
        target.geese[:] = self.geese[:]
        target.geese_tensor[:] = self.geese_tensor[:]
        target.head_ptrs[:] = self.head_ptrs[:]
        target.tail_ptrs[:] = self.tail_ptrs[:]
        target.last_move[:] = self.last_move[:]
        target.lengths[:] = self.lengths[:]
        target.rewards[:] = self.rewards[:]
        target.alive[:] = self.alive[:]
        target.food_tensor[:] = self.food_tensor[:]
        target.step_counters[:] = self.step_counters[:]
        target.dones[:] = self.dones[:]

    def copy_data_from(self, source: 'TorchEnv') -> NoReturn:
        source.copy_data_to(self)


# DEPRECATED:
"""
@torch.jit.script
def _did_goose_self_collide(
        goose_locs: torch.Tensor,
        head_ptr: int,
        tail_ptr: int,
        goose_body_idxs: torch.Tensor
) -> torch.Tensor:
    if head_ptr >= tail_ptr:
        goose_body_mask = torch.logical_and(
            goose_body_idxs < head_ptr,
            goose_body_idxs >= tail_ptr
        )
    else:
        goose_body_mask = torch.logical_or(
            goose_body_idxs < head_ptr,
            goose_body_idxs >= tail_ptr
        )
    head = goose_locs[head_ptr]
    return torch.any(head == goose_locs[goose_body_mask])


@torch.jit.script
def _check_for_self_collision(
        geese_locs: torch.Tensor,
        head_ptrs: torch.Tensor,
        tail_ptrs: torch.Tensor,
        goose_body_idxs: torch.Tensor
) -> torch.Tensor:
    n_envs = geese_locs.shape[0]
    n_geese = geese_locs.shape[1]
    # TODO remove:
    return torch.zeros((n_envs, n_geese), dtype=torch.bool)
    max_len = geese_locs.shape[2]
    futures = [torch.jit.fork(
        _did_goose_self_collide,
        geese_locs.view(n_envs * n_geese, max_len)[i],
        head_ptrs.view(-1)[i].item(),
        tail_ptrs.view(-1)[i].item(),
        goose_body_idxs
    ) for i in range(n_envs * n_geese)]
    results = [torch.jit.wait(fut) for fut in futures]
    return torch.stack(results).view(n_envs, n_geese)


@torch.jit.script
def _kill_goose(
        geese_tensor: torch.Tensor,
        goose: torch.Tensor,
        env_idx: int,
        goose_idx: int,
        head_ptr: int,
        tail_ptr: int,
        goose_body_idxs: torch.Tensor
):
    if head_ptr >= tail_ptr:
        goose_body_mask = torch.logical_and(
            goose_body_idxs <= head_ptr,
            goose_body_idxs >= tail_ptr
        )
    else:
        goose_body_mask = torch.logical_or(
            goose_body_idxs <= head_ptr,
            goose_body_idxs >= tail_ptr
        )
    geese_tensor[env_idx, goose_idx, goose[goose_body_mask, 0], goose[goose_body_mask, 1]] = 0


@torch.jit.script
def _kill_geese(
        geese_tensor: torch.Tensor,
        geese: torch.Tensor,
        geese_env_idxs: torch.Tensor,
        geese_idxs: torch.Tensor,
        head_ptrs: torch.Tensor,
        tail_ptrs: torch.Tensor,
        goose_body_idxs: torch.Tensor
):
    # Modifies geese_array in-place by removing all squares that the dead goose is on
    # Other aspects of goose death, such as modifying TorchEnv.alive are not done here
    n_dead_geese = geese.shape[0]
    if n_dead_geese == 0:
        return
    futures = [torch.jit.fork(
        _kill_goose,
        geese_tensor,
        geese[i],
        geese_env_idxs.view(-1)[i].item(),
        geese_idxs.view(-1)[i].item(),
        head_ptrs.view(-1)[i].item(),
        tail_ptrs.view(-1)[i].item(),
        goose_body_idxs
    ) for i in range(n_dead_geese)]
    results = [torch.jit.wait(fut) for fut in futures]
"""