from enum import *
try:
    import ujson as json
except ModuleNotFoundError:
    import json
from kaggle_environments.envs.hungry_geese.hungry_geese import Action
from kaggle_environments.envs.hungry_geese.hungry_geese import row_col as _row_col
from numbers import Number
import numpy as np
from pathlib import Path
import torch
from typing import *

from .config import *

STATE_TYPE = List[Dict]


class ActionMasking(Enum):
    """
    Available action masking settings returned by info_dict['available_actions_mask'].
    NONE: No action masking
    OPPOSITE: Only mask actions that are the opposite of the last action taken
    LETHAL: Mask opposite actions and any action that would result in moving into a square with a goose body
        (still allows movement into squares with goose heads for geese of length 1 or tails for any goose)
        Warning: LETHAL action masking will sometimes result in no available actions - take this into account
    """
    NONE = auto()
    OPPOSITE = auto()
    LETHAL = auto()

    def get_action_mask(self, state: STATE_TYPE) -> np.ndarray:
        available_actions_mask = np.ones((len(state), 4), dtype=np.bool)
        if self == ActionMasking.NONE:
            pass
        elif self == ActionMasking.OPPOSITE:
            if state[0]['observation']['step'] > 0:
                for agent_idx, agent in enumerate(state):
                    last_action = Action[agent['action']]
                    banned_action_idx = tuple(Action).index(last_action.opposite())
                    available_actions_mask[agent_idx, banned_action_idx] = False
        elif self == ActionMasking.LETHAL:
            if state[0]['observation']['step'] > 0:
                all_goose_locs = []
                for goose_loc_list in state[0]['observation']['geese']:
                    # Ignore geese that are only a head
                    if len(goose_loc_list) > 1:
                        # Don't mask the tail position
                        all_goose_locs += [row_col(n) for n in goose_loc_list[:-1]]
                for agent_idx, agent in enumerate(state):
                    goose_loc_list = state[0]['observation']['geese'][agent_idx]
                    if len(goose_loc_list) > 0:
                        last_action = Action[agent['action']]
                        banned_action_idx = tuple(Action).index(last_action.opposite())
                        available_actions_mask[agent_idx, banned_action_idx] = False
                        head_loc = np.array(row_col(goose_loc_list[0]))
                        for act in tuple(Action):
                            destination = head_loc + np.array(act.to_row_col())
                            destination[0] = (N_ROWS + destination[0]) % N_ROWS
                            destination[1] = (N_COLS + destination[1]) % N_COLS
                            if tuple(destination) in all_goose_locs:
                                banned_action_idx = tuple(Action).index(act)
                                available_actions_mask[agent_idx, banned_action_idx] = False
        else:
            raise ValueError(f'ActionMasking not yet implemented: {self.name}')

        return available_actions_mask


def row_col(position: int) -> Tuple[int, int]:
    return _row_col(position, N_COLS)


def rowwise_random_choice(p: np.ndarray) -> np.ndarray:
    """
    Numpy has no built-in way to take a random choice over a 2D array
    This implementation is based on the idea of inverse transform sampling

    Returns:
        a: The index of the sampled element from each row, with shape (p.shape[0],)
    """
    assert p.ndim == 2 and np.all(p >= 0.)
    p_norm = p / p.sum(axis=1, keepdims=True)
    return np.argmax(
        np.cumsum(p_norm, axis=1) > np.random.random_sample((p_norm.shape[0], 1)),
        axis=1
    )


def print_array_one_line(arr: Union[np.ndarray, Number]) -> str:
    if type(arr) == np.ndarray:
        return '[' + ', '.join([print_array_one_line(a) for a in arr]) + ']'
    else:
        return f'{arr:.2f}'


def read_json(file_path: Union[str, Path]):
    with open(file_path, 'rb') as f:
        return json.load(f)


def read_json_lines(file_path: Union[str, Path], line_idx: int):
    with open(file_path, 'rb') as f:
        return json.loads(f.readlines()[line_idx])


def read_ljson(file_path: Union[str, Path]) -> List:
    with open(file_path, 'rb') as f:
        return [json.loads(line) for line in f.readlines()]


def format_experiment_name(obs_type,
                           reward_type,
                           action_masking,
                           channel_dims: Sequence[int],
                           blocks: Sequence):
    experiment_name = ''
    experiment_name += f'{obs_type.name.lower()}_'
    experiment_name += f'{reward_type.name.lower()}_'
    experiment_name += f'{action_masking.name.lower()}_'
    experiment_name += f'{len(blocks)}_blocks_'
    experiment_name += '_'.join([str(c) for c in channel_dims]) + '_dims'
    return experiment_name


def torch_terminal_value_func(rewards: torch.Tensor) -> torch.Tensor:
    n_geese = rewards.shape[1]
    agent_rankings = torch_rankdata_average(rewards) - 1.
    agent_rankings_rescaled = 2. * agent_rankings / (n_geese - 1.) - 1.
    return agent_rankings_rescaled


def torch_rankdata_average(a: torch.Tensor) -> torch.Tensor:
    assert a.ndim == 2
    arr = a.clone()
    sorter = torch.argsort(arr, dim=-1)
    inv = torch.empty_like(sorter)
    inv.scatter_(-1, sorter, torch.arange(sorter.shape[-1], device=arr.device).unsqueeze(0).expand_as(sorter))

    arr = arr.gather(-1, sorter)
    obs = torch.cat([
        torch.ones((arr.shape[0], 1), dtype=torch.bool, device=arr.device),
        arr[:, 1:] != arr[:, :-1]
    ], dim=-1).to(torch.int64)
    dense = obs.cumsum(dim=-1).gather(-1, inv)
    """
    We need to take the rowwise indices of the nonzero elements. However, the number of nonzero elements may vary from
    row to row, so we pad the rows with additional elements so that the 1D nonzero indices can be reshaped to a matrix
    of shape (n_rows, n_cols). Note that this will not affect the final result, as any additional nonzero elements
    will not be gathered by the dense indices.
    """
    padding = torch.where(
        (obs == 0).sum(dim=-1, keepdim=True) > torch.arange(arr.shape[-1], device=arr.device).unsqueeze(0),
        torch.ones_like(obs),
        torch.zeros_like(obs)
    )
    # cumulative counts of each unique value
    count = torch.nonzero(torch.cat([obs, padding], dim=-1), as_tuple=True)[1].view(*arr.shape)
    count = torch.cat([count, torch.zeros((arr.shape[0], 1), dtype=count.dtype, device=arr.device)], dim=-1)
    count.scatter_(
        -1,
        (obs != 0).sum(dim=-1, keepdim=True),
        torch.zeros((arr.shape[0], 1), dtype=count.dtype, device=arr.device) + arr.shape[-1]
    )

    return 0.5 * (count.gather(-1, dense) + count.gather(-1, dense - 1) + 1)
