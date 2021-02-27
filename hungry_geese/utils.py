from enum import *
from kaggle_environments.envs.hungry_geese.hungry_geese import Action
from kaggle_environments.envs.hungry_geese.hungry_geese import row_col as _row_col
import numpy as np
from typing import *

from .config import *


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

    def get_action_mask(self, state: List[Dict]) -> np.ndarray:
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
