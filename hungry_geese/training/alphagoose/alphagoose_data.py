from kaggle_environments.envs.hungry_geese.hungry_geese import Action
import numpy as np
from pathlib import Path
import random
import torch
from torch.utils.data import Dataset
from typing import *

from ...env.goose_env import ObsType, create_obs_tensor
from ...utils import read_json


class AlphaGooseDataset(Dataset):
    def __init__(self,
                 root: Union[str, Path],
                 obs_type: ObsType,
                 transform: Optional[Callable] = None,
                 include_episode: Optional[Callable] = None,
                 file_ext: str = '.json'):
        self.episodes = [d for d in Path(root).glob('*') if d.is_dir() and include_episode(d)]
        self.samples = [f for e in self.episodes for f in e.glob(f'*{file_ext}')]
        self.obs_type = obs_type
        self.transform = transform
        if self.obs_type != ObsType.COMBINED_GRADIENT_OBS:
            raise ValueError('Other obs_types have not yet been implemented, '
                             'they will need different data concatenation')

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        step = read_json(self.samples[index])

        state = create_obs_tensor(step, self.obs_type)
        policies = []
        final_ranks = []
        head_locs = []
        still_alive = []
        for agent_idx, agent in enumerate(step):
            still_alive.append(agent['status'] == 'ACTIVE' and agent['next_action'] is not None)
            final_ranks.append(agent['final_rank'])
            if still_alive[-1]:
                policies.append(np.array(agent['policy']))
                head_locs.append(step[0]['observation']['geese'][agent_idx][0])
            else:
                policies.append(np.zeros(4))
                head_locs.append(-1)
        ranks_rescaled = 2. * np.array(final_ranks) / (4. - 1.) - 1.

        sample = (state.squeeze(axis=0),
                  np.stack(policies, axis=0),
                  ranks_rescaled,
                  np.array(head_locs),
                  np.array(still_alive))

        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self) -> int:
        return len(self.samples)


class AlphaGoosePretrainDataset(Dataset):
    def __init__(self,
                 root: Union[str, Path],
                 obs_type: ObsType,
                 transform: Optional[Callable] = None,
                 include_episode: Optional[Callable] = None,
                 file_ext: str = '.json'):
        self.episodes = [d for d in Path(root).glob('*') if d.is_dir() and include_episode(d)]
        self.samples = [f for e in self.episodes for f in e.glob(f'*{file_ext}')]
        self.obs_type = obs_type
        self.transform = transform
        if self.obs_type != ObsType.COMBINED_GRADIENT_OBS:
            raise ValueError('Other obs_types have not yet been implemented, '
                             'they will need different data concatenation')

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        step = read_json(self.samples[index])

        state = create_obs_tensor(step, self.obs_type)
        actions = []
        final_ranks = []
        head_locs = []
        still_alive = []
        for agent_idx, agent in enumerate(step):
            still_alive.append(agent['status'] == 'ACTIVE' and agent['next_action'] is not None)
            final_ranks.append(agent['final_rank'])
            if still_alive[-1]:
                actions.append(Action[agent['next_action']].value - 1)
                head_locs.append(step[0]['observation']['geese'][agent_idx][0])
            else:
                actions.append(-1)
                head_locs.append(-1)
        ranks_rescaled = 2. * np.array(final_ranks) / (4. - 1.) - 1.

        sample = (state.squeeze(axis=0),
                  np.array(actions),
                  ranks_rescaled,
                  np.array(head_locs),
                  np.array(still_alive))

        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self) -> int:
        return len(self.samples)


class AlphaGooseRandomReflect:
    """
    Given a tuple of (state, policy, ranks_rescaled, head_locs, still_alive) arrays, randomly reflect the
    states and actions either vertically, horizontally, or both
    """

    def __init__(self, obs_type: ObsType):
        self.obs_type = obs_type
        if self.obs_type != ObsType.COMBINED_GRADIENT_OBS:
            raise ValueError('Other obs_types have not yet been implemented.')

    def __call__(
            self,
            sample: Sequence[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        state, policies, ranks_rescaled, head_locs, still_alive = sample
        if self.obs_type == ObsType.COMBINED_GRADIENT_OBS:
            new_head_locs = np.arange(state.shape[-2] * state.shape[-1]).reshape(*state.shape[-2:])
            # Flip vertically
            if random.random() < 0.5:
                state = np.flip(state, axis=-2)
                actions = flip_policies(policies, 'rows')
                new_head_locs = np.flip(new_head_locs, axis=-2,)
            # Flip horizontally
            if random.random() < 0.5:
                state = np.flip(state, axis=-1,)
                actions = flip_policies(policies, 'cols')
                new_head_locs = np.flip(new_head_locs, axis=-1,)
            head_locs = np.where(
                still_alive,
                new_head_locs.ravel()[head_locs.ravel()].reshape(head_locs.shape),
                -1
            )
        else:
            raise ValueError(f'Not yet a supported obs_type: {self.obs_type}')
        return state, policies, ranks_rescaled, head_locs, still_alive


class PretrainRandomReflect:
    """
    Given a tuple of (state, actions, ranks_rescaled, head_locs, still_alive) arrays, randomly reflect the
    states and actions either vertically, horizontally, or both
    """

    def __init__(self, obs_type: ObsType):
        self.obs_type = obs_type
        if self.obs_type != ObsType.COMBINED_GRADIENT_OBS:
            raise ValueError('Other obs_types have not yet been implemented.')

    def __call__(
            self,
            sample: Sequence[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        state, actions, ranks_rescaled, head_locs, still_alive = sample
        if self.obs_type == ObsType.COMBINED_GRADIENT_OBS:
            new_head_locs = np.arange(state.shape[-2] * state.shape[-1]).reshape(*state.shape[-2:])
            # Flip vertically
            if random.random() < 0.5:
                state = np.flip(state, axis=-2)
                actions = flip_actions(actions, 'rows')
                new_head_locs = np.flip(new_head_locs, axis=-2,)
            # Flip horizontally
            if random.random() < 0.5:
                state = np.flip(state, axis=-1,)
                actions = flip_actions(actions, 'cols')
                new_head_locs = np.flip(new_head_locs, axis=-1,)
            head_locs = np.where(
                still_alive,
                new_head_locs.ravel()[head_locs.ravel()].reshape(head_locs.shape),
                -1
            )
        else:
            raise ValueError(f'Not yet a supported obs_type: {self.obs_type}')
        return state, actions, ranks_rescaled, head_locs, still_alive


class ToTensor:
    """
    Given a tuple of (state, actions, ranks_rescaled, head_locs, still_alive) arrays, convert them to pytorch tensors
    """
    def __call__(
            self,
            sample: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        state, actions, ranks_rescaled, head_locs, still_alive = sample
        return (torch.tensor(state.copy(), dtype=torch.float32),
                torch.tensor(actions, dtype=torch.int64),
                torch.tensor(ranks_rescaled, dtype=torch.float32),
                torch.tensor(head_locs, dtype=torch.int64),
                torch.tensor(still_alive, dtype=torch.bool))


def flip_policies(policies: np.ndarray, flipped_rows_or_cols: str) -> np.ndarray:
    if flipped_rows_or_cols == 'rows':
        # Flip NORTH and SOUTH
        flipped_policies = policies[:, [2, 1, 0, 3]]
    elif flipped_rows_or_cols == 'cols':
        # Flip EAST and WEST
        flipped_policies = policies[:, [0, 3, 2, 1]]
    else:
        raise ValueError(f'Unrecognized flipped_rows_or_cols value: {flipped_rows_or_cols}')
    return flipped_policies


def flip_actions(actions: np.ndarray, flipped_rows_or_cols: str) -> np.ndarray:
    if flipped_rows_or_cols == 'rows':
        # Flip NORTH and SOUTH
        flipped_actions = np.where(
            actions == 0,
            2,
            np.where(
                actions == 2,
                0,
                actions
            )
        )
    elif flipped_rows_or_cols == 'cols':
        # Flip EAST and WEST
        flipped_actions = np.where(
            actions == 1,
            3,
            np.where(
                actions == 3,
                1,
                actions
            )
        )
    else:
        raise ValueError(f'Unrecognized flipped_rows_or_cols value: {flipped_rows_or_cols}')
    return flipped_actions

