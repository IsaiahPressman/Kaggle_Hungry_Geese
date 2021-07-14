import functools
import itertools
from kaggle_environments.envs.hungry_geese.hungry_geese import Action
import numpy as np
from pathlib import Path
import random
import torch
from torch.utils.data import Dataset
from typing import *

from ...config import N_PLAYERS
from ...env.goose_env import ObsType, create_obs_tensor
from ...utils import read_json_lines


class AlphaGooseDataset(Dataset):
    def __init__(self,
                 dataset_dir: Union[str, Path],
                 obs_type: ObsType,
                 transform: Optional[Callable] = None,
                 is_valid_file: Optional[Callable] = None):
        if is_valid_file is None:
            def is_valid_file(_):
                return True
        self.episodes = [e for e in Path(dataset_dir).glob('*.ljson') if is_valid_file(e)]
        self.samples = []
        for episode_path in self.episodes:
            with open(episode_path, 'rb') as f:
                step_list = f.readlines()
                self.samples.extend([(episode_path, step_idx) for step_idx in range(len(step_list))])
        self.obs_type = obs_type
        self.transform = transform
        if self.obs_type not in (ObsType.COMBINED_GRADIENT_OBS_SMALL, ObsType.COMBINED_GRADIENT_OBS_LARGE):
            raise ValueError('Other obs_types have not yet been implemented, '
                             'they may need different data concatenation')

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        episode_path, step_idx = self.samples[index]
        try:
            step = read_json_lines(episode_path, step_idx)
        except ValueError:
            raise ValueError(f'Could not read line {step_idx} from file {episode_path}')

        state = create_obs_tensor(step, self.obs_type)
        policies = []
        available_actions_masks = []
        final_ranks = []
        head_locs = []
        still_alive = []
        for agent_idx, agent in enumerate(step):
            still_alive.append(agent['status'] == 'ACTIVE')
            final_ranks.append(agent['final_rank'])
            if still_alive[-1]:
                policies.append(np.array(agent['policy']))
                available_actions_masks.append(np.array(agent['available_actions_mask']))
                head_locs.append(step[0]['observation']['geese'][agent_idx][0])
            else:
                policies.append(np.zeros(4))
                available_actions_masks.append(np.zeros(4))
                head_locs.append(-1)
        ranks_rescaled = 2. * np.array(final_ranks) / (4. - 1.) - 1.

        sample = (state.squeeze(axis=0).astype(np.float32),
                  np.stack(policies, axis=0).astype(np.float32),
                  np.stack(available_actions_masks, axis=0).astype(np.bool),
                  ranks_rescaled.astype(np.float32),
                  np.array(head_locs).astype(np.int64),
                  np.array(still_alive).astype(np.bool))

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
                 include_episode: Optional[Callable] = None):
        if include_episode is None:
            def include_episode(_):
                return True
        self.episodes = [d for d in Path(root).glob('*.ljson') if include_episode(d)]
        self.samples = []
        for episode_path in self.episodes:
            with open(episode_path, 'rb') as f:
                step_list = f.readlines()
                self.samples.extend([(episode_path, step_idx) for step_idx in range(len(step_list))])
        self.obs_type = obs_type
        self.transform = transform
        if self.obs_type not in (ObsType.COMBINED_GRADIENT_OBS_SMALL, ObsType.COMBINED_GRADIENT_OBS_LARGE):
            raise ValueError('Other obs_types have not yet been implemented, '
                             'they will need different data concatenation')

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        episode_path, step_idx = self.samples[index]
        step = read_json_lines(episode_path, step_idx)

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

        sample = (state.squeeze(axis=0).astype(np.float32),
                  np.array(actions).astype(np.int64),
                  ranks_rescaled.astype(np.float32),
                  np.array(head_locs).astype(np.int64),
                  np.array(still_alive).astype(np.bool))

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
        if self.obs_type not in (ObsType.COMBINED_GRADIENT_OBS_SMALL, ObsType.COMBINED_GRADIENT_OBS_LARGE):
            raise ValueError('Other obs_types have not yet been implemented.')

    def __call__(
            self,
            sample: Sequence[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        state, policies, available_actions_masks, ranks_rescaled, head_locs, still_alive = sample
        if self.obs_type in (ObsType.COMBINED_GRADIENT_OBS_SMALL, ObsType.COMBINED_GRADIENT_OBS_LARGE):
            new_head_locs = np.arange(state.shape[-2] * state.shape[-1]).reshape(*state.shape[-2:])
            # Flip vertically
            if random.random() < 0.5:
                state = np.flip(state, axis=-2)
                policies = flip_policies(policies, 'rows')
                available_actions_masks = flip_policies(available_actions_masks, 'rows')
                new_head_locs = np.flip(new_head_locs, axis=-2)
            # Flip horizontally
            if random.random() < 0.5:
                state = np.flip(state, axis=-1)
                policies = flip_policies(policies, 'cols')
                available_actions_masks = flip_policies(available_actions_masks, 'cols')
                new_head_locs = np.flip(new_head_locs, axis=-1)
            head_locs = np.where(
                still_alive,
                new_head_locs.ravel()[head_locs.ravel()].reshape(head_locs.shape),
                -1
            )
        else:
            raise ValueError(f'Not yet a supported obs_type: {self.obs_type}')
        return state, policies, available_actions_masks, ranks_rescaled, head_locs, still_alive


class PretrainRandomReflect:
    """
    Given a tuple of (state, actions, ranks_rescaled, head_locs, still_alive) arrays, randomly reflect the
    states and actions either vertically, horizontally, or both
    """

    def __init__(self, obs_type: ObsType):
        self.obs_type = obs_type

    def __call__(
            self,
            sample: Sequence[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        state, actions, ranks_rescaled, head_locs, still_alive = sample
        if self.obs_type in (ObsType.COMBINED_GRADIENT_OBS_SMALL, ObsType.COMBINED_GRADIENT_OBS_LARGE):
            new_head_locs = np.arange(state.shape[-2] * state.shape[-1]).reshape(*state.shape[-2:])
            # Flip vertically
            if random.random() < 0.5:
                state = np.flip(state, axis=-2)
                actions = flip_actions(actions, 'rows')
                new_head_locs = np.flip(new_head_locs, axis=-2)
            # Flip horizontally
            if random.random() < 0.5:
                state = np.flip(state, axis=-1)
                actions = flip_actions(actions, 'cols')
                new_head_locs = np.flip(new_head_locs, axis=-1)
            head_locs = np.where(
                still_alive,
                new_head_locs.ravel()[head_locs.ravel()].reshape(head_locs.shape),
                -1
            )
        else:
            raise NotImplementedError(f'Not yet a supported obs_type: {self.obs_type}')
        return state, actions, ranks_rescaled, head_locs, still_alive


class ChannelShuffle:
    """
    Given a tuple of (state, actions, ranks_rescaled, head_locs, still_alive) arrays, randomly shuffle the channels
    pertaining to given geese, such that goose 1 may become goose 0.
    NB: Technically, there may be a couple edge cases where the goose index matters, but these are overwhelmingly few
    and far between.
    """

    def __init__(self, obs_type: ObsType, n_players: int = N_PLAYERS):
        self.obs_type = obs_type
        self.n_players = n_players

    def __call__(
            self,
            sample: Sequence[np.ndarray]
    ) -> Sequence[np.ndarray]:
        state, *other_arrays = sample
        shuffled_player_idxs = np.random.permutation(self.n_players)
        shuffled_channel_idxs = self.get_idxs(tuple(shuffled_player_idxs))

        return tuple([state[shuffled_channel_idxs, :, :]] + [arr[shuffled_player_idxs] for arr in other_arrays])

    @functools.lru_cache()
    def get_idxs(self, shuffled_player_idxs: Tuple) -> List[int]:
        if self.obs_type == ObsType.COMBINED_GRADIENT_OBS_SMALL:
            n = 2
            pre_idxs = []
            channel_idxs = [[i for i in range(p * n, p * n + n)] for p in range(self.n_players)]
            post_idxs = [-3, -2, -1]
        elif self.obs_type == ObsType.COMBINED_GRADIENT_OBS_LARGE:
            n = 3
            pre_idxs = []
            channel_idxs = [[i for i in range(p * n, p * n + n)] for p in range(self.n_players)]
            post_idxs = [-3, -2, -1]
        else:
            raise NotImplementedError(f'Not yet a supported obs_type: {self.obs_type}')

        shuffled_channel_idxs = list(itertools.chain.from_iterable([channel_idxs[i] for i in shuffled_player_idxs]))
        return pre_idxs + shuffled_channel_idxs + post_idxs


class ToTensor:
    """
    Given a sequence of numpy arrays, convert them to pytorch tensors
    """

    def __call__(
            self,
            sample: Sequence[np.ndarray]
    ) -> List[torch.Tensor]:
        return [torch.from_numpy(arr.copy()) for arr in sample]


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
