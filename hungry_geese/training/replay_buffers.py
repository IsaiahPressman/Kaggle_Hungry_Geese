import numpy as np
import random
import torch
from typing import *

from ..env import goose_env as ge


class BasicReplayBuffer:
    def __init__(self, s_shape: Tuple[int, ...],
                 max_len: Union[int, float] = 1e6,
                 starting_s_a_r_d_s: Optional[Tuple[torch.Tensor, ...]] = None):
        self.max_len = int(max_len)
        self._s_buffer = torch.zeros(self.max_len, *s_shape)
        self._a_buffer = torch.zeros(self.max_len, dtype=torch.long)
        self._r_buffer = torch.zeros(self.max_len)
        self._d_buffer = torch.zeros(self.max_len)
        self._next_s_buffer = torch.zeros(self.max_len, *s_shape)
        self.current_size = 0
        self._top = 0
        if starting_s_a_r_d_s is not None:
            self.append_samples_batch(*starting_s_a_r_d_s)
            # Randomly shuffle initial experiences
            shuffled_idxs = np.arange(self.current_size)
            np.random.shuffle(shuffled_idxs)
            shuffled_idxs = np.append(shuffled_idxs, np.arange(self.current_size, self.max_len))
            self._s_buffer = self._s_buffer[torch.from_numpy(shuffled_idxs)]
            self._a_buffer = self._a_buffer[torch.from_numpy(shuffled_idxs)]
            self._r_buffer = self._r_buffer[torch.from_numpy(shuffled_idxs)]
            self._d_buffer = self._d_buffer[torch.from_numpy(shuffled_idxs)]
            self._next_s_buffer = self._next_s_buffer[torch.from_numpy(shuffled_idxs)]

    def get_samples_batch(self, sample_size: int):
        # Sampling with replacement
        idxs = torch.randint(self.current_size, size=(sample_size,))
        # Sampling without replacement is possible, but quite a bit slower:
        # idxs = np.random.choice(self.current_size, size=sample_size, replace=(self.current_size < sample_size))
        return (self._s_buffer[idxs].clone(),
                self._a_buffer[idxs].clone(),
                self._r_buffer[idxs].clone(),
                self._d_buffer[idxs].clone(),
                self._next_s_buffer[idxs].clone())

    def append_samples_batch(self,
                             s_batch: torch.Tensor,
                             a_batch: torch.Tensor,
                             r_batch: torch.Tensor,
                             d_batch: torch.Tensor,
                             next_s_batch: torch.Tensor):
        batch_len = s_batch.shape[0]
        assert a_batch.shape[0] == batch_len
        assert r_batch.shape[0] == batch_len
        assert d_batch.shape[0] == batch_len
        assert next_s_batch.shape[0] == batch_len
        new_len = self._top + batch_len
        if new_len <= self.max_len:
            self._s_buffer[self._top:new_len] = s_batch
            self._a_buffer[self._top:new_len] = a_batch
            self._r_buffer[self._top:new_len] = r_batch
            self._d_buffer[self._top:new_len] = d_batch
            self._next_s_buffer[self._top:new_len] = next_s_batch
            self._top = new_len % self.max_len
            self.current_size = max(new_len, self.current_size)
        else:
            leftover_batch = new_len % self.max_len
            s_batch_split = s_batch.split((batch_len - leftover_batch, leftover_batch))
            a_batch_split = a_batch.split((batch_len - leftover_batch, leftover_batch))
            r_batch_split = r_batch.split((batch_len - leftover_batch, leftover_batch))
            d_batch_split = d_batch.split((batch_len - leftover_batch, leftover_batch))
            next_s_batch_split = next_s_batch.split((batch_len - leftover_batch, leftover_batch))
            self.append_samples_batch(s_batch_split[0],
                                      a_batch_split[0],
                                      r_batch_split[0],
                                      d_batch_split[0],
                                      next_s_batch_split[0])
            self.append_samples_batch(s_batch_split[1],
                                      a_batch_split[1],
                                      r_batch_split[1],
                                      d_batch_split[1],
                                      next_s_batch_split[1])

    def __len__(self):
        return self.current_size


class DataAugmentationReplayBuffer(BasicReplayBuffer):
    def __init__(self, obs_type: ge.ObsType, use_channel_shuffle: bool = True, *args, **kwargs):
        super(DataAugmentationReplayBuffer, self).__init__(*args, **kwargs)
        self.obs_type = obs_type
        self.use_channel_shuffle = use_channel_shuffle

    def get_samples_batch(self, sample_size):
        """
        Samples a batch of experiences and stochastically performs data-augmentation on the entire batch
        """
        s_batch, a_batch, r_batch, d_batch, next_s_batch = super().get_samples_batch(sample_size)
        if self.obs_type == ge.ObsType.SET_OBS:
            assert False, 'Not yet implemented'
        elif self.obs_type == ge.ObsType.HEAD_CENTERED_OBS_LARGE:
            n_geese = (s_batch.shape[1] - 3) / 11
            # Flip rows
            if random.random() < 0.5:
                flippable_channel_idxs = np.arange(n_geese * 11)
                flippable_channel_idxs = np.where(
                    np.logical_or(flippable_channel_idxs % 11 == 2, flippable_channel_idxs % 11 == 7),
                    flippable_channel_idxs + 2,
                    np.where(
                        np.logical_or(flippable_channel_idxs % 11 == 4, flippable_channel_idxs % 11 == 9),
                        flippable_channel_idxs - 2,
                        flippable_channel_idxs
                    )
                )
                channel_idxs = np.concatenate([flippable_channel_idxs,
                                               np.arange(n_geese * 11, n_geese * 11 + 3)])
                s_batch = torch.flip(s_batch, dims=(-2,))[:, channel_idxs]
                a_batch = flip_a_batch(a_batch, 'rows')
                next_s_batch = torch.flip(next_s_batch, dims=(-2,))[:, channel_idxs]
            # Flip columns
            if random.random() < 0.5:
                flippable_channel_idxs = np.arange(n_geese * 11)
                flippable_channel_idxs = np.where(
                    np.logical_or(flippable_channel_idxs % 11 == 3, flippable_channel_idxs % 11 == 8),
                    flippable_channel_idxs + 2,
                    np.where(
                        np.logical_or(flippable_channel_idxs % 11 == 5, flippable_channel_idxs % 11 == 10),
                        flippable_channel_idxs - 2,
                        flippable_channel_idxs
                    )
                )
                channel_idxs = np.concatenate([flippable_channel_idxs,
                                               np.arange(n_geese * 11, n_geese * 11 + 3)])
                s_batch = torch.flip(s_batch, dims=(-1,))[:, channel_idxs]
                a_batch = flip_a_batch(a_batch, 'cols')
                next_s_batch = torch.flip(next_s_batch, dims=(-1,))[:, channel_idxs]
            # Randomly shuffle channels pertaining to geese other than oneself
            if self.use_channel_shuffle:
                new_other_goose_idxs = np.random.permutation(np.arange(1, n_geese))
                new_channel_idxs_full = np.concatenate([
                    np.arange(11),
                    (new_other_goose_idxs[:, np.newaxis] * 11 + np.arange(11)).ravel(),
                    np.arange(n_geese * 11, n_geese * 11 + 3)
                ])
                s_batch = s_batch[:, new_channel_idxs_full]
                next_s_batch = next_s_batch[:, new_channel_idxs_full]
        elif self.obs_type == ge.ObsType.HEAD_CENTERED_OBS_SMALL:
            n_geese = (s_batch.shape[1] - 3) / 3
            # Flip rows
            if random.random() < 0.5:
                s_batch = torch.flip(s_batch, dims=(-2,))
                a_batch = flip_a_batch(a_batch, 'rows')
                next_s_batch = torch.flip(next_s_batch, dims=(-2,))
            # Flip columns
            if random.random() < 0.5:
                s_batch = torch.flip(s_batch, dims=(-1,))
                a_batch = flip_a_batch(a_batch, 'cols')
                next_s_batch = torch.flip(next_s_batch, dims=(-1,))
            # Randomly shuffle channels pertaining to geese other than oneself
            if self.use_channel_shuffle:
                new_other_goose_idxs = np.random.permutation(np.arange(1, n_geese))
                new_channel_idxs_full = np.concatenate([
                    np.arange(3),
                    (new_other_goose_idxs[:, np.newaxis] * 3 + np.arange(3)).ravel(),
                    np.arange(n_geese * 3, n_geese * 3 + 3)
                ])
                s_batch = s_batch[:, new_channel_idxs_full]
                next_s_batch = next_s_batch[:, new_channel_idxs_full]
        elif self.obs_type == ge.ObsType.HEAD_CENTERED_SET_OBS:
            assert False, 'Not yet implemented'
        else:
            raise ValueError(f'Unrecognized obs_type: {self.obs_type}')
        return s_batch, a_batch, r_batch, d_batch, next_s_batch


def flip_a_batch(a_batch, flipped_rows_or_cols):
    if flipped_rows_or_cols == 'rows':
        flipped_a_batch = torch.where(
            a_batch == 0,
            2,
            torch.where(
                a_batch == 2,
                0,
                a_batch
            )
        )
    elif flipped_rows_or_cols == 'cols':
        flipped_a_batch = torch.where(
            a_batch == 1,
            3,
            torch.where(
                a_batch == 3,
                1,
                a_batch
            )
        )
    else:
        raise ValueError(f'Unrecognized flipped_rows_or_cols value: {flipped_rows_or_cols}')
    return flipped_a_batch


class AlphaZeroReplayBuffer:
    """
    A replay buffer that stores (state, policy, result, head_loc, still_alive) tuples
    state: Tensor of shape (n_channels, n_rows, n_cols)
    policy: Tensor of shape (n_geese, 4) representing the post-MCTS policy
    result: Tensor of shape (n_geese,) representing the final reward of that goose
    head_loc: Tensor of shape (n_geese,) representing the integer indices of the head locations of the living geese
        NB: Dead geese masks are stored by setting head_loc to -1
    dead_mask: Boolean tensor of shape (n_geese,) representing which geese are no longer playing
    """
    def __init__(self,
                 s_shape: Tuple[int, ...],
                 max_len: Union[int, float] = 1e6,
                 starting_s_p_r_h_d: Optional[Tuple[torch.Tensor, ...]] = None):
        self.max_len = int(max_len)
        n_geese = self._p_buffer[1]
        self._s_buffer = torch.zeros(self.max_len, *s_shape)
        self._p_buffer = torch.zeros(self.max_len, n_geese, 4)
        self._r_buffer = torch.zeros(self.max_len, n_geese)
        self._h_buffer = torch.zeros(self.max_len, n_geese)
        self.current_size = 0
        self._top = 0
        if starting_s_p_r_h_d is not None:
            self.append_samples_batch(*starting_s_p_r_h_d)
            # Randomly shuffle initial experiences
            shuffled_idxs = np.arange(self.current_size)
            np.random.shuffle(shuffled_idxs)
            shuffled_idxs = np.append(shuffled_idxs, np.arange(self.current_size, self.max_len))
            self._s_buffer = self._s_buffer[torch.from_numpy(shuffled_idxs)]
            self._p_buffer = self._p_buffer[torch.from_numpy(shuffled_idxs)]
            self._r_buffer = self._r_buffer[torch.from_numpy(shuffled_idxs)]
            self._h_buffer = self._h_buffer[torch.from_numpy(shuffled_idxs)]

    def get_samples_batch(self, sample_size: int):
        # Sampling with replacement
        idxs = torch.randint(self.current_size, size=(sample_size,))
        # Sampling without replacement is possible, but quite a bit slower:
        # idxs = np.random.choice(self.current_size, size=sample_size, replace=(self.current_size < sample_size))
        h_batch = self._h_buffer[idxs].clone()
        d_batch = h_batch < 0
        return (self._s_buffer[idxs].clone(),
                self._p_buffer[idxs].clone(),
                self._r_buffer[idxs].clone(),
                h_batch,
                d_batch)

    def append_samples_batch(self,
                             s_batch: torch.Tensor,
                             p_batch: torch.Tensor,
                             r_batch: torch.Tensor,
                             h_batch: torch.Tensor,
                             d_batch: torch.Tensor):
        batch_len = s_batch.shape[0]
        assert p_batch.shape[0] == batch_len
        assert r_batch.shape[0] == batch_len
        assert h_batch.shape[0] == batch_len
        assert d_batch.shape[0] == batch_len
        new_len = self._top + batch_len
        if new_len <= self.max_len:
            h_batch = torch.where(
                d_batch,
                -100,
                h_batch
            )
            self._s_buffer[self._top:new_len] = s_batch
            self._p_buffer[self._top:new_len] = p_batch
            self._r_buffer[self._top:new_len] = r_batch
            self._h_buffer[self._top:new_len] = h_batch
            self._top = new_len % self.max_len
            self.current_size = max(new_len, self.current_size)
        else:
            leftover_batch = new_len % self.max_len
            s_batch_split = s_batch.split((batch_len - leftover_batch, leftover_batch))
            p_batch_split = p_batch.split((batch_len - leftover_batch, leftover_batch))
            r_batch_split = r_batch.split((batch_len - leftover_batch, leftover_batch))
            h_batch_split = h_batch.split((batch_len - leftover_batch, leftover_batch))
            d_batch_split = d_batch.split((batch_len - leftover_batch, leftover_batch))
            self.append_samples_batch(s_batch_split[0],
                                      p_batch_split[0],
                                      r_batch_split[0],
                                      h_batch_split[0],
                                      d_batch_split[0])
            self.append_samples_batch(s_batch_split[1],
                                      p_batch_split[1],
                                      r_batch_split[1],
                                      h_batch_split[1],
                                      d_batch_split[1])

    def __len__(self):
        return self.current_size
