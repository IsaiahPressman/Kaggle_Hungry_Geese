
import base64
import copy
from enum import *
from kaggle_environments import make as kaggle_make
from kaggle_environments.envs.hungry_geese.hungry_geese import Action, Configuration, Observation, histogram, translate
from kaggle_environments.envs.hungry_geese.hungry_geese import row_col as _row_col
from numbers import Number
import numpy as np
from pathlib import Path
import pickle
from random import sample
from scipy import stats
import time
import torch
from torch import distributions, nn
import torch.nn.functional as F
from typing import *

HUNGER_RATE = 40.
MAX_NUM_STEPS = 200.
GOOSE_MAX_LEN = 99.
N_ROWS = 7
N_COLS = 11
N_PLAYERS = 4


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


def print_array_one_line(arr: Union[np.ndarray, Number]) -> str:
    if type(arr) == np.ndarray:
        return '[' + ', '.join([print_array_one_line(a) for a in arr]) + ']'
    else:
        return f'{arr:.2f}'


def read_json(file_path: Union[str, Path]):
    with open(file_path, 'rb') as f:
        return ujson.load(f)


def read_json_lines(file_path: Union[str, Path], line_idx: int):
    with open(file_path, 'rb') as f:
        return ujson.loads(f.readlines()[line_idx])


def format_experiment_name(obs_type,
                           reward_type,
                           action_masking,
                           channel_dims: Sequence[int],
                           conv_block_kwargs: Sequence[Dict]):
    experiment_name = ''
    experiment_name += f'{obs_type.name.lower()}_'
    experiment_name += f'{reward_type.name.lower()}_'
    experiment_name += f'{action_masking.name.lower()}_'
    experiment_name += f'{len(conv_block_kwargs)}_blocks_'
    experiment_name += '_'.join([str(c) for c in channel_dims]) + '_dims'
    return experiment_name


class ObsType(Enum):
    """
    An enum of all available obs_types
    WARNING: enum order is subject to change
    """
    COMBINED_GRADIENT_OBS = auto()
    COMBINED_OBS_SMALL = auto()
    HEAD_CENTERED_OBS_LARGE = auto()
    HEAD_CENTERED_OBS_SMALL = auto()
    HEAD_CENTERED_SET_OBS = auto()
    SET_OBS = auto()

    def get_obs_spec(self, n_players: int = N_PLAYERS) -> Tuple[int, ...]:
        if self == ObsType.COMBINED_GRADIENT_OBS:
            return -1, 3 + 2 * n_players, N_ROWS, N_COLS
        elif self == ObsType.COMBINED_OBS_SMALL:
            return -1, 3 + 3 * n_players, N_ROWS, N_COLS
        elif self == ObsType.HEAD_CENTERED_OBS_LARGE:
            return -1, n_players, 11 * n_players + 3, N_ROWS, N_COLS
        elif self == ObsType.HEAD_CENTERED_OBS_SMALL:
            return -1, n_players, 3 + 3 * n_players, N_ROWS, N_COLS
        elif self == ObsType.HEAD_CENTERED_SET_OBS:
            return -1, n_players, 13, N_ROWS, N_COLS
        elif self == ObsType.SET_OBS:
            return -1, 13, N_ROWS, N_COLS
        else:
            raise ValueError(f'ObsType not yet implemented: {self.name}')


class RewardType(Enum):
    EVERY_STEP_ZERO_SUM = auto()
    EVERY_STEP_LENGTH = auto()
    ON_EAT_AND_ON_DEATH = auto()
    RANK_ON_DEATH = auto()

    def get_cumulative_reward_spec(self) -> Tuple[Optional[float], Optional[float]]:
        """
        The minimum/maximum cumulative available reward
        """
        if self == RewardType.EVERY_STEP_ZERO_SUM:
            return 0., 1.
        elif self == RewardType.EVERY_STEP_LENGTH:
            return 0., 1.
        elif self == RewardType.ON_EAT_AND_ON_DEATH:
            return -1., 1.
        elif self == RewardType.RANK_ON_DEATH:
            return -1., 1.
        else:
            raise ValueError(f'RewardType not yet implemented: {self.name}')

    def get_recommended_value_activation_scale_shift_dict(self) -> Dict:
        """
        The recommended value activation function, value_scale, and value_shift for the Q/value model
        """
        if self == RewardType.EVERY_STEP_ZERO_SUM:
            value_activation = nn.Sigmoid()
            value_scale = 1.
            value_shift = 0.
        elif self == RewardType.EVERY_STEP_LENGTH:
            value_activation = nn.Sigmoid()
            value_scale = 1.
            value_shift = 0.
        elif self == RewardType.ON_EAT_AND_ON_DEATH:
            value_activation = nn.Tanh()
            value_scale = 1.
            value_shift = 0.
        elif self == RewardType.RANK_ON_DEATH:
            value_activation = nn.Tanh()
            value_scale = 1.
            value_shift = 0.
        else:
            raise ValueError(f'RewardType not yet implemented: {self.name}')
        return {
            'value_activation': value_activation,
            'value_scale': value_scale,
            'value_shift': value_shift
        }


_DIRECTIONS_DICT = {act.to_row_col(): act.name for act in Action}
for key, val in copy.copy(_DIRECTIONS_DICT).items():
    if key == (-1, 0):
        _DIRECTIONS_DICT[(6, 0)] = val
    elif key == (1, 0):
        _DIRECTIONS_DICT[(-6, 0)] = val
    elif key == (0, 1):
        _DIRECTIONS_DICT[(0, -10)] = val
    elif key == (0, -1):
        _DIRECTIONS_DICT[(0, 10)] = val
    else:
        raise ValueError(f'Unrecognized direction_dict key-val pair: {key}, {val}')


def _get_direction(from_position: int, to_position: int) -> str:
    from_loc = np.array(row_col(from_position))
    to_loc = np.array(row_col(to_position))
    return _DIRECTIONS_DICT[tuple(to_loc - from_loc)]


class VectorizedEnv:
    def __init__(self, obs_type: Union[ObsType, Sequence[ObsType]], reward_type: RewardType,
                 action_masking: ActionMasking = ActionMasking.OPPOSITE,
                 n_envs: int = 1, n_players: int = N_PLAYERS, silent_reset: bool = True, make_fn=None):
        self.obs_type = obs_type
        self.reward_type = reward_type
        self.action_masking = action_masking
        self.n_envs = n_envs
        self.n_players = n_players
        self.silent_reset = silent_reset
        self.make_fn = make_fn

        self.multi_obs_type = type(self.obs_type) != ObsType
        self.wrapped_envs = [self.make_fn('hungry_geese') for _ in range(self.n_envs)]
        self.agent_dones = np.zeros((self.n_envs, self.n_players), dtype=np.bool)
        self.goose_head_locs = np.zeros((self.n_envs, self.n_players, 2), dtype=np.int64)
        self.available_actions_mask = np.ones((self.n_envs, self.n_players, 4), dtype=np.bool)
        self.episodes_finished_last_turn = np.zeros(self.n_envs, dtype=np.bool)
        self.episodes_finished_last_turn_info = [{} for _ in range(self.n_envs)]

        self._hard_reset()

    def hard_reset(self):
        """
        Resets all environments, whether or not they are done
        """
        self._hard_reset()
        rewards = np.zeros((self.n_envs, self.n_players))
        return self.obs, rewards, self.agent_dones, self.info_dict

    def _hard_reset(self):
        """
        Resets all environments, whether or not they are done, but does not return an observation
        """
        self._reset_specific_envs(np.ones_like(self.episodes_done))
        self._update_other_info()

    def soft_reset(self):
        """
        Only resets environments that are done and returns an observation
        """
        self._soft_reset()
        rewards = np.zeros((self.n_envs, self.n_players))
        return self.obs, rewards, self.agent_dones, self.info_dict

    def _soft_reset(self):
        """
        Only resets environments that are done, but does not return an observation
        """
        self._reset_specific_envs(self.episodes_done)
        self._update_other_info()

    def _reset_specific_envs(self, reset_mask: Sequence[bool]):
        for env_idx, do_reset in enumerate(reset_mask):
            env = self.wrapped_envs[env_idx]
            if do_reset:
                self.episodes_finished_last_turn[env_idx] = True
                finished_episode_info = {
                    'goose_death_times': [agent['reward'] // (GOOSE_MAX_LEN + 1.) for agent in env.steps[-1]],
                    'goose_lengths': [max(agent['reward'] % (GOOSE_MAX_LEN + 1.), 1.) for agent in env.steps[-1]],
                    'goose_rankings': stats.rankdata([agent['reward'] for agent in env.steps[-1]], method='max'),
                    'n_steps': env.steps[-1][0]['observation']['step'],
                }
                self.episodes_finished_last_turn_info[env_idx] = finished_episode_info
                env.reset(num_agents=self.n_players)
                self.agent_dones[env_idx] = 0.
            else:
                self.episodes_finished_last_turn[env_idx] = False
                self.episodes_finished_last_turn_info[env_idx] = {}

    def step(self, actions: np.ndarray):
        if actions.shape != (self.n_envs, self.n_players):
            raise RuntimeError(f'actions.shape should have been {(self.n_envs, self.n_players)}, '
                               f'was {actions.shape}')
        if self.episodes_done.any():
            raise RuntimeError('The environment needs to be reset, or silent_reset needs to be True')
        assert np.all(0 <= actions)
        assert np.all(actions < len(Action))
        agent_already_dones = self.agent_dones.copy()

        current_standings = np.zeros((self.n_envs, self.n_players))
        for i in range(self.n_envs):
            observations = self.wrapped_envs[i].step([tuple(Action)[act].name for act in actions[i]])
            self.agent_dones[i] = [agent['status'] == 'DONE' for agent in observations]
            current_standings[i] = [agent['reward'] for agent in observations]

        rewards = np.zeros((self.n_envs, self.n_players))
        if self.reward_type == RewardType.EVERY_STEP_ZERO_SUM:
            # A small reward is given to a goose when it eats
            new_kaggle_rewards = np.array([[agent['reward'] for agent in env.steps[-1]] for env in self.wrapped_envs])
            goose_ate = np.logical_or(
                new_kaggle_rewards > (self.kaggle_rewards + GOOSE_MAX_LEN + 1.),
                np.logical_and(
                    new_kaggle_rewards == (self.kaggle_rewards + GOOSE_MAX_LEN + 1.),
                    (self.kaggle_timesteps[:, np.newaxis] + 1) % HUNGER_RATE == 0
                )
            )
            # Rewards start at 0, not 101, so this np.where catches that case
            goose_ate = np.where(
                self.kaggle_timesteps[:, np.newaxis] == 0,
                new_kaggle_rewards > 2 * (GOOSE_MAX_LEN + 1.) + 1.,
                goose_ate
            )
            # For the episodes that aren't finished, a reward of 1 is divided among the remaining players
            epsilon = 1e-10
            rewards = np.where(
                ~self.agent_dones,
                goose_ate / (N_ROWS * N_COLS) + (1. / ((~self.agent_dones).sum(axis=1, keepdims=True) + epsilon)),
                0.
            )
            n_steps_remaining = [MAX_NUM_STEPS - env.steps[-1][0]['observation']['step'] for env in self.wrapped_envs]
            n_steps_remaining = np.array(n_steps_remaining, dtype=np.float32)[:, np.newaxis]
            assert np.all(n_steps_remaining > 0)
            # Awards 1st place all remaining points
            # There is a tactical edge case where 2nd should get more reward than 3rd, but does not
            n_first_places = (current_standings == current_standings.max(axis=1, keepdims=True)).sum(axis=1,
                                                                                                     keepdims=True)
            rewards = np.where(
                self.agent_dones.all(axis=1, keepdims=True),
                np.where(
                    current_standings == current_standings.max(axis=1, keepdims=True),
                    n_steps_remaining / n_first_places,
                    0.
                ),
                rewards
            )
            rewards = rewards / MAX_NUM_STEPS
        elif self.reward_type == RewardType.EVERY_STEP_LENGTH:
            goose_lengths = np.array([[len(goose) for goose in env.steps[-1][0]['observation']['geese']]
                                      for env in self.wrapped_envs])
            epsilon = 1e-10
            goose_lengths_relative = goose_lengths / (goose_lengths.sum(axis=1, keepdims=True) + epsilon)
            # For the episodes that aren't finished, a reward of 1 is divided among the remaining players, based on size
            rewards = np.where(
                ~self.agent_dones,
                goose_lengths_relative,
                0.
            )
            # Awards 1st place all remaining points
            # There is a tactical edge case where 2nd should get more reward than 3rd, but does not
            n_steps_remaining = [MAX_NUM_STEPS - env.steps[-1][0]['observation']['step'] for env in self.wrapped_envs]
            n_steps_remaining = np.array(n_steps_remaining, dtype=np.float32)[:, np.newaxis]
            assert np.all(n_steps_remaining > 0)
            n_first_places = (current_standings == current_standings.max(axis=1, keepdims=True)).sum(axis=1,
                                                                                                     keepdims=True)
            rewards = np.where(
                self.agent_dones.all(axis=1, keepdims=True),
                np.where(
                    current_standings == current_standings.max(axis=1, keepdims=True),
                    n_steps_remaining / n_first_places,
                    0.
                ),
                rewards
            )
            rewards = rewards / MAX_NUM_STEPS
        elif self.reward_type == RewardType.ON_EAT_AND_ON_DEATH:
            # A small reward is given to a goose when it eats
            new_kaggle_rewards = np.array([[agent['reward'] for agent in env.steps[-1]] for env in self.wrapped_envs])
            goose_ate = np.logical_or(
                new_kaggle_rewards > (self.kaggle_rewards + GOOSE_MAX_LEN + 1.),
                np.logical_and(
                    new_kaggle_rewards == (self.kaggle_rewards + GOOSE_MAX_LEN + 1.),
                    (self.kaggle_timesteps[:, np.newaxis] + 1) % HUNGER_RATE == 0
                )
            )
            # Rewards start at 0, not 101, so this np.where catches that case
            goose_ate = np.where(
                self.kaggle_timesteps[:, np.newaxis] == 0,
                new_kaggle_rewards > 2 * (GOOSE_MAX_LEN + 1.) + 1.,
                goose_ate
            )
            rewards = goose_ate / (N_ROWS * N_COLS)
            # A reward of -1 is assigned to an agent on death, unless they win, in which case they receive 0
            agent_rankings = stats.rankdata(current_standings, method='max', axis=1)
            rewards = rewards + np.where(
                np.logical_and(np.logical_xor(agent_already_dones, self.agent_dones), agent_rankings < self.n_players),
                -1.,
                0.
            )
        elif self.reward_type == RewardType.RANK_ON_DEATH:
            agent_rankings = stats.rankdata(current_standings, method='average', axis=1)
            # Rescale rankings from 1 to n_players to lie between -1 to 1
            agent_rankings = 2. * (agent_rankings - 1.) / (self.n_players - 1.) - 1.
            rewards = np.where(
                np.logical_xor(agent_already_dones, self.agent_dones),
                agent_rankings,
                0.
            )
        else:
            raise ValueError(f'Unsupported reward_type: {self.reward_type}')

        agent_dones_cached = self.agent_dones.copy()
        if self.episodes_done.any() and self.silent_reset:
            self._soft_reset()

        self._update_other_info()
        return self.obs, rewards, agent_dones_cached, self.info_dict

    @property
    def state(self) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Alias for self.obs
        """
        return self.obs

    @property
    def obs(self) -> Union[np.ndarray, List[np.ndarray]]:
        if self.multi_obs_type:
            return [self._get_obs(ot) for ot in self.obs_type]
        else:
            return self._get_obs(self.obs_type)

    def _get_obs(self, obs_type: ObsType) -> np.ndarray:
        return np.concatenate([create_obs_tensor(env.steps[-1], obs_type) for env in self.wrapped_envs], axis=0)

    def _update_other_info(self):
        self.goose_head_locs = np.zeros((self.n_envs, self.n_players), dtype=np.int64) - 1
        for env_idx, env in enumerate(self.wrapped_envs):
            for agent_idx, goose_loc_list in enumerate(env.steps[-1][0]['observation']['geese']):
                if len(goose_loc_list) > 0:
                    self.goose_head_locs[env_idx, agent_idx] = goose_loc_list[0]
        self.available_actions_mask = np.stack([self.action_masking.get_action_mask(env.steps[-1])
                                                for env in self.wrapped_envs], axis=0)
        self.kaggle_rewards = np.array([[agent['reward'] for agent in env.steps[-1]] for env in self.wrapped_envs])
        self.kaggle_timesteps = np.array([env.steps[-1][0]['observation']['step'] for env in self.wrapped_envs])

    @property
    def info_dict(self) -> Dict:
        return {
            'head_locs': self.goose_head_locs,
            'available_actions_mask': self.available_actions_mask,
            'episodes_finished_last_turn': self.episodes_finished_last_turn,
            'episodes_finished_last_turn_info': self.episodes_finished_last_turn_info,
        }

    @property
    def episodes_done(self):
        return self.agent_dones.all(axis=1)


def create_obs_tensor(observation, obs_type):
    n_players = len(observation[0]['observation']['geese'])
    if obs_type == ObsType.COMBINED_GRADIENT_OBS:
        """
        Returns a tensor of shape (1, 3 + 2*n_players, 7, 11)
        The channels contain the following information about each cell of the board:
        for i in range (n_players):
            * contains_head[i]
            * contains_body[i], where the value of this cell represents how close to the tail this cell is
                Values of 1 / GOOSE_MAX_LEN represent the tail, and n / GOOSE_MAX_LEN represents the nth element
                counting from the tail, and INCLUDING the head
        * contains_food
        * steps_since_starvation (normalized to be in the range 0-1)
        * current_step (normalized to be in the range 0-1)
        """
        player_channel_list = [
            'contains_head',
            'contains_body',
        ]
        idx_dict = {c: i for i, c in enumerate(player_channel_list)}
        player_channels = np.zeros((len(player_channel_list) * n_players,
                                    N_ROWS,
                                    N_COLS),
                                   dtype=np.float32)
        contains_food = np.zeros((1, N_ROWS, N_COLS), dtype=np.float32)
        steps_since_starvation = np.zeros_like(contains_food)
        current_step = np.zeros_like(contains_food)
        for food_loc_n in observation[0]['observation']['food']:
            food_loc = row_col(food_loc_n)
            contains_food[:, food_loc[0], food_loc[1]] = 1.
        steps_since_starvation[:] = observation[0]['observation']['step'] % HUNGER_RATE
        current_step[:] = observation[0]['observation']['step']
        for agent_idx in range(n_players):
            channel_idx_base = agent_idx * len(player_channel_list)
            goose_loc_list = observation[0]['observation']['geese'][agent_idx]
            # Make sure the goose is still alive
            if len(goose_loc_list) > 0:
                head_loc = row_col(goose_loc_list[0])
                player_channels[channel_idx_base + idx_dict['contains_head'],
                                (*head_loc)] = 1.
            # Check if the goose is more than just a head
            if len(goose_loc_list) > 1:
                for i, body_loc_n in enumerate(goose_loc_list[::-1]):
                    body_loc = row_col(body_loc_n)
                    player_channels[channel_idx_base + idx_dict['contains_body'],
                                    (*body_loc)] = (i + 1.) / GOOSE_MAX_LEN
        obs = np.concatenate([
            player_channels,
            contains_food,
            steps_since_starvation / HUNGER_RATE,
            current_step / MAX_NUM_STEPS
        ], axis=0)
    elif obs_type == ObsType.COMBINED_OBS_SMALL:
        """
        Returns a tensor of shape (1, 3 + 3*n_players, 7, 11)
        The channels contain the following information about each cell of the board:
        for i in range (n_players):
            * contains_head[i]
            * contains_body[i]
            * contains_tail[i]
        * contains_food
        * steps_since_starvation (normalized to be in the range 0-1)
        * current_step (normalized to be in the range 0-1)
        """
        player_channel_list = [
            'contains_head',
            'contains_body',
            'contains_tail',
        ]
        idx_dict = {c: i for i, c in enumerate(player_channel_list)}
        player_channels = np.zeros((len(player_channel_list) * n_players,
                                    N_ROWS,
                                    N_COLS),
                                   dtype=np.float32)
        contains_food = np.zeros((1, N_ROWS, N_COLS), dtype=np.float32)
        steps_since_starvation = np.zeros_like(contains_food)
        current_step = np.zeros_like(contains_food)
        for food_loc_n in observation[0]['observation']['food']:
            food_loc = row_col(food_loc_n)
            contains_food[:, food_loc[0], food_loc[1]] = 1.
        steps_since_starvation[:] = observation[0]['observation']['step'] % HUNGER_RATE
        current_step[:] = observation[0]['observation']['step']
        for agent_idx in range(n_players):
            channel_idx_base = agent_idx * len(player_channel_list)
            goose_loc_list = observation[0]['observation']['geese'][agent_idx]
            # Make sure the goose is still alive
            if len(goose_loc_list) > 0:
                head_loc = row_col(goose_loc_list[0])
                player_channels[channel_idx_base + idx_dict['contains_head'],
                                (*head_loc)] = 1.
            # Check if the goose has a tail
            if len(goose_loc_list) > 1:
                tail_loc = row_col(goose_loc_list[-1])
                player_channels[channel_idx_base + idx_dict['contains_tail'],
                                (*tail_loc)] = 1.
            # Check if the goose is more than just a head and tail
            if len(goose_loc_list) > 2:
                for body_loc_n in goose_loc_list[1:-1]:
                    body_loc = row_col(body_loc_n)
                    player_channels[channel_idx_base + idx_dict['contains_body'],
                                    (*body_loc)] = 1.
        obs = np.concatenate([
            player_channels,
            contains_food,
            steps_since_starvation / HUNGER_RATE,
            current_step / MAX_NUM_STEPS
        ], axis=0)
    elif obs_type == ObsType.HEAD_CENTERED_OBS_LARGE:
        """
        Returns a tensor of shape (1, n_players, 3 + 11*n_players, 7, 11)
        Axes represent (1, n_players, n_channels, n_rows, n_cols)
        The channels contain the following information about each cell of the board:
        for i in range (n_players): (myself always first)
            * contains_head[i]
            * goose_length[i] (normalized to be in the range 0-1)
            * last_action_n[i]
            * last_action_e[i]
            * last_action_s[i]
            * last_action_w[i]
            * contains_body[i]
            * connected_n[i] (only active when contains_head or contains_body)
            * connected_e[i] (only active when contains_head or contains_body)
            * connected_s[i] (only active when contains_head or contains_body)
            * connected_w[i] (only active when contains_head or contains_body)
        * contains_food
        * steps_since_starvation (normalized to be in the range 0-1)
        * current_step (normalized to be in the range 0-1)
        """
        player_channel_list = [
            'contains_head',
            'goose_length',
            'last_action_n',
            'last_action_e',
            'last_action_s',
            'last_action_w',
            'contains_body',
            'connected_n',
            'connected_e',
            'connected_s',
            'connected_w',
        ]
        idx_dict = {c: i for i, c in enumerate(player_channel_list)}
        last_action_dict = {
            Action.NORTH.name: 'last_action_n',
            Action.EAST.name: 'last_action_e',
            Action.SOUTH.name: 'last_action_s',
            Action.WEST.name: 'last_action_w'
        }
        connected_dict = {
            Action.NORTH.name: 'connected_n',
            Action.EAST.name: 'connected_e',
            Action.SOUTH.name: 'connected_s',
            Action.WEST.name: 'connected_w'
        }
        player_channels = np.zeros((n_players,
                                    len(player_channel_list) * n_players,
                                    N_ROWS,
                                    N_COLS),
                                   dtype=np.float32)
        contains_food = np.zeros((n_players, 1, N_ROWS, N_COLS), dtype=np.float32)
        steps_since_starvation = np.zeros_like(contains_food)
        current_step = np.zeros_like(contains_food)

        for food_loc_n in observation[0]['observation']['food']:
            food_loc = row_col(food_loc_n)
            contains_food[:, :, food_loc[0], food_loc[1]] = 1.
        steps_since_starvation[:] = observation[0]['observation']['step'] % HUNGER_RATE
        current_step[:] = observation[0]['observation']['step']
        for main_agent_idx in range(n_players):
            for agent_idx in range(n_players):
                if agent_idx == main_agent_idx:
                    channel_idx_base = 0
                elif agent_idx < main_agent_idx:
                    channel_idx_base = (agent_idx + 1) * len(player_channel_list)
                else:
                    channel_idx_base = agent_idx * len(player_channel_list)
                goose_loc_list = observation[0]['observation']['geese'][agent_idx]
                # Make sure the goose is still alive
                if len(goose_loc_list) > 0:
                    head_loc = row_col(goose_loc_list[0])
                    player_channels[main_agent_idx,
                                    channel_idx_base + idx_dict['contains_head'],
                                    (*head_loc)] = 1.
                    goose_len = float(len(goose_loc_list))
                    player_channels[main_agent_idx,
                                    channel_idx_base + idx_dict['goose_length']] = goose_len / (N_ROWS * N_COLS)
                    if observation[0]['observation']['step'] > 0:
                        last_action_channel = last_action_dict[observation[agent_idx]['action']]
                        player_channels[main_agent_idx,
                                        channel_idx_base + idx_dict[last_action_channel]] = 1.
                # Make sure the goose is more than just a head
                if len(goose_loc_list) > 1:
                    for body_loc_n in goose_loc_list[1:]:
                        body_loc = row_col(body_loc_n)
                        player_channels[main_agent_idx,
                                        channel_idx_base + idx_dict['contains_body'],
                                        (*body_loc)] = 1.
                    for i in range(len(goose_loc_list) - 1):
                        connection_channel = connected_dict[_get_direction(
                            goose_loc_list[i],
                            goose_loc_list[i + 1]
                        )]
                        body_loc = row_col(goose_loc_list[i])
                        player_channels[main_agent_idx,
                                        channel_idx_base + idx_dict[connection_channel],
                                        (*body_loc)] = 1.
                        next_connection_channel = connected_dict[_get_direction(
                            goose_loc_list[i + 1],
                            goose_loc_list[i]
                        )]
                        next_body_loc = row_col(goose_loc_list[i + 1])
                        player_channels[main_agent_idx,
                                        channel_idx_base + idx_dict[next_connection_channel],
                                        (*next_body_loc)] = 1.
        obs = np.concatenate([
            player_channels,
            contains_food,
            steps_since_starvation / HUNGER_RATE,
            current_step / MAX_NUM_STEPS
        ], axis=1)
        for centered_agent_idx, goose_loc_list in enumerate(observation[0]['observation']['geese']):
            if len(goose_loc_list) > 0:
                head_loc = np.array(row_col(goose_loc_list[0]))
                translation = np.array([int((N_ROWS - 1) / 2), int((N_COLS - 1) / 2)]) - head_loc
                obs[centered_agent_idx] = np.roll(
                    obs[centered_agent_idx], translation, axis=(-2, -1)
                )
                if obs[centered_agent_idx,
                       0,
                       int((N_ROWS - 1) / 2),
                       int((N_COLS - 1) / 2)] != 1.:
                    print('Head not centered!')
                    print(head_loc)
                    print(obs[centered_agent_idx, 0])
                    assert False
    elif obs_type == ObsType.HEAD_CENTERED_OBS_SMALL:
        """
        Returns a tensor of shape (1, n_players, 3 + 3*n_players, 7, 11)
        The channels contain the following information about each cell of the board:
        for i in range (n_players): (myself always first)
            * contains_head[i]
            * contains_body[i]
            * contains_tail[i]
        * contains_food
        * steps_since_starvation (normalized to be in the range 0-1)
        * current_step (normalized to be in the range 0-1)
        """
        player_channel_list = [
            'contains_head',
            'contains_body',
            'contains_tail',
        ]
        idx_dict = {c: i for i, c in enumerate(player_channel_list)}
        player_channels = np.zeros((n_players,
                                    len(player_channel_list) * n_players,
                                    N_ROWS,
                                    N_COLS),
                                   dtype=np.float32)
        contains_food = np.zeros((n_players, 1, N_ROWS, N_COLS), dtype=np.float32)
        steps_since_starvation = np.zeros_like(contains_food)
        current_step = np.zeros_like(contains_food)
        for food_loc_n in observation[0]['observation']['food']:
            food_loc = row_col(food_loc_n)
            contains_food[:, :, food_loc[0], food_loc[1]] = 1.
        steps_since_starvation[:] = observation[0]['observation']['step'] % HUNGER_RATE
        current_step[:] = observation[0]['observation']['step']
        for main_agent_idx in range(n_players):
            for agent_idx in range(n_players):
                if agent_idx == main_agent_idx:
                    channel_idx_base = 0
                elif agent_idx < main_agent_idx:
                    channel_idx_base = (agent_idx + 1) * len(player_channel_list)
                else:
                    channel_idx_base = agent_idx * len(player_channel_list)
                goose_loc_list = observation[0]['observation']['geese'][agent_idx]
                # Make sure the goose is still alive
                if len(goose_loc_list) > 0:
                    head_loc = row_col(goose_loc_list[0])
                    player_channels[main_agent_idx,
                                    channel_idx_base + idx_dict['contains_head'],
                                    (*head_loc)] = 1.
                # Check if the goose has a tail
                if len(goose_loc_list) > 1:
                    tail_loc = row_col(goose_loc_list[-1])
                    player_channels[main_agent_idx,
                                    channel_idx_base + idx_dict['contains_tail'],
                                    (*tail_loc)] = 1.
                # Check if the goose is more than just a head and tail
                if len(goose_loc_list) > 2:
                    for body_loc_n in goose_loc_list[1:-1]:
                        body_loc = row_col(body_loc_n)
                        player_channels[main_agent_idx,
                                        channel_idx_base + idx_dict['contains_body'],
                                        (*body_loc)] = 1.
        obs = np.concatenate([
            player_channels,
            contains_food,
            steps_since_starvation / HUNGER_RATE,
            current_step / MAX_NUM_STEPS
        ], axis=1)
        for centered_agent_idx, goose_loc_list in enumerate(observation[0]['observation']['geese']):
            if len(goose_loc_list) > 0:
                head_loc = np.array(row_col(goose_loc_list[0]))
                translation = np.array([int((N_ROWS - 1) / 2), int((N_COLS - 1) / 2)]) - head_loc
                obs[centered_agent_idx] = np.roll(
                    obs[centered_agent_idx], translation, axis=(-2, -1)
                )
                if obs[centered_agent_idx,
                       0,
                       int((N_ROWS - 1) / 2),
                       int((N_COLS - 1) / 2)] != 1.:
                    print('Head not centered!')
                    print(head_loc)
                    print(obs[centered_agent_idx, 0])
                    assert False
    elif obs_type == ObsType.HEAD_CENTERED_SET_OBS:
        """
        Returns a tensor of shape (1, n_players, 13, 7, 11)
        Axes represent (1, n_players, n_channels, n_rows, n_cols)
        The channels are (mostly) one-hot matrices and contain the following information about each cell of the board:
        The selected agent's values (except for food, steps_since_starvation, and current_step) are positive
        Opponent values are negative
        * contains_head
        * contains_food
        * contains_body
        * connected_n (only active when contains_head or contains_body)
        * connected_e (only active when contains_head or contains_body)
        * connected_s (only active when contains_head or contains_body)
        * connected_w (only active when contains_head or contains_body)
        * last_action_n (only active when contains_head == True)
        * last_action_e (only active when contains_head == True)
        * last_action_s (only active when contains_head == True)
        * last_action_w (only active when contains_head == True)
        * steps_since_starvation (normalized to be in the range 0-1)
        * current_step (normalized to be in the range 0-1)
        """
        contains_head = np.zeros((n_players, N_ROWS, N_COLS), dtype=np.float32)
        contains_food = np.zeros_like(contains_head)
        contains_body = np.zeros_like(contains_head)
        connected_n = np.zeros_like(contains_head)
        connected_e = np.zeros_like(contains_head)
        connected_s = np.zeros_like(contains_head)
        connected_w = np.zeros_like(contains_head)
        last_action_n = np.zeros_like(contains_head)
        last_action_e = np.zeros_like(contains_head)
        last_action_s = np.zeros_like(contains_head)
        last_action_w = np.zeros_like(contains_head)
        steps_since_starvation = np.zeros_like(contains_head)
        current_step = np.zeros_like(contains_head)

        connected_dict = {
            Action.NORTH.name: connected_n,
            Action.EAST.name: connected_e,
            Action.SOUTH.name: connected_s,
            Action.WEST.name: connected_w
        }
        last_action_dict = {
            Action.NORTH.name: last_action_n,
            Action.EAST.name: last_action_e,
            Action.SOUTH.name: last_action_s,
            Action.WEST.name: last_action_w
        }
        for food_loc_n in observation[0]['observation']['food']:
            food_loc = row_col(food_loc_n)
            contains_food[:, food_loc[0], food_loc[1]] = 1.
        steps_since_starvation[:] = observation[0]['observation']['step'] % HUNGER_RATE
        current_step[:] = observation[0]['observation']['step']
        for centered_agent_idx in range(n_players):
            for agent_idx in range(n_players):
                is_centered_multiplier = 1. if centered_agent_idx == agent_idx else -1.
                goose_loc_list = observation[0]['observation']['geese'][agent_idx]
                # Make sure the goose is still alive
                if len(goose_loc_list) > 0:
                    head_loc = row_col(goose_loc_list[0])
                    contains_head[centered_agent_idx, (*head_loc)] = 1. * is_centered_multiplier
                    if observation[0]['observation']['step'] > 0:
                        last_action_channel = last_action_dict[observation[agent_idx]['action']]
                        last_action_channel[centered_agent_idx, (*head_loc)] = 1. * is_centered_multiplier
                # Make sure the goose is more than just a head
                if len(goose_loc_list) > 1:
                    for body_loc_n in goose_loc_list[1:]:
                        body_loc = row_col(body_loc_n)
                        contains_body[centered_agent_idx, (*body_loc)] = 1. * is_centered_multiplier
                    for i in range(len(goose_loc_list) - 1):
                        connection_channel = connected_dict[_get_direction(
                            goose_loc_list[i],
                            goose_loc_list[i + 1]
                        )]
                        body_loc = row_col(goose_loc_list[i])
                        connection_channel[centered_agent_idx, (*body_loc)] = 1. * is_centered_multiplier
                        next_connection_channel = connected_dict[_get_direction(
                            goose_loc_list[i + 1],
                            goose_loc_list[i]
                        )]
                        next_body_loc = row_col(goose_loc_list[i + 1])
                        next_connection_channel[centered_agent_idx, (*next_body_loc)] = 1. * is_centered_multiplier
        obs = np.stack([
            contains_head,
            contains_food,
            contains_body,
            connected_n,
            connected_e,
            connected_s,
            connected_w,
            last_action_n,
            last_action_e,
            last_action_s,
            last_action_w,
            steps_since_starvation / HUNGER_RATE,
            current_step / MAX_NUM_STEPS
        ], axis=1)
        for centered_agent_idx, goose_loc_list in enumerate(observation[0]['observation']['geese']):
            if len(goose_loc_list) > 0:
                head_loc = np.array(row_col(goose_loc_list[0]))
                translation = np.array([3, 5]) - head_loc
                obs[centered_agent_idx] = np.roll(
                    obs[centered_agent_idx], translation, axis=(-2, -1)
                )
                if obs[centered_agent_idx, 0, 3, 5] != 1.:
                    print('Head not centered!')
                    print(head_loc)
                    print(obs[centered_agent_idx, 1])
                    assert False
    elif obs_type == ObsType.SET_OBS:
        """
        Returns a tensor of shape (1, 13, 7, 11)
        Axes represent (1, n_channels, n_rows, n_cols)
        The channels are (mostly) one-hot matrices and contain the following information about each cell of the board:
        * contains_head
        * contains_food
        * contains_body
        * connected_n (only active when contains_head or contains_body)
        * connected_e (only active when contains_head or contains_body)
        * connected_s (only active when contains_head or contains_body)
        * connected_w (only active when contains_head or contains_body)
        * last_action_n (only active when contains_head == True)
        * last_action_e (only active when contains_head == True)
        * last_action_s (only active when contains_head == True)
        * last_action_w (only active when contains_head == True)
        * steps_since_starvation (normalized to be in the range 0-1)
        * current_step (normalized to be in the range 0-1)
        """
        contains_head = np.zeros((N_ROWS, N_COLS), dtype=np.float32)
        contains_food = np.zeros_like(contains_head)
        contains_body = np.zeros_like(contains_head)
        connected_n = np.zeros_like(contains_head)
        connected_e = np.zeros_like(contains_head)
        connected_s = np.zeros_like(contains_head)
        connected_w = np.zeros_like(contains_head)
        last_action_n = np.zeros_like(contains_head)
        last_action_e = np.zeros_like(contains_head)
        last_action_s = np.zeros_like(contains_head)
        last_action_w = np.zeros_like(contains_head)
        steps_since_starvation = np.zeros_like(contains_head)
        current_step = np.zeros_like(contains_head)

        connected_dict = {
            Action.NORTH.name: connected_n,
            Action.EAST.name: connected_e,
            Action.SOUTH.name: connected_s,
            Action.WEST.name: connected_w
        }
        last_action_dict = {
            Action.NORTH.name: last_action_n,
            Action.EAST.name: last_action_e,
            Action.SOUTH.name: last_action_s,
            Action.WEST.name: last_action_w
        }
        for food_loc_n in observation[0]['observation']['food']:
            food_loc = row_col(food_loc_n)
            contains_food[food_loc[0], food_loc[1]] = 1.
        for agent_idx in range(n_players):
            goose_loc_list = observation[0]['observation']['geese'][agent_idx]
            # Make sure the goose is still alive
            if len(goose_loc_list) > 0:
                head_loc = row_col(goose_loc_list[0])
                contains_head[head_loc[0], head_loc[1]] = 1.
                if observation[0]['observation']['step'] > 0:
                    last_action_channel = last_action_dict[observation[agent_idx]['action']]
                    last_action_channel[head_loc[0], head_loc[1]] = 1.
            # Make sure the goose is more than just a head
            if len(goose_loc_list) > 1:
                for body_loc_n in goose_loc_list[1:]:
                    body_loc = row_col(body_loc_n)
                    contains_body[body_loc[0], body_loc[1]] = 1.
                for i in range(len(goose_loc_list) - 1):
                    connection_channel = connected_dict[_get_direction(
                        goose_loc_list[i],
                        goose_loc_list[i + 1]
                    )]
                    body_loc = row_col(goose_loc_list[i])
                    connection_channel[body_loc[0], body_loc[1]] = 1.
                    next_connection_channel = connected_dict[_get_direction(
                        goose_loc_list[i + 1],
                        goose_loc_list[i]
                    )]
                    next_body_loc = row_col(goose_loc_list[i + 1])
                    next_connection_channel[next_body_loc[0], next_body_loc[1]] = 1.
        steps_since_starvation[:] = observation[0]['observation']['step'] % HUNGER_RATE
        current_step[:] = observation[0]['observation']['step']
        obs = np.stack([
            contains_head,
            contains_food,
            contains_body,
            connected_n,
            connected_e,
            connected_s,
            connected_w,
            last_action_n,
            last_action_e,
            last_action_s,
            last_action_w,
            steps_since_starvation / HUNGER_RATE,
            current_step / MAX_NUM_STEPS
        ], axis=0)
    else:
        raise ValueError(f'Unsupported obs_type: {obs_type}')
    return np.expand_dims(obs, 0)


ACTIONS_TUPLE = tuple(Action)


class LightweightEnv:
    """
    Mostly the same as Kaggle hungry_geese env, but faster to work with and copy
    """
    def __init__(self, configuration: Configuration, debug: bool = False):
        self.configuration = configuration
        self.n_rows = configuration.rows
        self.n_cols = configuration.columns
        self.debug = debug

        self.agent_count = None
        self.geese = None
        self.food = None
        self.last_actions = None
        self.step_counter = None
        self.rewards = None
        self.steps = None

        self.reset()

    def reset(self, num_agents: int = N_PLAYERS) -> List[Dict]:
        self.agent_count = num_agents
        heads = sample(range(self.n_cols * self.n_rows), self.agent_count)
        self.geese = [[head] for head in heads]
        food_candidates = set(range(self.n_cols * self.n_rows)).difference(heads)
        # Ensure we only place as many food as there are open squares
        min_food = min(self.configuration.min_food, len(food_candidates))
        self.food = sample(food_candidates, min_food)
        self.last_actions = [Action.NORTH for _ in range(self.agent_count)]
        self.step_counter = 0
        self.rewards = [0 for _ in range(self.agent_count)]
        self.steps = []
        self.generate_and_append_next_state()

        return self.steps[-1]

    def step(self, actions: Union[List[str], Sequence[int], np.ndarray]):
        assert not self.done

        if type(actions) == np.ndarray:
            actions = actions.ravel()
        assert len(actions) == self.agent_count, f'Got {len(actions)} actions for {self.agent_count} agents'
        if type(actions[0]) != str:
            actions = [ACTIONS_TUPLE[i].name for i in actions]

        for index, goose in enumerate(self.geese):
            if len(goose) == 0:
                continue
            action = Action[actions[index]]

            # Check action direction on any step after the first
            if self.last_actions[index] == action.opposite() and self.step_counter > 0:
                self.debug_print(f'Opposite action: {index, action, self.last_actions[index]}')
                self.geese[index] = []
                continue
            self.last_actions[index] = action

            head = translate(goose[0], action, self.n_cols, self.n_rows)

            # Consume food or drop a tail piece
            if head in self.food:
                self.food.remove(head)
            else:
                goose.pop()

            # Self collision
            if head in goose:
                self.debug_print(f'Body Hit: {index, action, head, goose}')
                self.geese[index] = []
                continue

            while len(goose) >= self.configuration.max_length:
                # Free a spot for the new head if needed
                goose.pop()
            # Add New Head to the Goose
            goose.insert(0, head)

            # If hunger strikes, remove from the tail
            if self.step_counter % self.configuration.hunger_rate == 0 and self.step_counter > 0:
                if len(goose) > 0:
                    goose.pop()
                if len(goose) == 0:
                    self.debug_print(f'Goose Starved: {index}')
                    continue

        goose_positions = histogram(
            position
            for goose in self.geese
            for position in goose
        )

        # Check for collisions
        for index, goose in enumerate(self.geese):
            if len(goose) > 0:
                head = self.geese[index][0]
                if goose_positions[head] > 1:
                    self.debug_print(f'Goose Collision: {index, self.last_actions[index]}')
                    self.geese[index] = []

        # Add food if min_food threshold reached
        needed_food = self.configuration.min_food - len(self.food)
        if needed_food > 0:
            collisions = {
                position
                for goose in self.geese
                for position in goose
            }
            available_positions = set(range(self.n_rows * self.n_cols)).difference(collisions).difference(self.food)
            # Ensure we don't sample more food than available positions
            needed_food = min(needed_food, len(available_positions))
            self.food.extend(sample(available_positions, needed_food))

        self.step_counter += 1
        # Set rewards after deleting all geese to ensure that geese don't receive a reward on the turn they perish
        for index, goose in enumerate(self.geese):
            if len(goose) > 0:
                # Adding 1 to len(env.steps) ensures that if an agent gets reward 4507, it died on turn 45 with length 7
                self.rewards[index] = (self.step_counter + 1) * (self.configuration.max_length + 1) + len(goose)

        self.generate_and_append_next_state()
        return self.steps[-1]

    def generate_and_append_next_state(self) -> NoReturn:
        state_dict_list = []
        statuses = self.get_statuses()
        for i in range(self.agent_count):
            dict_i = {
                'action': self.last_actions[i].name,
                'reward': self.rewards[i],
                'info': {},
                'observation': {
                    # 'remainingOverageTime' is not computed and is included only for compatibility
                    'index': i
                },
                'status': statuses[i]
            }
            if i == 0:
                dict_i['observation'].update({
                    'step': self.step_counter,
                    'geese': [[g for g in goose] for goose in self.geese],
                    'food': [f for f in self.food]
                })
            state_dict_list.append(dict_i)

        self.steps.append(state_dict_list)

    @property
    def state(self) -> List[Dict]:
        return self.steps[-1]

    @property
    def done(self) -> bool:
        n_geese_alive = len([True for goose in self.geese if len(goose) > 0])
        return n_geese_alive <= 1 or self.step_counter >= self.configuration.episode_steps - 1

    def get_statuses(self) -> List[str]:
        if self.done:
            return ['DONE' for _ in range(self.agent_count)]
        else:
            return ['ACTIVE' if len(goose) > 0 else 'DONE' for goose in self.geese]

    def lightweight_clone(self):
        cloned_env = LightweightEnv(self.configuration)

        cloned_env.agent_count = self.agent_count
        cloned_env.geese = [[g for g in goose] for goose in self.geese]
        cloned_env.food = [f for f in self.food]
        cloned_env.last_actions = [a for a in self.last_actions]
        cloned_env.step_counter = self.step_counter
        cloned_env.rewards = [r for r in self.rewards]
        cloned_env.steps = [None] * (len(self.steps) - 1)
        cloned_env.steps.append(copy.deepcopy(self.steps[-1]))

        return cloned_env

    def canonical_string_repr(self, include_food=True) -> str:
        if self.done:
            raise RuntimeError('Environment has finished')
        else:
            canonical_string = ''
            canonical_string += f'S: {self.step_counter} '
            if include_food:
                canonical_string += 'F: ' + '_'.join([str(f) for f in self.food]) + ' '
            canonical_string += 'G: '
            for index, goose, status in zip(range(self.agent_count), self.geese, self.get_statuses()):
                canonical_string += f'{index}_'
                if status == 'DONE':
                    canonical_string += f'D '
                else:
                    canonical_string += '_'.join([str(g) for g in goose]) + f'_{self.last_actions[index].value} '

            return canonical_string

    def debug_print(self, out: str):
        if self.debug:
            print(out)

    def render_ansi(self) -> str:
        food_symbol = "F"
        column_divider = "|"
        row_divider = "+" + "+".join(["---"] * self.n_cols) + "+\n"

        board = [" "] * (self.n_rows * self.n_cols)
        for pos in self.food:
            board[pos] = food_symbol

        for index, goose in enumerate(self.geese):
            for position in goose:
                board[position] = str(index)

        out = row_divider
        for row in range(self.n_rows):
            for col in range(self.n_cols):
                out += column_divider + f" {board[(row * self.n_cols) + col]} "
            out += column_divider + "\n" + row_divider

        return out


def make(environment: str = 'hungry_geese', debug: bool = False, **kwargs) -> LightweightEnv:
    assert environment == 'hungry_geese'
    config = Configuration(kaggle_make(environment, debug=debug, **kwargs).configuration)
    return LightweightEnv(config, debug=debug)


def make_from_state(state: Observation,
                    last_actions: List[Action],
                    configuration: Optional[Configuration] = None,
                    *args, **kwargs) -> LightweightEnv:
    if configuration is None:
        configuration = kaggle_make('hungry_geese').configuration
    configuration = Configuration(configuration)
    env = LightweightEnv(configuration, *args, **kwargs)

    env.agent_count = len(state.geese)
    env.geese = copy.deepcopy(state.geese)
    env.food = copy.copy(state.food)
    env.last_actions = copy.copy(last_actions)
    env.step_counter = state.step

    rewards = [0.] * len(state.geese)
    for index, goose in enumerate(state.geese):
        if len(goose) > 0:
            # Adding 1 to len(env.steps) ensures that if an agent gets reward 4507, it died on turn 45 with length 7
            rewards[index] = (state.step + 1) * (configuration.max_length + 1) + len(goose)
    env.rewards = rewards

    env.steps = [None] * state.step
    env.generate_and_append_next_state()

    return env


class Node:
    def __init__(
            self,
            geese_still_playing_mask: Sequence[bool],
            available_actions_masks: Optional[np.ndarray],
            initial_policies: np.ndarray,
            initial_values: np.ndarray
    ):
        self.geese_still_playing = np.array(geese_still_playing_mask)
        self.n_geese = len(self.geese_still_playing)
        assert self.geese_still_playing.any()
        if available_actions_masks is None:
            self.available_actions_masks = np.ones_like(initial_policies)
        else:
            self.available_actions_masks = np.where(
                available_actions_masks.any(axis=-1, keepdims=True),
                available_actions_masks,
                np.ones_like(available_actions_masks)
            )
        self.initial_policies = initial_policies
        assert self.initial_policies.shape == (self.n_geese, 4)
        assert np.all(0. <= self.initial_policies) and np.all(self.initial_policies <= 1.)
        assert np.allclose(self.initial_policies.sum(axis=-1), 1., atol=1e-2)
        # Re-normalize policy distribution
        self.initial_policies = self.initial_policies * self.available_actions_masks
        if np.any(self.initial_policies.sum(axis=1) == 0.):
            if np.logical_and(self.initial_policies.sum(axis=1) == 0.,
                              self.available_actions_masks.any(axis=1)).any():
                print('WARNING: All available actions have 0% probability')
            self.initial_policies = np.where(
                self.initial_policies.sum(axis=1, keepdims=True) == 0.,
                0.25,
                self.initial_policies
            )
        self.initial_policies = self.initial_policies / self.initial_policies.sum(axis=1, keepdims=True)
        # Initial values are stored for logging purposes only
        if initial_values.ndim == 1:
            initial_values = initial_values[:, np.newaxis]
        if initial_values.shape != (self.n_geese, 1):
            raise RuntimeError(f'Initial_values should be of shape {(self.n_geese, 1)}, got {initial_values.shape}')
        self.initial_values = initial_values

        self.q_vals = np.zeros_like(self.initial_policies)
        self.n_visits = np.zeros_like(self.initial_policies)

    def update(self, actions: np.ndarray, values: np.ndarray) -> NoReturn:
        if actions.shape != (self.n_geese,):
            raise RuntimeError(f'Actions should be of shape {(self.n_geese,)}, got {actions.shape}')
        if values.ndim == 1:
            values = values[:, np.newaxis]
        if values.shape != (self.n_geese, 1):
            raise RuntimeError(f'Values should be of shape {(self.n_geese, 1)}, got {values.shape}')
        if not np.isclose(values.sum(), 0., atol=1e-2):
            raise RuntimeError(f'Values should sum to 0, got {values.ravel()} which sums to {values.sum()}')
        if (values.ravel()[self.geese_still_playing].min(initial=float('inf')) <=
                values.ravel()[~self.geese_still_playing].max(initial=float('-inf'))):
            raise RuntimeError(f'Values for dead geese should always be less than those for still living geese.\n'
                               f'Values:\n{values.ravel()}\n'
                               f'Geese still playing:\n{self.geese_still_playing}\n')

        selected_actions_mask = np.eye(4)[actions]
        self.q_vals = np.where(
            selected_actions_mask,
            (self.n_visits * self.q_vals + values) / (self.n_visits + 1.),
            self.q_vals
        )
        self.n_visits += selected_actions_mask

    def get_puct_actions(self, c_puct: float) -> np.ndarray:
        uct_vals = self.q_vals + (c_puct *
                                  self.initial_policies *
                                  np.sqrt(self.n_visits.sum(axis=1, keepdims=True)) / (1. + self.n_visits))
        uct_vals = np.where(
            self.available_actions_masks,
            uct_vals,
            -100.
        )
        """
        We could just go from here using a simple np.argmax(uct_vals, axis=1) to select actions,
        but this would always tiebreak towards actions with lower indices.
        Therefore, we sample from all actions whose uct value equals the max uct value for that agent.
        """
        uct_max = uct_vals.max(axis=1, keepdims=True)
        return np.where(
            self.geese_still_playing,
            rowwise_random_choice((uct_vals == uct_max).astype(np.float)),
            0
        )

    def get_improved_policies(self, temp: float = 1.) -> np.ndarray:
        assert temp >= 0.
        if temp == 0.:
            max_n_visits = self.n_visits.max(axis=1, keepdims=True)
            probs = np.where(
                self.n_visits == max_n_visits,
                1.,
                0.
            )
        else:
            probs = np.power(self.n_visits, 1. / temp)
        return probs / probs.sum(axis=1, keepdims=True)

    def get_improved_actions(self, temp: float = 1.) -> np.ndarray:
        probs = self.get_improved_policies(temp)
        return np.where(
            self.geese_still_playing,
            rowwise_random_choice(probs),
            0
        )

    def get_max_policy_actions(self) -> np.ndarray:
        return self.get_improved_actions(0.)

    def get_max_value_actions(self) -> np.ndarray:
        max_values = self.q_vals.max(axis=1, keepdims=True)
        probs = np.where(
            self.q_vals == max_values,
            1.,
            0.
        )
        probs /= probs.sum(axis=1, keepdims=True)
        return np.where(
            self.geese_still_playing,
            rowwise_random_choice(probs),
            0
        )

    @staticmethod
    def max_policy_actions(node: "Node") -> np.ndarray:
        return node.get_max_policy_actions()

    @staticmethod
    def max_value_actions(node: "Node") -> np.ndarray:
        return node.get_max_value_actions()


class BasicMCTS:
    def __init__(
            self,
            action_mask_func: Callable,
            actor_critic_func: Callable,
            terminal_value_func: Callable,
            c_puct: float = 1.,
            add_noise: bool = False,
            noise_val: float = 2.,
            noise_weight: float = 0.25,
            include_food: bool = True,
    ):
        self.action_mask_func = action_mask_func
        self.actor_critic_func = actor_critic_func
        self.terminal_value_func = terminal_value_func
        self.c_puct = c_puct
        self.add_noise = add_noise
        self.noise_val = noise_val
        self.noise_weight = noise_weight
        self.include_food = include_food
        self.nodes = {}

    def _search(
            self,
            env: LightweightEnv,
            add_noise: bool = False
    ) -> np.ndarray:
        """
        This function performs one iteration of MCTS. It is recursively called
        until a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.
        Once a leaf node is found, the actor_critic function is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path.

        @return: the values of the current state
        """
        if env.done:
            return self.terminal_value_func(env.state)

        s = env.canonical_string_repr(include_food=self.include_food)
        node = self.nodes.get(s, None)
        if node is None:
            # Leaf node
            full_state = env.state
            policy_est, value_est = self.actor_critic_func(full_state)
            if add_noise:
                noise = np.random.dirichlet(np.zeros(4) + self.noise_val, size=(len(env.geese),))
                policy_est = (1. - self.noise_weight) * policy_est + self.noise_weight * noise
            self.nodes[s] = Node(
                [status == 'ACTIVE' for status in env.get_statuses()],
                self.action_mask_func(full_state),
                policy_est,
                value_est
            )
            return value_est

        a = node.get_puct_actions(self.c_puct).ravel()
        env.step(a)
        v = self._search(env)
        node.update(a, v)
        return v

    def expand(
            self,
            env: LightweightEnv,
            trajectory: Tuple[Tuple[str, Optional[np.ndarray]], ...] = ()
    ) -> Tuple[
        List[Dict],
        Tuple[Tuple[str, Optional[Tuple[int]]], ...],
        bool,
        Optional[List[bool]],
        Optional[np.ndarray]
    ]:
        if env.done:
            return env.state, trajectory, True, None, None

        s = env.canonical_string_repr(include_food=self.include_food)
        node = self.nodes.get(s, None)
        if node is None:
            # Leaf node
            full_state = env.state
            still_alive = [status == 'ACTIVE' for status in env.get_statuses()]
            available_actions = self.action_mask_func(full_state)
            return full_state, trajectory + ((s, None),), False, still_alive, available_actions

        a = node.get_puct_actions(self.c_puct).ravel()
        env.step(a)
        return self.expand(env, trajectory + ((s, a),))

    def backpropagate(
            self,
            trajectory: Tuple[Tuple[str, Optional[Tuple[int]]], ...],
            policy_est: Optional[np.ndarray],
            value_est: np.ndarray,
            still_alive: List[bool],
            available_actions: np.ndarray
    ):
        for i, (s, a) in enumerate(trajectory):
            if a is not None:
                node = self.nodes[s]
                node.update(a, value_est)
            else:
                # Noise should only be added to the root node
                if self.add_noise and i == 0:
                    noise = np.random.dirichlet(np.zeros(4) + self.noise_val, size=(policy_est.shape[0],))
                    policy_est = (1. - self.noise_weight) * policy_est + self.noise_weight * noise
                self.nodes[s] = Node(
                    still_alive,
                    available_actions,
                    policy_est,
                    value_est
                )

    def run_mcts(
            self,
            env: LightweightEnv,
            n_iter: int,
            max_time: float = float('inf')
    ) -> Node:
        start_time = time.time()
        for _ in range(n_iter):
            if time.time() - start_time >= max_time:
                break
            self._search(env.lightweight_clone(), add_noise=self.add_noise)

        return self.get_root_node(env)

    def get_root_node(self, env: LightweightEnv) -> Node:
        return self.nodes[env.canonical_string_repr(include_food=self.include_food)]

    def reset(self) -> NoReturn:
        self.nodes = {}


class BasicConvolutionalBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            dilation: int = 1,
            n_layers: int = 2,
            normalize: bool = True,
            activation: Callable = nn.ReLU,
            downsample: nn.Module = nn.Identity()):
        super(BasicConvolutionalBlock, self).__init__()
        assert n_layers >= 1
        padding = (dilation * (kernel_size - 1.)) / 2.
        if padding == int(padding):
            padding = int(padding)
        else:
            raise ValueError(f'Padding must be an integer, but was {padding:0.2f}')
        if downsample is None:
            downsample = nn.Identity()

        layers = [nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            padding_mode='circular'
        )]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(activation(inplace=True))

        for i in range(n_layers - 1):
            layers.append(nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                padding_mode='circular'
            ))
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(activation(inplace=True))
        # Remove final activation layer - to be applied after residual connection
        layers = layers[:-1]
        layers.append(downsample)
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class InvertedResidualBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            expansion_factor: int = 6,
            dilation: int = 1,
            normalize: bool = True,
            activation: Callable = nn.ReLU,
            downsample: nn.Module = nn.Identity()):
        super(InvertedResidualBlock, self).__init__()
        padding = (dilation * (kernel_size - 1.)) / 2.
        if padding == int(padding):
            padding = int(padding)
        else:
            raise ValueError(f'Padding must be an integer, but was {padding:0.2f}')
        if downsample is None:
            self.downsample = nn.Identity()
        else:
            self.downsample = downsample
        mid_channels = in_channels * expansion_factor

        self.expansion = nn.Conv2d(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=1,
        )
        self.expansion_norm = nn.BatchNorm2d(mid_channels) if normalize else nn.Identity()
        self.expansion_act = activation(inplace=True)

        self.depthwise_conv = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=mid_channels,
            padding_mode='circular'
        )
        self.depthwise_conv_norm = nn.BatchNorm2d(mid_channels) if normalize else nn.Identity()
        self.depthwise_conv_act = activation(inplace=True)

        self.projection_conv = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
        )
        self.projection_norm = nn.BatchNorm2d(mid_channels) if normalize else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expanded = self.expansion_norm(self.expansion_act(self.expansion(x)))
        convolved = self.depthwise_conv_norm(self.depthwise_conv_act(self.depthwise_conv(expanded)))
        projected = self.projection_norm(self.projection_conv(convolved))
        return self.downsample(projected)


class SELayer(nn.Module):
    def __init__(self, n_channels, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(n_channels, n_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(n_channels // reduction, n_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualModel(nn.Module):
    def __init__(
            self,
            block_class: Union[BasicConvolutionalBlock, InvertedResidualBlock],
            conv_block_kwargs: Sequence[Dict],
            squeeze_excitation: bool = True,
    ):
        super(ResidualModel, self).__init__()
        assert len(conv_block_kwargs) >= 1
        self.conv_blocks = nn.ModuleList()
        self.change_n_channels = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        self.se_layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        for kwargs in conv_block_kwargs:
            self.conv_blocks.append(block_class(**kwargs))
            if kwargs['in_channels'] != kwargs['out_channels']:
                self.change_n_channels.append(nn.Conv2d(kwargs['in_channels'], kwargs['out_channels'], 1))
            else:
                self.change_n_channels.append(nn.Identity())
            self.downsamplers.append(kwargs.get('downsample', nn.Identity()))
            self.se_layers.append(SELayer(kwargs['out_channels']) if squeeze_excitation else nn.Identity())
            self.activations.append(kwargs.get('activation', nn.ReLU)(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for cb, cnc, d, se, a in zip(
            self.conv_blocks,
            self.change_n_channels,
            self.downsamplers,
            self.se_layers,
            self.activations
        ):
            identity = x
            x = se(cb(x))
            x = x + d(cnc(identity))
            x = a(x)
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


class FullConvActorCriticNetwork(nn.Module):
    def __init__(
            self,
            cross_normalize_value: bool = True,
            value_activation: Optional[nn.Module] = None,
            value_scale: float = 1.,
            value_shift: float = 0.,
            **residual_model_kwargs
    ):
        super(FullConvActorCriticNetwork, self).__init__()
        self.base = ResidualModel(**residual_model_kwargs)
        self.base_out_channels = residual_model_kwargs['conv_block_kwargs'][-1]['out_channels']
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
        ).view(-1)
        # Base_out_indexed (before .view()) is a tensor of shape (batch_size * n_geese, n_channels)
        # After .view(), base_out_indexed has shape (batch_size, n_geese, n_channels)
        base_out_indexed = base_out[batch_indices, head_indices].view(batch_size, n_geese, -1)
        actor_out = self.actor(base_out_indexed)
        critic_out = self.critic(base_out_indexed).squeeze(dim=-1)
        logits = torch.where(
            still_alive.unsqueeze(-1),
            actor_out,
            torch.zeros_like(actor_out)
        )
        values = torch.where(
            still_alive,
            critic_out,
            torch.zeros_like(critic_out)
        )
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


"""
"Imported" the following packages:
config
utils
goose_env
lightweight_env
basic_mcts
models
"""
BOARD_DIMS = np.array([N_ROWS, N_COLS])


def wrap(position: np.ndarray) -> np.ndarray:
    assert position.shape == (2,), f'{position.shape}'
    return (position + BOARD_DIMS) % BOARD_DIMS


# Precompute directions_dict for get_direction function
DIRECTIONS_DICT = {tuple(wrap(np.array(act.to_row_col()))): act for act in Action}


def get_direction(from_loc: np.ndarray, to_loc: np.ndarray) -> Action:
    return DIRECTIONS_DICT[tuple(wrap(to_loc - from_loc))]


def action_mask_func(state):
    return ActionMasking.LETHAL.get_action_mask(state)


def terminal_value_func(state):
    agent_rankings = stats.rankdata([agent['reward'] for agent in state], method='average') - 1.
    ranks_rescaled = 2. * agent_rankings / (len(state) - 1.) - 1.
    return ranks_rescaled


"""
Tunable parameters: 
C_PUCT: [0, inf) How much the MCTS algorithm should prioritize the predictions of the actor
    relative to the predictions of the rollouts and critic
    Larger values increase the priority of the critic, whereas smaller values increase the priority of the actor
ACTION_SELECTION_FUNC: One of (Node.max_policy_actions, Node.max_policy_actions)
    Node.max_policy_actions: Selects the final action by picking the action with the most visits (default)
    Node.max_value_actions: Selects the final action by picking the action with the highest value
    NB: Node.max_value_actions may sometimes pick actions with very few visits (and therefore a bad/unstable value_est)
RESET_SEARCH: Whether to reset search at each timestep
EXPECTED_END_STEP: Controls the time management of the agent
OVERAGE_BUFFER: How much overage time to leave as a buffer for the steps after EXPECTED_END_STEP
"""
C_PUCT = 1.
ACTION_SELECTION_FUNC = Node.max_policy_actions
DELTA = 0.05
MIN_THRESHOLD_FOR_CONSIDERATION = 0.15
MAX_SEARCH_ITER = 10
RESET_SEARCH = True
OVERAGE_BUFFER = 1.

assert C_PUCT >= 0.


class Agent:
    def __init__(self, obs: Observation, conf: Configuration):
        self.index = obs.index

        obs_type = ObsType.COMBINED_GRADIENT_OBS
        n_channels = 128
        activation = nn.ReLU
        model_kwargs = dict(
            block_class=BasicConvolutionalBlock,
            conv_block_kwargs=[
                dict(
                    in_channels=obs_type.get_obs_spec()[-3],
                    out_channels=n_channels,
                    kernel_size=3,
                    activation=activation,
                    normalize=False
                ),
                dict(
                    in_channels=n_channels,
                    out_channels=n_channels,
                    kernel_size=3,
                    activation=activation,
                    normalize=False
                ),
                dict(
                    in_channels=n_channels,
                    out_channels=n_channels,
                    kernel_size=3,
                    activation=activation,
                    normalize=False
                ),
            ],
            squeeze_excitation=True,
            cross_normalize_value=True,
        )
        self.model = FullConvActorCriticNetwork(**model_kwargs)
        state_dict_bytes = base64.b64decode(serialized_string)
        loaded_state_dicts = pickle.loads(state_dict_bytes)
        self.model.load_state_dict(loaded_state_dicts)
        self.model.eval()

        self.obs_type = obs_type
        self.search_tree = BasicMCTS(
            action_mask_func=action_mask_func,
            actor_critic_func=self.actor_critic_func,
            terminal_value_func=terminal_value_func,
            c_puct=C_PUCT,
            include_food=False,
        )
        self.last_head_locs = [row_col(goose[0]) for goose in obs.geese]
        self.last_actions = [Action.NORTH for _ in range(4)]

    def preprocess(self, obs: Observation, conf: Configuration):
        for goose_idx, goose in enumerate(obs.geese):
            if len(goose) > 0:
                if self.last_head_locs[goose_idx] is not None and obs.step > 0:
                    self.last_actions[goose_idx] = get_direction(np.array(self.last_head_locs[goose_idx]),
                                                                 np.array(row_col(goose[0])))
                else:
                    self.last_actions[goose_idx] = Action.NORTH
                self.last_head_locs[goose_idx] = row_col(goose[0])
            else:
                self.last_actions[goose_idx] = Action.NORTH
                self.last_head_locs[goose_idx] = None

    def __call__(self, obs: Observation, conf: Configuration):
        self.preprocess(obs, conf)
        env = make_from_state(obs, self.last_actions)
        # Remove excess nodes from dictionary to avoid memory explosion
        csr = env.canonical_string_repr(include_food=self.search_tree.include_food)
        if RESET_SEARCH:
            self.search_tree.reset()
        else:
            for key in list(self.search_tree.nodes.keys()):
                if key.startswith(f'S: {obs.step - 1}') or (key.startswith(f'S: {obs.step}') and key != csr):
                    del self.search_tree.nodes[key]
        # TODO: More intelligently allocate overage time when search results are uncertain
        remaining_overage_time = max(obs.remaining_overage_time - OVERAGE_BUFFER, 0.)
        search_start_time = time.time()
        root_node = self.search_tree.run_mcts(
            env=env,
            n_iter=10000,
            max_time=0.9
        )
        initial_policy = root_node.initial_policies[self.index]
        improved_policy = root_node.get_improved_policies(temp=1.)[self.index]
        actions_to_consider = improved_policy >= MIN_THRESHOLD_FOR_CONSIDERATION
        # Stop search if the following conditions are met
        if (
            improved_policy.argmax() == initial_policy.argmax() and
            improved_policy.max() >= initial_policy.max()
        ) or (
            initial_policy.max() >= 0.95
        ) or (
            actions_to_consider.sum() < 2
        ):
            my_best_action_idx = ACTION_SELECTION_FUNC(root_node)[self.index]
        else:
            if obs.step < 50:
                dynamic_max_iter = 2
            elif obs.step < 100:
                dynamic_max_iter = 4
            else:
                dynamic_max_iter = MAX_SEARCH_ITER
            n_iter = 0
            while n_iter < MAX_SEARCH_ITER and n_iter < dynamic_max_iter:
                root_node = self.search_tree.run_mcts(
                    env=env,
                    n_iter=10000,
                    max_time=min(0.5, remaining_overage_time - (time.time() - search_start_time))
                )
                new_improved_policy = root_node.get_improved_policies(temp=1.)[self.index]
                promising_actions = (new_improved_policy > initial_policy) & actions_to_consider
                if (
                    new_improved_policy.argmax() == initial_policy.argmax() and
                    new_improved_policy.max() >= initial_policy.max()
                ) or (
                    new_improved_policy.argmax() == improved_policy.argmax() and
                    new_improved_policy.argmax() != initial_policy.argmax() and
                    new_improved_policy.max() >= 0.5
                ):
                    my_best_action_idx = ACTION_SELECTION_FUNC(root_node)[self.index]
                    break
                elif (
                    promising_actions.any() and
                    new_improved_policy[promising_actions].argmax() == initial_policy[promising_actions].argmax() and
                    new_improved_policy[promising_actions].max() >= initial_policy[promising_actions].max() + DELTA
                ):
                    my_best_action_idx = np.arange(4)[
                        new_improved_policy == new_improved_policy[promising_actions].max()
                    ]
                    if len(my_best_action_idx.ravel()) == 1:
                        my_best_action_idx = int(my_best_action_idx.item())
                        print('Stopping search early!', end=' ')
                        break
                improved_policy = new_improved_policy
                n_iter += 1
            else:
                my_best_action_idx = ACTION_SELECTION_FUNC(root_node)[self.index]
        final_policies = root_node.get_improved_policies(temp=1.)
        q_vals = root_node.q_vals
        # Greedily select best action
        selected_action = tuple(Action)[my_best_action_idx].name
        print(f'Step: {obs.step + 1}', end=' ')
        print(f'Index: {self.index}', end=' ')
        print(f'My initial policy: {print_array_one_line(initial_policy)}', end=' ')
        print(f'My improved policy: {print_array_one_line(final_policies[self.index])}', end=' ')
        print(f'My Q-values: {print_array_one_line(q_vals[self.index])}', end=' ')
        print(f'Selected action: {selected_action}', end=' ')
        print(f'N-visits: {root_node.n_visits.sum(axis=1)[self.index]:.0f}', end=' ')
        print(f'Time allotted: {time.time() - search_start_time:.2f}', end=' ')
        print(f'Remaining overage time: {obs.remaining_overage_time:.2f}', end=' ')
        print(f'All initial values: {print_array_one_line(root_node.initial_values)}', end=' ')
        print(f'All policies: {print_array_one_line(final_policies)}', end=' ')
        print(f'All Q-values: {print_array_one_line(q_vals)}', end=' ')
        print()
        return selected_action

    def actor_critic_func(self, state):
        geese = state[0]['observation']['geese']
        n_geese = len(geese)

        obs = create_obs_tensor(state, self.obs_type)
        head_locs = [goose[0] if len(goose) > 0 else -1 for goose in geese]
        still_alive = [agent['status'] == 'ACTIVE' for agent in state]
        with torch.no_grad():
            logits, values = self.model(torch.from_numpy(obs),
                                        torch.tensor(head_locs).unsqueeze(0),
                                        torch.tensor(still_alive).unsqueeze(0))

        # Score the dead geese
        dead_geese_mask = np.array([len(goose) for goose in geese]) == 0
        agent_rankings = stats.rankdata([agent['reward'] for agent in state], method='average') - 1.
        agent_rankings_rescaled = 2. * agent_rankings / (n_geese - 1.) - 1.

        logits = F.softmax(logits, -1)
        final_values = np.where(
            dead_geese_mask,
            agent_rankings_rescaled,
            values.squeeze(0).numpy()
        )

        # Logits should be of shape (4, 4)
        # Values should be of shape (4, 1)
        return logits.squeeze(0).numpy().astype(np.float), final_values[:, np.newaxis]


AGENT = None


def call_agent(obs, conf):
    global AGENT

    obs = Observation(obs)
    conf = Configuration(conf)
    if AGENT is None:
        AGENT = Agent(obs, conf)

    return AGENT(obs, conf)