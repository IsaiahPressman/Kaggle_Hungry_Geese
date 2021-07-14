import copy
from enum import auto, Enum
from kaggle_environments.envs.hungry_geese.hungry_geese import Action, row_col
import numpy as np
from scipy import stats
from torch import nn
from typing import *

from ..config import *
from .lightweight_env import make
from ..utils import ActionMasking, row_col


class ObsType(Enum):
    """
    An enum of all available obs_types
    WARNING: enum order is subject to change
    """
    COMBINED_GRADIENT_OBS_SMALL = auto()
    COMBINED_GRADIENT_OBS_LARGE = auto()
    COMBINED_GRADIENT_OBS_FULL = auto()
    COMBINED_OBS_SMALL = auto()
    HEAD_CENTERED_OBS_LARGE = auto()
    HEAD_CENTERED_OBS_SMALL = auto()
    HEAD_CENTERED_SET_OBS = auto()
    SET_OBS = auto()

    def get_obs_spec(self, n_players: int = N_PLAYERS) -> Tuple[int, ...]:
        if self == ObsType.COMBINED_GRADIENT_OBS_SMALL:
            return -1, 3 + 2 * n_players, N_ROWS, N_COLS
        elif self == ObsType.COMBINED_GRADIENT_OBS_LARGE:
            return -1, 3 + 3 * n_players, N_ROWS, N_COLS
        elif self == ObsType.COMBINED_GRADIENT_OBS_FULL:
            return -1, 3 + 4 * n_players, N_ROWS, N_COLS
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


_ACTION_TO_MOVE_DICT = {}
for act in Action:
    vertical_move, horizontal_move = act.to_row_col()
    _ACTION_TO_MOVE_DICT[act] = vertical_move * N_COLS + horizontal_move


class VectorizedEnv:
    def __init__(self, obs_type: Union[ObsType, Sequence[ObsType]], reward_type: RewardType,
                 action_masking: ActionMasking = ActionMasking.OPPOSITE,
                 n_envs: int = 1, n_players: int = N_PLAYERS, silent_reset: bool = True, make_fn=make):
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
    if obs_type == ObsType.COMBINED_GRADIENT_OBS_SMALL:
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
    elif obs_type == ObsType.COMBINED_GRADIENT_OBS_LARGE:
        """
        Returns a tensor of shape (1, 3 + 3*n_players, 7, 11)
        The channels contain the following information about each cell of the board:
        for i in range (n_players):
            * contains_head[i]
            * contains_tail[i]
            * contains_body[i], where the value of this cell represents how close to the tail this cell is
                Values of 1 / GOOSE_MAX_LEN represent the tail, and n / GOOSE_MAX_LEN represents the nth element
                counting from the tail, and including the head and tail
        * contains_food
        * steps_since_starvation (normalized to be in the range 0-1)
        * current_step (normalized to be in the range 0-1)
        """
        player_channel_list = [
            'contains_head',
            'contains_tail',
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
                tail_loc = row_col(goose_loc_list[-1])
                player_channels[channel_idx_base + idx_dict['contains_tail'],
                                (*tail_loc)] = 1.
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
    elif obs_type == ObsType.COMBINED_GRADIENT_OBS_FULL:
        """
        Returns a tensor of shape (1, 3 + 4*n_players, 7, 11)
        The channels contain the following information about each cell of the board:
        for i in range (n_players):
            * contains_head[i]
            * contains_tail[i]
            * last_head_loc[i], where the head was last turn
            * contains_body[i], where the value of this cell represents how close to the tail this cell is
                Values of 1 / GOOSE_MAX_LEN represent the tail, and n / GOOSE_MAX_LEN represents the nth element
                counting from the tail, and including the head and tail
        * contains_food
        * steps_since_starvation (normalized to be in the range 0-1)
        * current_step (normalized to be in the range 0-1)
        """
        player_channel_list = [
            'contains_head',
            'contains_tail',
            'last_head_loc',
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
                player_channels[channel_idx_base + idx_dict['contains_head'], (*head_loc)] = 1.
                tail_loc = row_col(goose_loc_list[-1])
                player_channels[channel_idx_base + idx_dict['contains_tail'], (*tail_loc)] = 1.
                if observation[0]['observation']['step'] > 0:
                    reverse_action = Action[observation[agent_idx]['action']].opposite()
                    last_head_pos = (goose_loc_list[0] + _ACTION_TO_MOVE_DICT[reverse_action]) % (N_ROWS * N_COLS)
                    last_head_loc = row_col(last_head_pos)
                    player_channels[channel_idx_base + idx_dict['last_head_loc'], (*last_head_loc)] = 1.
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
