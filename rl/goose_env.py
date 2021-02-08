import contextlib
import copy
import io
import numpy as np
from scipy import stats
from typing import *

with contextlib.redirect_stdout(io.StringIO()):
    # Silence gfootball import error
    from kaggle_environments import make
    from kaggle_environments.envs.hungry_geese.hungry_geese import Action, row_col

HEAD_CENTERED_OBS = 'head_centered_obs'
SET_OBS = 'set_obs'
OBS_TYPES = (
    HEAD_CENTERED_OBS,
    SET_OBS,
)

RANK_ON_DEATH = 'rank_on_death'
REWARD_TYPES = (
    RANK_ON_DEATH
)

HUNGER_RATE = 40.
MAX_NUM_STEPS = 200.
GOOSE_MAX_LEN = 99.

ACTION_SPACE = (Action.NORTH, Action.EAST, Action.SOUTH, Action.WEST)


class GooseEnvVectorized:
    def __init__(self, obs_type: str, reward_type: str,
                 n_envs: int = 1, n_players: int = 4, silent_reset: bool = True):
        self.obs_type = obs_type
        self.reward_type = reward_type
        self.n_envs = n_envs
        self.n_players = n_players
        self.silent_reset = silent_reset

        self.multi_obs_type = type(self.obs_type) != str
        self.wrapped_envs = [make('hungry_geese') for _ in range(self.n_envs)]
        self.agent_dones = np.zeros((self.n_envs, self.n_players))
        self.goose_head_locs = np.zeros((self.n_envs, self.n_players, 2), dtype=np.int64)
        self.episodes_finished_last_turn = np.zeros(self.n_envs, dtype=np.bool)
        self.episodes_finished_last_turn_info = [{} for _ in range(self.n_envs)]

        self.hard_reset()

    def hard_reset(self):
        """
        Resets all environments, whether or not they are done
        """
        self._reset_specific_envs(np.ones_like(self.episodes_done))
        self._update_other_info()

        rewards = np.zeros((self.n_envs, self.n_players))
        return self.obs, rewards, self.agent_dones, self.info_dict

    def soft_reset(self):
        """
        Only resets environments that are done and returns an observation
        """
        self._soft_reset()
        rewards = np.zeros((self.n_envs, self.n_players))
        return self.obs, rewards, self.agent_dones, self.info_dict

    def _soft_reset(self):
        """
        Only resets environments that are done
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
                    'goose_lengths': [agent['reward'] % (GOOSE_MAX_LEN + 1.) for agent in env.steps[-1]],
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
        assert np.all(actions < len(ACTION_SPACE))
        agent_already_dones = self.agent_dones.copy()

        current_standings = np.zeros((self.n_envs, self.n_players))
        for i in range(self.n_envs):
            observations = self.wrapped_envs[i].step([ACTION_SPACE[act].name for act in actions[i]])
            self.agent_dones[i] = [agent['status'] == 'DONE' for agent in observations]
            current_standings[i] = [agent['reward'] for agent in observations]

        rewards = np.zeros((self.n_envs, self.n_players))
        if self.reward_type == RANK_ON_DEATH:
            reward_range = np.linspace(-1., 1., self.n_players)
            agent_rankings = stats.rankdata(current_standings, method='max', axis=1) - 1
            rewards = np.where(
                np.abs(agent_already_dones - self.agent_dones),
                reward_range[agent_rankings],
                rewards
            )
        else:
            raise ValueError(f'Unsupported reward_type: {self.reward_type}')

        agent_dones_cached = self.agent_dones.copy()
        if self.episodes_done.any() and self.silent_reset:
            self._soft_reset()

        self._update_other_info()
        return self.obs, rewards, agent_dones_cached, self.info_dict

    @property
    def obs(self):
        if self.multi_obs_type:
            return [self._get_obs(ot) for ot in self.obs_type]
        else:
            return self._get_obs(self.obs_type)

    def _get_obs(self, obs_type: str):
        if obs_type == HEAD_CENTERED_OBS:
            obs = self._get_head_centered_obs()
        elif obs_type == SET_OBS:
            obs = self._get_set_obs()
        else:
            raise ValueError(f'Unsupported obs_type: {obs_type}')
        return obs

    def _get_head_centered_obs(self) -> np.ndarray:
        """
        Returns a tensor of shape (n_envs, n_players, 13, 7, 11)
        Axes represent (n_envs, n_players, n_channels, n_rows, n_cols)
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
        contains_head = np.zeros((self.n_envs, self.n_players, 7, 11), dtype=np.float32)
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
        for env_idx, env in enumerate(self.wrapped_envs):
            observation = env.steps[-1]
            for food_loc_n in observation[0]['observation']['food']:
                food_loc = _row_col(food_loc_n)
                contains_food[env_idx, :, food_loc[0], food_loc[1]] = 1.
            steps_since_starvation[env_idx] = observation[0]['observation']['step'] % HUNGER_RATE
            current_step[env_idx] = observation[0]['observation']['step']
            for centered_agent_idx in range(self.n_players):
                for agent_idx in range(self.n_players):
                    is_centered_multiplier = 1. if centered_agent_idx == agent_idx else -1.
                    goose_loc_list = observation[0]['observation']['geese'][agent_idx]
                    # Make sure the goose is still alive
                    if len(goose_loc_list) > 0:
                        head_loc = _row_col(goose_loc_list[0])
                        contains_head[env_idx, centered_agent_idx, (*head_loc)] = 1. * is_centered_multiplier
                        if observation[0]['observation']['step'] > 0:
                            last_action_channel = last_action_dict[observation[agent_idx]['action']]
                            last_action_channel[env_idx, centered_agent_idx, (*head_loc)] = 1. * is_centered_multiplier
                    # Make sure the goose is more than just a head
                    if len(goose_loc_list) > 1:
                        for body_loc_n in goose_loc_list[1:]:
                            body_loc = _row_col(body_loc_n)
                            contains_body[env_idx, centered_agent_idx, (*body_loc)] = 1. * is_centered_multiplier
                        for i in range(len(goose_loc_list) - 1):
                            connection_channel = connected_dict[_get_direction(
                                goose_loc_list[i],
                                goose_loc_list[i + 1]
                            )]
                            body_loc = _row_col(goose_loc_list[i])
                            connection_channel[env_idx, centered_agent_idx, (*body_loc)] = 1. * is_centered_multiplier
                            next_connection_channel = connected_dict[_get_direction(
                                goose_loc_list[i + 1],
                                goose_loc_list[i]
                            )]
                            next_body_loc = _row_col(goose_loc_list[i + 1])
                            next_connection_channel[
                                env_idx,
                                centered_agent_idx,
                                (*next_body_loc)
                            ] = 1. * is_centered_multiplier
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
        ], axis=2)
        for env_idx in range(self.n_envs):
            for centered_agent_idx in range(self.n_players):
                if not self.agent_dones[env_idx, centered_agent_idx]:
                    head_loc = self.goose_head_locs[env_idx, centered_agent_idx]
                    translation = np.array([3, 5]) - head_loc
                    obs[env_idx, centered_agent_idx] = np.roll(
                        obs[env_idx, centered_agent_idx], translation, axis=(-2, -1)
                    )
                    if obs[env_idx, centered_agent_idx, 0, 3, 5] != 1.:
                        print('Head not centered!')
                        print(head_loc)
                        print(obs[env_idx, centered_agent_idx, 1])
                        assert False
        return obs

    def _get_set_obs(self) -> np.ndarray:
        """
        Returns a tensor of shape (n_envs, 13, 7, 11)
        Axes represent (n_envs, n_channels, n_rows, n_cols)
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
        contains_head = np.zeros((self.n_envs, 7, 11), dtype=np.float32)
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
        for env_idx, env in enumerate(self.wrapped_envs):
            observation = env.steps[-1]
            for food_loc_n in observation[0]['observation']['food']:
                food_loc = _row_col(food_loc_n)
                contains_food[env_idx, (*food_loc)] = 1.
            for agent_idx in range(self.n_players):
                goose_loc_list = observation[0]['observation']['geese'][agent_idx]
                # Make sure the goose is still alive
                if len(goose_loc_list) > 0:
                    head_loc = _row_col(goose_loc_list[0])
                    contains_head[env_idx, (*head_loc)] = 1.
                    if observation[0]['observation']['step'] > 0:
                        last_action_channel = last_action_dict[observation[agent_idx]['action']]
                        last_action_channel[env_idx, (*head_loc)] = 1.
                # Make sure the goose is more than just a head
                if len(goose_loc_list) > 1:
                    for body_loc_n in goose_loc_list[1:]:
                        body_loc = _row_col(body_loc_n)
                        contains_body[env_idx, (*body_loc)] = 1.
                    for i in range(len(goose_loc_list) - 1):
                        connection_channel = connected_dict[_get_direction(
                            goose_loc_list[i],
                            goose_loc_list[i+1]
                        )]
                        body_loc = _row_col(goose_loc_list[i])
                        connection_channel[env_idx, (*body_loc)] = 1.
                        next_connection_channel = connected_dict[_get_direction(
                            goose_loc_list[i+1],
                            goose_loc_list[i]
                        )]
                        next_body_loc = _row_col(goose_loc_list[i+1])
                        next_connection_channel[env_idx, (*next_body_loc)] = 1.
            steps_since_starvation[env_idx] = observation[0]['observation']['step'] % HUNGER_RATE
            current_step[env_idx] = observation[0]['observation']['step']
        return np.stack([
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

    def _update_other_info(self):
        self.goose_head_locs = np.zeros((self.n_envs, self.n_players, 2), dtype=np.int64)
        for env_idx, env in enumerate(self.wrapped_envs):
            for agent_idx, goose_loc_list in enumerate(env.steps[-1][0]['observation']['geese']):
                if len(goose_loc_list) > 0:
                    self.goose_head_locs[env_idx, agent_idx] = _row_col(goose_loc_list[0])

    @property
    def info_dict(self) -> Dict:
        return {
            'head_positions': self.goose_head_locs,
            'episodes_finished_last_turn': self.episodes_finished_last_turn,
            'episodes_finished_last_turn_info': self.episodes_finished_last_turn_info,
        }

    @property
    def episodes_done(self):
        return self.agent_dones.all(axis=1)

    @staticmethod
    def get_obs_spec(obs_type: str) -> Tuple[int, ...]:
        board_shape = (7, 11)
        if obs_type == HEAD_CENTERED_OBS:
            return -1, -1, 13, board_shape[0], board_shape[1]
        elif obs_type == SET_OBS:
            return -1, 13, board_shape[0], board_shape[1]
        else:
            raise ValueError(f'Unrecognized obs_type: {obs_type}')


_DIRECTIONS_DICT = {act.to_row_col(): act.name for act in ACTION_SPACE}
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
    from_loc = np.array(_row_col(from_position))
    to_loc = np.array(_row_col(to_position))
    return _DIRECTIONS_DICT[tuple(to_loc - from_loc)]


def _row_col(position: int) -> Tuple[int, int]:
    return row_col(position, 11)
