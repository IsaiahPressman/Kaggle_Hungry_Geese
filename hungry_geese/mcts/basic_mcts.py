import numpy as np
import time
from typing import *

from ..env.lightweight_env import LightweightEnv
from ..utils import rowwise_random_choice


class Node:
    def __init__(
            self,
            geese_still_playing_mask: Sequence[bool],
            available_actions_masks: Optional[np.ndarray],
            initial_policies: Optional[np.ndarray]
    ):
        self.geese_still_playing = np.array(geese_still_playing_mask)
        self.n_geese = len(self.geese_still_playing)
        assert self.geese_still_playing.any()
        if available_actions_masks is None:
            self.available_actions_masks = np.ones_like(initial_policies)
        else:
            self.available_actions_masks = available_actions_masks
        self.initial_policies = initial_policies
        assert self.initial_policies.shape == (self.n_geese, 4)
        assert np.all(0. <= self.initial_policies) and np.all(self.initial_policies <= 1.)
        assert np.allclose(self.initial_policies.sum(axis=-1), 1.)
        # Re-normalize policy distribution
        self.initial_policies = self.initial_policies * available_actions_masks
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

        self.q_vals = np.zeros_like(self.initial_policies)
        self.n_visits = np.zeros_like(self.initial_policies)

    def update(self, actions: np.ndarray, values: np.ndarray) -> NoReturn:
        if actions.shape != (self.n_geese,):
            raise RuntimeError(f'Actions should be of shape {(self.n_geese,)}, got {actions.shape}')
        if values.ndim == 1:
            values = values[:, np.newaxis]
        if values.shape != (self.n_geese, 1):
            raise RuntimeError(f'Values should be of shape {(self.n_geese, 1)}, got {values.shape}')
        if not np.isclose(values.sum(), 0.):
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


class BasicMCTS:
    def __init__(
            self,
            action_mask_func: Callable,
            actor_critic_func: Callable,
            terminal_value_func: Callable,
            c_puct: float = np.sqrt(2.),
            include_food: bool = False,
    ):
        self.action_mask_func = action_mask_func
        self.actor_critic_func = actor_critic_func
        self.terminal_value_func = terminal_value_func
        self.c_puct = c_puct
        self.include_food = include_food
        self.nodes = {}

    def _search(self, env: LightweightEnv) -> np.ndarray:
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
            self.nodes[s] = Node(
                [status != 'DONE' for status in env.get_statuses()],
                self.action_mask_func(full_state),
                policy_est
            )
            return value_est

        a = node.get_puct_actions(self.c_puct).ravel()
        env.step(a)
        v = self._search(env)
        node.update(a, v)
        return v

    def run_mcts(
            self,
            env: LightweightEnv,
            n_iter: int,
            max_time: float = float('inf'),
    ) -> Node:
        start_time = time.time()
        for _ in range(n_iter):
            if time.time() - start_time >= max_time:
                break
            self._search(env.clone())

        return self.nodes[env.canonical_string_repr(include_food=self.include_food)]