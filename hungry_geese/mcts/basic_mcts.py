import numpy as np
import time
from typing import *

from ..env.lightweight_env import LightweightEnv
from ..utils import rowwise_random_choice


class Node:
    def __init__(
            self,
            parent: Optional["Node"],
            last_actions: Optional[Tuple[int, ...]],
            env_cloned: LightweightEnv,
            available_actions_masks: Optional[np.ndarray],
            initial_policies: np.ndarray
    ):
        self.parent = parent
        self.last_actions = last_actions
        if self.parent is not None and self.last_actions is None:
            raise ValueError('If parent is not None, last_actions must be provided')
        self.env = env_cloned
        if available_actions_masks is None:
            self.available_actions_masks = np.ones_like(initial_policies)
        else:
            self.available_actions_masks = available_actions_masks
        self.initial_policies = initial_policies
        assert self.initial_policies.shape == (len(self.env.state), 4)
        self.geese_still_playing = np.array(env_cloned.get_statuses()) != 'DONE'
        assert self.geese_still_playing.any()

        self.children = {}
        self.q_vals = np.zeros_like(self.initial_policies)
        self.n_visits = np.zeros_like(self.initial_policies)

    def get_puct_actions(self, c_puct: float) -> np.ndarray:
        uct_vals = self.q_vals + (c_puct *
                                  self.initial_policies *
                                  np.sqrt(self.n_visits.sum(axis=1, keepdims=True)) / (1. + self.n_visits))
        uct_vals *= self.available_actions_masks
        """
        We could just go from here using a simple np.argmax(uct_vals, axis=1) to select actions,
        but this would always tiebreak towards actions with lower indices.
        Therefore, we sample from all actions whose uct value equals the max uct value for that agent.
        """
        uct_max = uct_vals.max(axis=1, keepdims=True)
        return np.where(
            self.geese_still_playing,
            rowwise_random_choice((uct_vals == uct_max).astype(np.float)),
            -1
        )

    def backpropagate(self, actions: Tuple[int, ...], values: np.ndarray) -> NoReturn:
        if values.ndim == 1:
            values = values[:, np.newaxis]
        if values.shape != (len(actions), 1):
            raise RuntimeError(f'Values should be of shape {(len(actions), 1)}, got {values.shape}')
        if not np.isclose(values.sum(), 0.):
            raise RuntimeError(f'Values should sum to 0, got {values.ravel()} which sums to {values.sum()}')
        if (values.ravel()[self.geese_still_playing].min(initial=float('inf')) <=
                values.ravel()[~self.geese_still_playing].max(initial=float('-inf'))):
            raise RuntimeError(f'Values for dead geese should always be less than those for still living geese.\n'
                               f'Values:\n{values.ravel()}\n'
                               f'Geese still playing:\n{self.geese_still_playing}\n')
        selected_actions_mask = np.eye(4)[np.array(actions)]
        self.q_vals = np.where(
            selected_actions_mask,
            (self.n_visits * self.q_vals + values) / (self.n_visits + 1.),
            self.q_vals
        )
        self.n_visits += selected_actions_mask
        if self.parent is not None:
            self.parent.backpropagate(self.last_actions, values)

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
            -1
        )


class TerminalNode:
    def __init__(
            self,
            parent: Node,
            last_actions: Tuple[int, ...],
            values: np.ndarray
    ):
        self.parent = parent
        self.last_actions = last_actions
        if values.ndim == 1:
            values = values[:, np.newaxis]
        if values.shape != (len(last_actions), 1):
            raise RuntimeError(f'Values should be of shape {(len(last_actions), 1)}, got {values.shape}')
        if not np.isclose(values.sum(), 0.):
            raise RuntimeError(f'Values should sum to 0, got {values.ravel()} which sums to {values.sum()}')
        self.values = values

        self.children = {}

    def backpropagate_from_terminal(self) -> NoReturn:
        self.parent.backpropagate(self.last_actions, self.values)


def run_mcts(
        env: LightweightEnv,
        n_iter: int,
        action_mask_func: Callable,
        actor_critic_func: Callable,
        terminal_value_func: Callable,
        max_time: float = float('inf'),
        c_puct: float = np.sqrt(2.),
) -> Node:
    start_time = time.time()
    state = env.state
    root_node = Node(None, None, env.clone(), action_mask_func(state), actor_critic_func(state)[0])
    for i in range(n_iter):
        if time.time() - start_time >= max_time:
            break

        # Expand until unexplored node is found
        parent = root_node
        selected_actions = tuple(parent.get_puct_actions(c_puct).ravel())
        child = parent.children.get(selected_actions, None)
        while child is not None and type(child) != TerminalNode:
            parent = child
            selected_actions = tuple(parent.get_puct_actions(c_puct).ravel())
            child = parent.children.get(selected_actions, None)

        if type(child) == TerminalNode:
            child.backpropagate_from_terminal()
        else:
            new_env = parent.env.clone()
            state = new_env.step(selected_actions)
            if new_env.done:
                new_child = TerminalNode(
                    parent,
                    selected_actions,
                    terminal_value_func(state)
                )
                new_child.backpropagate_from_terminal()
            else:
                policy_est, value_est = actor_critic_func(state)
                new_child = Node(
                    parent,
                    selected_actions,
                    new_env,
                    action_mask_func(state),
                    policy_est
                )
                parent.backpropagate(selected_actions, value_est)
            parent.children[selected_actions] = new_child

    return root_node
