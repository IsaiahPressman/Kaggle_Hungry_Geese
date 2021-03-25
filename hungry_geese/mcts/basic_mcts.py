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
            initial_policies: np.ndarray,
            initial_values: np.ndarray
    ):
        self.geese_still_playing = np.array(geese_still_playing_mask)
        self.n_geese = len(self.geese_still_playing)
        self.virtual_loss_multiplier = np.linspace(
                -1.,
                1.,
                self.n_geese
        )[np.logical_not(self.geese_still_playing).sum()]
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

    def update(self, actions: np.ndarray, values: np.ndarray, virtual_loss: float) -> NoReturn:
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
        assert virtual_loss >= 0, f'virtual_loss must be >= 0, was {virtual_loss}'

        selected_actions_mask = np.eye(4)[actions]
        virtual_value = virtual_loss * self.virtual_loss_multiplier
        # First undo virtual_loss
        self.q_vals = np.where(
            selected_actions_mask,
            np.where(
                self.n_visits - virtual_loss > 0.,
                (self.n_visits * self.q_vals - virtual_value) / (self.n_visits - virtual_loss),
                0.
            ),
            self.q_vals
        )
        self.n_visits -= selected_actions_mask * virtual_loss
        # Then update normally
        self.q_vals = np.where(
            selected_actions_mask,
            (self.n_visits * self.q_vals + values) / (self.n_visits + 1.),
            self.q_vals
        )
        self.n_visits += selected_actions_mask

    def virtual_visit(self, actions: np.ndarray, virtual_loss: float) -> NoReturn:
        if actions.shape != (self.n_geese,):
            raise RuntimeError(f'Actions should be of shape {(self.n_geese,)}, got {actions.shape}')
        assert virtual_loss >= 0, f'virtual_loss must be >= 0, was {virtual_loss}'

        if virtual_loss != 0:
            selected_actions_mask = np.eye(4)[actions]
            virtual_value = virtual_loss * self.virtual_loss_multiplier
            self.q_vals = np.where(
                selected_actions_mask,
                (self.n_visits * self.q_vals + virtual_value) / (self.n_visits + virtual_loss),
                self.q_vals
            )
            self.n_visits += selected_actions_mask * virtual_loss
        
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
            virtual_loss: float = 3.,
            add_noise: bool = False,
            noise_val: float = 2.,
            noise_weight: float = 0.25,
            include_food: bool = True,
    ):
        self.action_mask_func = action_mask_func
        self.actor_critic_func = actor_critic_func
        self.terminal_value_func = terminal_value_func
        self.c_puct = c_puct
        self.virtual_loss = virtual_loss
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
        node.virtual_visit(a, self.virtual_loss)
        v = self._search(env)
        node.update(a, v, self.virtual_loss)
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
        node.virtual_visit(a, self.virtual_loss)
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
                node.update(a, value_est, self.virtual_loss)
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

    def run_batch_mcts(
            self,
            env: LightweightEnv,
            batch_size: int,
            n_iter: int,
            max_time: float = float('inf')
    ) -> Node:
        start_time = time.time()
        for _ in range(n_iter):
            if time.time() - start_time >= max_time:
                break

            if env.canonical_string_repr(include_food=self.include_food) not in self.nodes.keys():
                corrected_batch_size = 1
            else:
                corrected_batch_size = batch_size
            state_batch, trajectory_batch, done_batch, still_alive_batch, available_actions_batch = zip(
                *[self.expand(env.lightweight_clone()) for b in range(corrected_batch_size)]
            )
            policies_batch, values_batch = self.actor_critic_func(state_batch)

            for batch_idx in reversed(range(corrected_batch_size)):
                backprop_kwargs = dict(
                    trajectory=trajectory_batch[batch_idx],
                    still_alive=still_alive_batch[batch_idx],
                    available_actions=available_actions_batch[batch_idx]
                )
                if done_batch[batch_idx]:
                    self.backpropagate(
                        policy_est=None,
                        value_est=self.terminal_value_func(state_batch[batch_idx]),
                        **backprop_kwargs
                    )
                else:
                    self.backpropagate(
                        policy_est=policies_batch[batch_idx],
                        value_est=values_batch[batch_idx],
                        **backprop_kwargs
                    )

        return self.get_root_node(env)

    def get_root_node(self, env: LightweightEnv) -> Node:
        return self.nodes[env.canonical_string_repr(include_food=self.include_food)]

    def reset(self) -> NoReturn:
        self.nodes = {}
