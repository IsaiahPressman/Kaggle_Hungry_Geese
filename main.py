from kaggle_environments.envs.hungry_geese.hungry_geese import Action, Configuration, Observation
import numpy as np
from pathlib import Path
from scipy import stats
import sys
import time
import torch
from torch import nn
import torch.nn.functional as F

# See https://www.kaggle.com/c/google-football/discussion/191257
sys.path.append('/kaggle_simulations/agent/')
from hungry_geese.config import *
from hungry_geese.utils import ActionMasking, row_col, print_array_one_line
from hungry_geese.env import goose_env as ge
from hungry_geese.env.lightweight_env import make_from_state
from hungry_geese.mcts.basic_mcts import Node, BasicMCTS
from hungry_geese.nns import models, conv_blocks

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
DELTA = 0.08
MIN_THRESHOLD_FOR_CONSIDERATION = 0.15
MAX_SEARCH_ITER = 10
RESET_SEARCH = True
OVERAGE_BUFFER = 1.
PER_ROUND_BATCHED_TIME_ALLOCATION = 0.9
BATCH_SIZE = 1

assert C_PUCT >= 0.


class Agent:
    def __init__(self, obs: Observation, conf: Configuration):
        self.index = obs.index

        obs_type = ge.ObsType.COMBINED_GRADIENT_OBS
        n_channels = 128
        activation = nn.ReLU
        model_kwargs = dict(
            block_class=conv_blocks.BasicConvolutionalBlock,
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
        self.model = models.FullConvActorCriticNetwork(**model_kwargs)
        try:
            self.model.load_state_dict(torch.load('/kaggle_simulations/agent/cp.pt'))
        except FileNotFoundError:
            self.model.load_state_dict(torch.load(Path.home() / 'goose_agent/cp.pt'))
        self.model.eval()

        self.obs_type = obs_type
        self.search_tree = BasicMCTS(
            action_mask_func=action_mask_func,
            actor_critic_func=self.batch_actor_critic_func,
            terminal_value_func=terminal_value_func,
            c_puct=C_PUCT,
            virtual_loss=3.,
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
        self.search_tree.run_batch_mcts(
            env=env,
            batch_size=BATCH_SIZE,
            n_iter=10000,
            max_time=PER_ROUND_BATCHED_TIME_ALLOCATION
        )
        root_node = self.search_tree.run_batch_mcts(
            env=env,
            batch_size=2,
            n_iter=10000,
            max_time=max(0.9 - (time.time() - search_start_time), 0.)
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
                root_node = self.search_tree.run_batch_mcts(
                    env=env,
                    batch_size=BATCH_SIZE,
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

    def batch_actor_critic_func(self, state_batch):
        obs_list = []
        head_locs_list = []
        still_alive_list = []
        rewards_list = []
        n_geese = len(state_batch[0][0]['observation']['geese'])
        for state in state_batch:
            geese = state[0]['observation']['geese']
            assert len(geese) == n_geese, 'All environments must have the same number of geese for batching'

            obs_list.append(ge.create_obs_tensor(state, self.obs_type))
            head_locs_list.append([goose[0] if len(goose) > 0 else -1 for goose in geese])
            still_alive_list.append([agent['status'] == 'ACTIVE' for agent in state])
            rewards_list.append([agent['reward'] for agent in state])
        # TODO: Don't perform inference on terminal states
        with torch.no_grad():
            logits, values = self.model(
                torch.from_numpy(np.concatenate(obs_list, axis=0)),
                torch.tensor(head_locs_list),
                torch.tensor(still_alive_list)
            )
            probs = F.softmax(logits, dim=-1).numpy().astype(np.float)
        # Score the dead geese
        dead_geese_mask = ~np.array(still_alive_list)
        # This doesn't work in Kaggle environment:
        # agent_rankings = stats.rankdata(np.array(rewards_list), method='average', axis=-1) - 1.
        agent_rankings = np.stack(
            [stats.rankdata(r, method='average') for r in rewards_list],
            axis=0
        ) - 1.
        agent_rankings_rescaled = 2. * agent_rankings / (n_geese - 1.) - 1.

        final_values = np.where(
            dead_geese_mask,
            agent_rankings_rescaled,
            values.numpy()
        )
        # Logits should be of shape (n_envs, n_geese, 4)
        # Values should be of shape (n_envs, n_geese, 1)
        return probs, np.expand_dims(final_values, axis=-1)


AGENT = None


def call_agent(obs, conf):
    global AGENT

    obs = Observation(obs)
    conf = Configuration(conf)
    if AGENT is None:
        AGENT = Agent(obs, conf)

    return AGENT(obs, conf)
