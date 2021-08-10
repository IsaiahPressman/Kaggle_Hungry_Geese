from kaggle_environments.envs.hungry_geese.hungry_geese import Action, Configuration, Observation
import numpy as np
from pathlib import Path
from scipy import stats
import sys
import time
import torch
from torch import nn
import torch.nn.functional as F
from typing import *

# See https://www.kaggle.com/c/google-football/discussion/191257
sys.path.append('/kaggle_simulations/agent/')
from hungry_geese.config import *
from hungry_geese.utils import STATE_TYPE, ActionMasking, row_col, print_array_one_line
from hungry_geese.env import goose_env as ge
from hungry_geese.env.lightweight_env import LightweightEnv, make_from_state
from hungry_geese.mcts.basic_mcts import BasicMCTS, Node
from hungry_geese.nns import conv_blocks, models

BOARD_DIMS = np.array([N_ROWS, N_COLS])


def wrap(position: np.ndarray) -> np.ndarray:
    assert position.shape == (2,), f'{position.shape}'
    return (position + BOARD_DIMS) % BOARD_DIMS


# Precompute directions_dict for get_direction function
DIRECTIONS_DICT = {tuple(wrap(np.array(act.to_row_col()))): act for act in Action}


def get_direction(from_loc: np.ndarray, to_loc: np.ndarray) -> Action:
    return DIRECTIONS_DICT[tuple(wrap(to_loc - from_loc))]


# Credit for scaled and inverse_scaled sigmoid:
# https://stackoverflow.com/questions/52972158/sigmoid-scale-and-inverse-with-custom-range
def scaled_sigmoid(x: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    return min_val + (max_val - min_val) / (1. + torch.exp(-x))


def inverse_scaled_sigmoid(x: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    return torch.log((x - min_val) / (max_val - x))


def entropy(pk: torch.Tensor, dim: int = 0, keepdim: bool = False) -> torch.Tensor:
    pk = pk / pk.sum(dim=dim, keepdim=True)
    result = torch.where(
        pk != 0.,
        pk * pk.log(),
        torch.zeros_like(pk)
    )
    return -result.sum(dim=dim, keepdim=keepdim)


def kl_divergence(
        pk: torch.Tensor,
        qk: torch.Tensor,
        dim: int = 0,
        keepdim: bool = False,
        eps: float = 1e-10
) -> torch.Tensor:
    pk = pk + eps
    qk = qk + eps
    pk = pk / pk.sum(dim=dim, keepdim=True)
    qk = qk / qk.sum(dim=dim, keepdim=True)
    if eps > 0.:
        result = pk * torch.log(pk / qk)
    elif eps == 0.:
        result = torch.where(
            pk != 0.,
            pk * torch.log(pk / qk),
            torch.zeros_like(pk)
        )
    else:
        raise ValueError(f'eps should be >= 0., was {eps}')

    return result.sum(dim=dim, keepdim=keepdim)


"""
Tunable parameters: 
C_PUCT: [0, inf) How much the MCTS algorithm should prioritize the predictions of the actor
    relative to the predictions of the rollouts and critic
    Larger values increase the priority of the critic, whereas smaller values increase the priority of the actor
RESET_SEARCH: Whether to reset search at each timestep
EXPECTED_END_STEP: Controls the time management of the agent
OVERAGE_BUFFER: How much overage time to leave as a buffer for the steps after EXPECTED_END_STEP
"""
C_PUCT = 1.
DELTA = 0.12
POLICY_TEMP = 1.
MIN_THRESHOLD_FOR_CONSIDERATION = 0.15
MAX_SEARCH_ITER = 6
RESET_SEARCH = True
OVERAGE_BUFFER = 2.
PER_ROUND_BATCHED_TIME_ALLOCATION = 0.9
BATCH_SIZE = 1
RESCALE_VALUES = True

# Dynamic noise for opponent actions
NOISE_TYPE = 'linear'
MIN_NOISE = 0.0
MAX_NOISE = 0.75
STARTING_NOISE = 0.1
N_ITER_NOISE_OPT = 4
NOISE_OPT_LR = 1. / N_ITER_NOISE_OPT
NOISE_OPT_MOMENTUM = 0.
UPDATE_NOISE_BASED_ON_ALL_STEPS = True

# Used for local evaluation
LOCAL_EVAL_MULTIPLIER = 0.5
LOCAL_EVAL_USE_SEARCH = True
LOCAL_EVAL_MAX_ITER = 10000 if LOCAL_EVAL_USE_SEARCH else 0


class Agent:
    def __init__(self, obs: Observation, conf: Configuration):
        self.index = obs.index
        self.n_geese = len(obs.geese)

        obs_type = ge.ObsType.COMBINED_GRADIENT_OBS_FULL
        n_channels = 64
        activation = nn.GELU
        normalize = False
        use_mhsa = False
        use_preprocessing = False
        model_kwargs = dict(
            preprocessing_layer=nn.Sequential(
                nn.Conv2d(
                    obs_type.get_obs_spec()[-3],
                    n_channels,
                    (1, 1)
                ),
                activation(),
                nn.Conv2d(
                    n_channels,
                    n_channels,
                    (1, 1)
                ),
                activation(),
            ) if use_preprocessing else None,
            block_class=conv_blocks.BasicConvolutionalBlock,
            block_kwargs=[
                dict(
                    in_channels=n_channels if use_preprocessing else obs_type.get_obs_spec()[-3],
                    out_channels=n_channels,
                    kernel_size=5,
                    activation=activation,
                    normalize=normalize,
                    use_mhsa=False
                ),
                dict(
                    in_channels=n_channels,
                    out_channels=n_channels,
                    kernel_size=3,
                    activation=activation,
                    normalize=normalize,
                    use_mhsa=False
                ),
                dict(
                    in_channels=n_channels,
                    out_channels=n_channels,
                    kernel_size=3,
                    activation=activation,
                    normalize=normalize,
                    use_mhsa=False
                ),
                dict(
                    in_channels=n_channels,
                    out_channels=n_channels,
                    kernel_size=3,
                    activation=activation,
                    normalize=normalize,
                    use_mhsa=False
                ),
                dict(
                    in_channels=n_channels,
                    out_channels=n_channels,
                    kernel_size=3,
                    activation=activation,
                    normalize=normalize,
                    use_mhsa=False
                ),
                dict(
                    in_channels=n_channels,
                    out_channels=n_channels,
                    kernel_size=3,
                    activation=activation,
                    normalize=normalize,
                    use_mhsa=False
                ),
            ],
            n_action_value_layers=2,
            squeeze_excitation=True,
            cross_normalize_value=True,
            use_separate_action_value_heads=True,
            # **ge.RewardType.RANK_ON_DEATH.get_recommended_value_activation_scale_shift_dict()
        )
        self.model = models.FullConvActorCriticNetwork(**model_kwargs)
        try:
            self.model.load_state_dict(torch.load('/kaggle_simulations/agent/cp.pt'))
            self.le_mult = 1.
            self.max_n_iter = 10000
            # Always use search at test time
            self.use_search = True
        except FileNotFoundError:
            self.le_mult = LOCAL_EVAL_MULTIPLIER
            self.max_n_iter = LOCAL_EVAL_MAX_ITER
            self.use_search = LOCAL_EVAL_USE_SEARCH
            print('Running local evaluation!')
            try:
                self.model.load_state_dict(torch.load(Path.home() / 'GitHub/Kaggle/Hungry_Geese/cp.pt'))
            except FileNotFoundError:
                self.model.load_state_dict(torch.load(Path.home() / 'github_misc/Kaggle_Hungry_Geese/cp.pt'))
        self.model.eval()
        self.obs_type = obs_type

        """
        # Torchscript trace the model
        dummy_state = torch.zeros(self.obs_type.get_obs_spec()[1:])
        dummy_head_loc = torch.arange(4).to(dtype=torch.int64)
        still_alive = torch.ones(4).to(dtype=torch.bool)
        self.model = torch.jit.trace(
            self.model,
            (dummy_state.unsqueeze(0),
             dummy_head_loc.unsqueeze(0),
             still_alive.unsqueeze(0))
        )
        """

        self.search_tree = BasicMCTS(
            action_mask_func=ActionMasking.LETHAL.get_action_mask,
            actor_critic_func=self.batch_actor_critic_func if BATCH_SIZE > 1 else self.actor_critic_func,
            terminal_value_func=self.terminal_value_func,
            c_puct=C_PUCT,
            virtual_loss=3.,
            include_food=False,
            enforce_values=False
        )
        self.last_head_locs = [row_col(goose[0]) for goose in obs.geese]
        self.last_actions = [Action.NORTH for _ in range(self.n_geese)]
        if NOISE_TYPE == 'linear':
            self.noise_weight_logits = torch.tensor([STARTING_NOISE] * self.n_geese)
        elif NOISE_TYPE == 'sigmoid':
            self.noise_weight_logits = inverse_scaled_sigmoid(
                torch.tensor([STARTING_NOISE] * self.n_geese),
                MIN_NOISE,
                MAX_NOISE
            )
        else:
            raise ValueError(f'Unrecognized NOISE_TYPE: {NOISE_TYPE}')
        self.noise_weight_logits = self.noise_weight_logits.view(1, self.n_geese, 1).requires_grad_()
        self.noise_opt = torch.optim.SGD(
            (self.noise_weight_logits,),
            lr=NOISE_OPT_LR,
            momentum=NOISE_OPT_MOMENTUM
        )
        self.all_actions = torch.zeros((conf.episode_steps - 1, self.n_geese, 4), dtype=torch.float32)
        self.all_policy_preds = torch.zeros_like(self.all_actions)
        self.all_action_masks = torch.zeros_like(self.all_actions)
        self.still_alive = np.ones(self.n_geese, dtype=bool)

    def preprocess(self, obs: Observation, conf: Configuration) -> NoReturn:
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
                self.still_alive[goose_idx] = False

    def __call__(self, obs: Observation, conf: Configuration) -> str:
        start_time = time.time()
        self.preprocess(obs, conf)

        noise_update_start_time = time.time()
        if obs.step >= 1:
            self.all_actions[
                [obs.step - 1] * self.n_geese,
                list(range(self.n_geese)),
                [a.value - 1 for a in self.last_actions]
            ] = 1.
            update_mask = np.array([len(g) > 0 for g in obs.geese])
            update_mask[self.index] = False
            noise_weights_logging = self.update_noise_weights(obs.step, update_mask)
        else:
            noise_weights_logging = ''
        noise_update_time = time.time() - noise_update_start_time

        search_start_time = time.time()
        env = make_from_state(obs, self.last_actions)
        csr = env.canonical_string_repr(include_food=self.search_tree.include_food)
        if RESET_SEARCH:
            self.search_tree.reset()
        else:
            # Remove excess nodes from dictionary to avoid memory explosion
            for key in list(self.search_tree.nodes.keys()):
                if key.startswith(f'S: {obs.step - 1}') or (key.startswith(f'S: {obs.step}') and key != csr):
                    del self.search_tree.nodes[key]
        my_best_action_idx, root_node, stopped_early = self.run_search(obs, env)
        search_time = time.time() - search_start_time

        initial_values = root_node.initial_values
        final_policies = root_node.get_improved_policies(temp=1.)
        q_vals = root_node.q_vals
        self.all_policy_preds[obs.step] = torch.from_numpy(final_policies).to(torch.float32)
        self.all_action_masks[obs.step] = torch.from_numpy(root_node.available_actions_masks).to(torch.float32)

        # Greedily select best action
        if self.use_search:
            selected_action = tuple(Action)[my_best_action_idx].name
        else:
            my_policy = root_node.initial_policies[self.index]
            selected_action = tuple(Action)[my_policy.argmax()].name

        # Logging stuff
        early_stopped_logging = 'Stopped search early!' if stopped_early else ''
        total_time = time.time() - start_time
        initial_values = np.where(
            self.still_alive[:, None],
            initial_values,
            float('NaN')
        )
        final_policies = np.where(
            self.still_alive[:, None],
            final_policies,
            float('NaN')
        )
        q_vals = np.where(
            self.still_alive[:, None],
            q_vals,
            float('NaN')
        )
        print(
            f'Step: {obs.step + 1} '
            f'Index: {self.index} '
            f'My initial policy: {print_array_one_line(root_node.initial_policies[self.index])} '
            f'My improved policy: {print_array_one_line(final_policies[self.index])} '
            f'My Q-values: {print_array_one_line(q_vals[self.index])} '
            f'Selected action: {selected_action} '
            f'N-visits: {root_node.n_visits.sum(axis=1)[self.index]:.0f} '
            f'Total/search/noise update time: {total_time:.2f}/{search_time:.2f}/{noise_update_time:.2f} '
            f'Remaining overage time: {obs.remaining_overage_time:.2f} '
            f'All initial values: {print_array_one_line(initial_values.ravel())} '
            f'All policies: {print_array_one_line(final_policies)} '
            f'All Q-values: {print_array_one_line(q_vals)} '
            f'{noise_weights_logging} '
            f'{early_stopped_logging} '
        )
        return selected_action

    def run_search(self, obs: Observation, env: LightweightEnv) -> Tuple[int, Node, bool]:
        remaining_overage_time = max(obs.remaining_overage_time - OVERAGE_BUFFER, 0.) * self.le_mult
        search_start_time = time.time()
        if BATCH_SIZE > 1:
            self.search_tree.run_batch_mcts(
                env=env,
                batch_size=BATCH_SIZE,
                n_iter=self.max_n_iter + 2,
                max_time=PER_ROUND_BATCHED_TIME_ALLOCATION * self.le_mult
            )
            root_node = self.search_tree.run_batch_mcts(
                env=env,
                batch_size=min(BATCH_SIZE, 2),
                n_iter=self.max_n_iter,
                max_time=max(0.93 * self.le_mult - (time.time() - search_start_time), 0.)
            )
        else:
            root_node = self.search_tree.run_mcts(
                env=env,
                n_iter=self.max_n_iter + 2,
                max_time=0.93 * self.le_mult
            )
        initial_policy = root_node.initial_policies[self.index]
        improved_policy = root_node.get_improved_policies(temp=1.)[self.index]
        actions_to_consider = initial_policy >= MIN_THRESHOLD_FOR_CONSIDERATION
        early_stop = False
        # Stop search if the following conditions are met
        if (
                improved_policy.argmax() == initial_policy.argmax() and
                improved_policy.max() >= initial_policy.max()
        ) or (
                initial_policy.max() >= 0.95
        ) or (
                actions_to_consider.sum() < 2
        ):
            my_best_action_idx = root_node.get_max_policy_actions()[self.index]
        else:
            if obs.step < 50:
                dynamic_max_iter = 2
            elif obs.step < 100:
                dynamic_max_iter = 4
            else:
                dynamic_max_iter = MAX_SEARCH_ITER
            n_iter = 0
            while n_iter < MAX_SEARCH_ITER and n_iter < dynamic_max_iter:
                if BATCH_SIZE > 1:
                    root_node = self.search_tree.run_batch_mcts(
                        env=env,
                        batch_size=BATCH_SIZE,
                        n_iter=self.max_n_iter,
                        max_time=min(0.5 * self.le_mult, remaining_overage_time - (time.time() - search_start_time))
                    )
                else:
                    root_node = self.search_tree.run_mcts(
                        env=env,
                        n_iter=self.max_n_iter,
                        max_time=min(0.5 * self.le_mult, remaining_overage_time - (time.time() - search_start_time))
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
                    my_best_action_idx = root_node.get_max_policy_actions()[self.index]
                    break
                elif (
                        promising_actions.any() and
                        new_improved_policy[promising_actions].argmax() == initial_policy[
                            promising_actions].argmax() and
                        new_improved_policy[promising_actions].max() >= initial_policy[promising_actions].max() + DELTA
                ):
                    my_best_action_idx = np.arange(4)[
                        new_improved_policy == new_improved_policy[promising_actions].max()
                        ]
                    if len(my_best_action_idx.ravel()) == 1:
                        my_best_action_idx = int(my_best_action_idx.item())
                        early_stop = True
                        break
                improved_policy = new_improved_policy
                n_iter += 1
            else:
                my_best_action_idx = root_node.get_max_policy_actions()[self.index]

        return my_best_action_idx, root_node, early_stop

    def update_noise_weights(self, current_step: int, update_mask: Sequence[bool], eps: float = 1e-10) -> str:
        info = ''
        masked_actions = self.all_actions[:, update_mask]
        masked_policy_preds = self.all_policy_preds[:, update_mask]
        masked_action_masks = self.all_action_masks[:, update_mask]
        overall_kl_divs = kl_divergence(
            self.all_actions[:current_step],
            self.all_policy_preds[:current_step],
            dim=-1,
            keepdim=True,
            eps=0.
        ) / (entropy(self.all_action_masks[:current_step], dim=-1, keepdim=True) + eps)
        overall_kl_divs = torch.where(
            torch.isinf(overall_kl_divs),
            torch.ones_like(overall_kl_divs),
            overall_kl_divs
        )

        last_step_noise = torch.ones((1, 1, 4), dtype=torch.float32) * self.all_action_masks[current_step - 1]
        last_step_noise = last_step_noise / last_step_noise.sum(dim=-1, keepdim=True)
        all_steps_noise = torch.ones((1, 1, 4), dtype=torch.float32) * self.all_action_masks[:current_step]
        all_steps_noise = all_steps_noise / all_steps_noise.sum(dim=-1, keepdim=True)
        noise_grads = []
        for _ in range(N_ITER_NOISE_OPT):
            masked_noise_weights = self.noise_weights[:, update_mask]
            self.noise_opt.zero_grad()
            if UPDATE_NOISE_BASED_ON_ALL_STEPS:
                # Update noise_weights based on all previous experience
                masked_noise = all_steps_noise[:, update_mask]
                kl_divs_with_noise = kl_divergence(
                    masked_actions[:current_step],
                    masked_policy_preds[:current_step] * (1. - masked_noise_weights) + masked_noise * masked_noise_weights,
                    dim=-1,
                    keepdim=True
                ) / (entropy(masked_action_masks[:current_step], dim=-1, keepdim=True) + eps)
                # Ensures consistent gradient sizing no matter the timestep
                kl_divs_with_noise = kl_divs_with_noise.mean(dim=0)
            else:
                # Update noise_weights based only on the last step
                masked_noise = last_step_noise[:, update_mask]
                kl_divs_with_noise = kl_divergence(
                    masked_actions[current_step - 1],
                    (1. - masked_noise_weights) * masked_policy_preds[current_step - 1] + masked_noise_weights * masked_noise,
                    dim=-1,
                    keepdim=True
                ) / (entropy(masked_action_masks[current_step - 1], dim=-1, keepdim=True) + eps)
            kl_divs_with_noise.sum().backward()
            if NOISE_TYPE == 'linear':
                clip_val = 0.25 / NOISE_OPT_LR
            elif NOISE_TYPE == 'sigmoid':
                clip_val = 1. / NOISE_OPT_LR
            else:
                raise ValueError(f'Unrecognized NOISE_TYPE: {NOISE_TYPE}')
            torch.nn.utils.clip_grad_value_(self.noise_weight_logits, clip_val)
            noise_grads.append(self.noise_weight_logits.grad.numpy().ravel())
            self.noise_opt.step()
            if NOISE_TYPE == 'linear':
                with torch.no_grad():
                    self.noise_weight_logits.clamp_(MIN_NOISE, MAX_NOISE)
            elif NOISE_TYPE == 'sigmoid':
                pass
            else:
                raise ValueError(f'Unrecognized NOISE_TYPE: {NOISE_TYPE}')

        with torch.no_grad():
            kl_divs_with_noise = kl_divergence(
                self.all_actions[:current_step],
                (1. - self.noise_weights) * self.all_policy_preds[:current_step] + self.noise_weights * all_steps_noise,
                dim=-1,
                keepdim=True,
                eps=0.
            ) / (entropy(self.all_action_masks[:current_step], dim=-1, keepdim=True) + eps)
            kl_divs_with_noise = torch.where(
                torch.isinf(kl_divs_with_noise),
                torch.ones_like(kl_divs_with_noise),
                kl_divs_with_noise
            )
            printable_noise_grads = np.where(
                update_mask,
                np.stack(noise_grads, axis=0).mean(axis=0),
                float('nan')
            )
            printable_noise_weights_logits = np.where(
                update_mask,
                self.noise_weight_logits.numpy().ravel(),
                float('nan')
            )
            info += f'Noise gradients: {print_array_one_line(printable_noise_grads)} '
            info += f'Noise logits: {print_array_one_line(printable_noise_weights_logits)} '
            info += f'Noise weights: {print_array_one_line(self.noise_weights.view(-1).numpy())} '
        info += f'Pre-noise KL-divergences: {print_array_one_line(overall_kl_divs.mean(dim=0).numpy().ravel())} '
        info += f'Post-noise KL-divergences: {print_array_one_line(kl_divs_with_noise.mean(dim=0).numpy().ravel())}'
        return info

    def batch_actor_critic_func(self, state_batch: List[STATE_TYPE]) -> Tuple[np.ndarray, np.ndarray]:
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
            probs = F.softmax(logits * POLICY_TEMP, dim=-1).numpy().astype(np.float)
            policy_noise = np.ones((1, 1, 4))
            policy_noise = policy_noise / policy_noise.sum(axis=-1, keepdims=True)
            noise_weights = self.noise_weights.numpy().astype(np.float)
            probs = (1. - noise_weights) * probs + noise_weights * policy_noise
        # Score the dead geese
        dead_geese_mask = ~np.array(still_alive_list)
        # This doesn't work in Kaggle environment:
        # agent_rankings = stats.rankdata(np.array(rewards_list), method='average', axis=-1) - 1.
        agent_rankings = np.stack(
            [stats.rankdata(r, method='average') for r in rewards_list],
            axis=0
        ) - 1.
        agent_rankings_rescaled = 2. * agent_rankings / (n_geese - 1.) - 1.

        value_noise = dead_geese_mask.astype(np.float).sum(axis=-1, keepdims=True) / (n_geese - 1.)
        noise_weights = noise_weights.squeeze(-1)
        final_values = np.where(
            dead_geese_mask,
            agent_rankings_rescaled,
            values.numpy()
        )
        final_values = (1. - noise_weights) * final_values + noise_weights * value_noise

        # Rescale values to [-1, 1] based on the number of geese alive at the start of the turn
        # This can improve the MCTS efficiency when there are fewer geese still alive
        if RESCALE_VALUES:
            n_geese_alive = self.still_alive.sum()
            if n_geese_alive == 2:
                # Rescale from [1/3, 1] -> [-1, 1]
                final_values = final_values * 3. - 2.
            elif n_geese_alive == 3:
                # Rescale from [-1/3, 1] -> [-1, 1]
                final_values = final_values * 1.5 - 0.5

        # Probs should be of shape (n_envs, n_geese, 4)
        # Values should be of shape (n_envs, n_geese, 1)
        return probs, np.expand_dims(final_values, axis=-1)

    def actor_critic_func(self, state: STATE_TYPE) -> Tuple[np.ndarray, np.ndarray]:
        probs, values = self.batch_actor_critic_func([state])
        return probs.squeeze(0), values.squeeze(0)

    def terminal_value_func(self, state: STATE_TYPE) -> np.ndarray:
        agent_rankings = stats.rankdata([agent['reward'] for agent in state], method='average') - 1.
        ranks_rescaled = 2. * agent_rankings / (len(state) - 1.) - 1.
        # Rescale values to [-1, 1] based on the number of geese alive at the start of the turn
        # This can improve the MCTS efficiency when there are fewer geese still alive
        if RESCALE_VALUES:
            n_geese_alive = self.still_alive.sum()
            if n_geese_alive == 2:
                # Rescale from [1/3, 1] -> [-1, 1]
                ranks_rescaled = ranks_rescaled * 3. - 2.
            elif n_geese_alive == 3:
                # Rescale from [-1/3, 1] -> [-1, 1]
                ranks_rescaled = ranks_rescaled * 1.5 - 0.5
        return ranks_rescaled

    @property
    def noise_weights(self) -> torch.Tensor:
        if NOISE_TYPE == 'linear':
            noise = self.noise_weight_logits
        elif NOISE_TYPE == 'sigmoid':
            noise = scaled_sigmoid(self.noise_weight_logits, MIN_NOISE, MAX_NOISE)
        else:
            raise ValueError(f'Unrecognized NOISE_TYPE: {NOISE_TYPE}')
        return torch.where(
            (torch.arange(self.n_geese) != self.index).view(self.noise_weight_logits.shape),
            noise,
            torch.zeros_like(self.noise_weight_logits)
        )


AGENT = None


def call_agent(obs: Dict, conf: Dict):
    global AGENT

    obs = Observation(obs)
    conf = Configuration(conf)
    if AGENT is None:
        AGENT = Agent(obs, conf)

    return AGENT(obs, conf)
