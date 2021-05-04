import torch
import tqdm
from typing import *

from ..config import N_PLAYERS
from ..env.torch_env import TorchEnv


class TorchMCTSTree:
    def __init__(
            self,
            n_envs: int,
            max_size: int,
            n_geese: int = N_PLAYERS,
            device: torch.device = torch.device('cuda')
    ):
        self.n_envs = n_envs
        self.max_size = max_size
        self.n_geese = n_geese
        self.device = device

        int_tensor_kwargs = dict(
            dtype=torch.int64,
            device=self.device
        )
        float_tensor_kwargs = dict(
            dtype=torch.float32,
            device=self.device
        )
        self.parents = torch.empty((self.n_envs, self.max_size), **int_tensor_kwargs)
        self.children = torch.empty((self.n_envs, self.max_size, 4 ** self.n_geese), **int_tensor_kwargs)
        self.depths = torch.empty((self.n_envs, self.max_size), **int_tensor_kwargs)
        self.actions = torch.empty((self.n_envs, self.max_size, self.n_geese), **int_tensor_kwargs)
        self.is_leaf = torch.empty((self.n_envs, self.max_size), dtype=torch.bool, device=self.device)
        self.visits = torch.empty((self.n_envs, self.max_size, self.n_geese, 4), **float_tensor_kwargs)
        self.q_vals = torch.empty((self.n_envs, self.max_size, self.n_geese, 4), **float_tensor_kwargs)
        self.init_pi = torch.empty((self.n_envs, self.max_size, self.n_geese, 4), **float_tensor_kwargs)
        self.ptrs = torch.empty((self.n_envs,), **int_tensor_kwargs)
        self.sizes = torch.empty((self.n_envs,), **int_tensor_kwargs)

        self.env_idxs = torch.arange(self.n_envs, device=self.device)
        self.moves_to_key_vals = 4 ** torch.arange(4, device=self.device).unsqueeze(0)
        self.reset()

    def reset(self) -> NoReturn:
        self.parents[:] = -1
        self.children[:] = 0
        self.depths[:] = 0
        self.is_leaf[:] = True
        self.actions[:] = -1
        self.visits[:] = 0.0
        self.q_vals[:] = 0.0
        self.init_pi[:] = 0.0
        self.ptrs[:] = 0
        self.sizes[:] = 1

    def moves_to_key(self, moves: torch.Tensor) -> torch.Tensor:
        return (moves * self.moves_to_key_vals).sum(axis=-1)

    def go_to_parents(self, envs_mask: torch.Tensor) -> NoReturn:
        assert (self.ptrs[envs_mask] >= 0).all()
        self.ptrs[envs_mask] = self.parents[self.env_idxs[envs_mask], self.ptrs[envs_mask]]

    def go_to_or_make_children(self, actions: torch.Tensor, envs_mask: torch.Tensor) -> NoReturn:
        # Get children
        keys = self.moves_to_key(actions)
        child_ptrs_masked = self.children[
            self.env_idxs[envs_mask],
            self.ptrs[envs_mask],
            keys[envs_mask]
        ]
        child_exists = child_ptrs_masked > 0
        self.ptrs[self.env_idxs[envs_mask][child_exists]] = child_ptrs_masked[child_exists]

        # Make children where they do not exist
        env_idxs_masked = self.env_idxs[envs_mask][~child_exists]
        ptrs_masked = self.ptrs[env_idxs_masked]
        child_ptrs_masked = self.sizes[env_idxs_masked]
        assert (child_ptrs_masked < self.max_size).all()

        self.sizes[env_idxs_masked] += 1
        self.children[
            env_idxs_masked,
            ptrs_masked,
            keys[env_idxs_masked]
        ] = child_ptrs_masked
        self.parents[
            env_idxs_masked,
            child_ptrs_masked
        ] = ptrs_masked
        self.depths[
            env_idxs_masked,
            child_ptrs_masked
        ] = self.depths[env_idxs_masked, ptrs_masked] + 1
        self.actions[
            env_idxs_masked,
            child_ptrs_masked
        ] = actions[env_idxs_masked]
        self.ptrs[env_idxs_masked] = child_ptrs_masked

    def go_to_root(self) -> NoReturn:
        self.ptrs[:] = 0

    def get_puct_actions(
            self,
            geese_still_playing: torch.Tensor,
            c_puct: float
    ) -> torch.Tensor:
        visits_masked = self.visits[
            self.env_idxs,
            self.ptrs
        ]
        values_masked = self.q_vals[
            self.env_idxs,
            self.ptrs
        ]
        init_pi_masked = self.init_pi[
            self.env_idxs,
            self.ptrs
        ]
        uct_vals = values_masked + (c_puct *
                                    init_pi_masked *
                                    torch.sqrt(visits_masked.sum(
                                        dim=-1,
                                        keepdims=True
                                    )) / (1. + visits_masked))
        """
        We could just go from here using a simple torch.argmax(uct_vals, dim=-1) to select actions,
        but this would always tiebreak towards actions with lower indices.
        Therefore, we sample from all actions whose uct value equals the max uct value for that agent.
        """
        uct_max = uct_vals.max(dim=-1, keepdims=True).values
        actions = torch.multinomial(
            (uct_vals == uct_max).view(-1, 4).to(torch.float32),
            1
        ).view(-1, self.n_geese)
        return torch.where(
            geese_still_playing,
            actions,
            torch.zeros_like(actions)
        )

    def set_policy_priors(self, policies: torch.Tensor) -> NoReturn:
        self.init_pi[self.env_idxs, self.ptrs] = policies

    def update_values(self, values: torch.Tensor, envs_mask: torch.Tensor) -> NoReturn:
        env_idxs_masked = self.env_idxs[envs_mask]
        ptrs_masked = self.ptrs[envs_mask]
        env_idxs_masked_repeated = env_idxs_masked.repeat_interleave(4)
        parent_ptrs_masked_repeated = self.parents[env_idxs_masked, ptrs_masked].repeat_interleave(4)
        assert (parent_ptrs_masked_repeated > -1).all()
        goose_idxs_masked_repeated = torch.arange(self.n_geese, device=self.device).repeat(envs_mask.sum())
        actions_masked = self.actions[
            env_idxs_masked,
            ptrs_masked
        ].view(-1)
        q_vals_masked = self.q_vals[
            env_idxs_masked_repeated,
            parent_ptrs_masked_repeated,
            goose_idxs_masked_repeated,
            actions_masked
        ]
        visits_masked = self.visits[
            env_idxs_masked_repeated,
            parent_ptrs_masked_repeated,
            goose_idxs_masked_repeated,
            actions_masked
        ]
        self.q_vals[
            env_idxs_masked_repeated,
            parent_ptrs_masked_repeated,
            goose_idxs_masked_repeated,
            actions_masked
        ] = (visits_masked * q_vals_masked + values[envs_mask].view(-1)) / (visits_masked + 1.)
        self.visits[
            env_idxs_masked_repeated,
            parent_ptrs_masked_repeated,
            goose_idxs_masked_repeated,
            actions_masked
        ] += 1.

    def set_leaf_status(self, new_leaf_statuses: torch.Tensor) -> NoReturn:
        self.is_leaf[self.env_idxs, self.ptrs] = new_leaf_statuses

    @property
    def is_at_leaf(self) -> torch.Tensor:
        return self.is_leaf[self.env_idxs, self.ptrs]

    @property
    def is_at_root(self) -> torch.Tensor:
        return self.ptrs == 0

    def get_improved_policies(self, temp: float = 1.) -> torch.Tensor:
        assert temp >= 0.
        visits_indexed = self.visits[self.env_idxs, self.ptrs]
        if temp == 0.:
            max_visits = visits_indexed.max(dim=-1, keepdim=True)
            probs = (visits_indexed == max_visits).to(torch.float32)
        else:
            probs = torch.pow(visits_indexed, 1. / temp)
        return probs / probs.sum(dim=-1, keepdim=True)


class TorchMCTS:
    def __init__(
            self,
            actor_critic_func: Callable,
            terminal_value_func: Callable,
            c_puct: float = 1.,
            add_noise: bool = False,
            noise_val: float = 2.,
            noise_weight: float = 0.25,
            n_geese: int = N_PLAYERS,
            device: torch.device = torch.device('cuda'),
            **tree_kwargs
    ):
        self.actor_critic_func = actor_critic_func
        self.terminal_value_func = terminal_value_func
        self.c_puct = c_puct
        self.add_noise = add_noise
        self.noise_dist = torch.distributions.dirichlet.Dirichlet(
            torch.tensor([noise_val] * n_geese, dtype=torch.float32, device=device)
        )
        self.noise_weight = noise_weight
        self.tree = TorchMCTSTree(n_geese=n_geese, device=device, **tree_kwargs)

    def _expand_and_backpropagate(
            self,
            env_copy: TorchEnv,
            add_policy_noise: bool
    ) -> NoReturn:
        # Loop until every env is at a leaf state
        while not self.tree.is_at_leaf.all():
            actions = self.tree.get_puct_actions(env_copy.alive, self.c_puct)
            env_copy.step(actions, ~self.tree.is_at_leaf)
            self.tree.go_to_or_make_children(actions, ~self.tree.is_at_leaf)
        # Evaluate the leaf states
        policy_ests, values = self.actor_critic_func(
            env_copy.obs,
            env_copy.head_locs,
            env_copy.alive,
            env_copy.rewards
        )
        values = torch.where(
            env_copy.dones.unsqueeze(dim=-1),
            self.terminal_value_func(env_copy.rewards),
            values
        )
        # Store the policy priors
        if add_policy_noise:
            noise = self.noise_dist.sample((env_copy.n_envs,))
            policy_ests = (1. - self.noise_weight) * policy_ests + self.noise_weight * noise
        self.tree.set_policy_priors(policy_ests)
        # Set the visited non-terminal leaves to be expanded next time
        self.tree.set_leaf_status(env_copy.dones)
        # Backpropagate the values
        is_at_root = self.tree.is_at_root
        while not is_at_root.all():
            self.tree.update_values(values, ~is_at_root)
            self.tree.go_to_parents(~is_at_root)
            is_at_root = self.tree.is_at_root

    def run_mcts(
            self,
            env: TorchEnv,
            env_placeholder: TorchEnv,
            n_iter: int,
            show_progress: bool = False
    ) -> NoReturn:
        assert env.n_envs == self.tree.n_envs
        # The +1 is for the initial expansion of the root node
        if show_progress:
            iterator = tqdm.trange(n_iter + 1)
        else:
            iterator = range(n_iter + 1)
        for _ in iterator:
            env.copy_data_to(env_placeholder)
            self._expand_and_backpropagate(
                env_placeholder,
                self.add_noise and self.tree.is_at_root.all() and self.tree.is_at_leaf.all()
            )

    def reset(self) -> NoReturn:
        self.tree.reset()
