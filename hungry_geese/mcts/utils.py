import numpy as np
from scipy import stats
import torch
import torch.nn.functional as F
from typing import *

from ..env import goose_env as ge
from hungry_geese.nns.models import FullConvActorCriticNetwork


def terminal_value_func(state: List[Dict]):
    agent_rankings = stats.rankdata([agent['reward'] for agent in state], method='average') - 1.
    ranks_rescaled = 2. * agent_rankings / (len(state) - 1.) - 1.
    return ranks_rescaled


def actor_critic_factory(model: FullConvActorCriticNetwork, obs_type: ge.ObsType):
    def actor_critic_func(state: List[Dict]):
        geese = state[0]['observation']['geese']
        n_geese = len(geese)

        obs = ge.create_obs_tensor(state, obs_type)
        head_locs = [goose[0] if len(goose) > 0 else -1 for goose in geese]
        still_alive = [agent['status'] == 'ACTIVE' for agent in state]
        with torch.no_grad():
            logits, values = model(torch.from_numpy(obs),
                                   torch.tensor(head_locs).unsqueeze(0),
                                   torch.tensor(still_alive).unsqueeze(0))
            probs = F.softmax(logits, dim=-1)

        # Score the dead geese
        dead_geese_mask = ~np.array(still_alive)
        agent_rankings = stats.rankdata([agent['reward'] for agent in state], method='average') - 1.
        agent_rankings_rescaled = 2. * agent_rankings / (n_geese - 1.) - 1.

        final_values = np.where(
            dead_geese_mask,
            agent_rankings_rescaled,
            values.squeeze(0).numpy()
        )

        # Logits should be of shape (n_geese, 4)
        # Values should be of shape (n_geese, 1)
        return probs.squeeze(0).numpy().astype(np.float), final_values[:, np.newaxis]

    return actor_critic_func


def batch_actor_critic_factory(model: FullConvActorCriticNetwork,
                               obs_type: ge.ObsType,
                               float_precision: torch.dtype):
    def batch_actor_critic_func(states: List[List[Dict]], device: torch.device):
        obs_list = []
        head_locs_list = []
        still_alive_list = []
        rewards_list = []
        n_geese = len(states[0][0]['observation']['geese'])
        for state in states:
            geese = state[0]['observation']['geese']
            assert len(geese) == n_geese, 'All environments must have the same number of geese for batching'

            obs_list.append(ge.create_obs_tensor(state, obs_type))
            head_locs_list.append([goose[0] if len(goose) > 0 else -1 for goose in geese])
            still_alive_list.append([agent['status'] == 'ACTIVE' for agent in state])
            rewards_list.append([agent['reward'] for agent in state])
        with torch.no_grad():
            logits, values = model(
                torch.from_numpy(np.concatenate(obs_list, axis=0)).to(device=device, dtype=float_precision),
                torch.tensor(head_locs_list).to(device=device),
                torch.tensor(still_alive_list).to(device=device)
            )
            probs = F.softmax(logits, dim=-1).cpu()
        # Score the dead geese
        dead_geese_mask = ~np.array(still_alive_list)
        agent_rankings = stats.rankdata(np.array(rewards_list), method='average', axis=-1) - 1.
        agent_rankings_rescaled = 2. * agent_rankings / (n_geese - 1.) - 1.

        final_values = np.where(
            dead_geese_mask,
            agent_rankings_rescaled,
            values.cpu().numpy()
        )
        # Logits should be of shape (n_envs, n_geese, 4)
        # Values should be of shape (n_envs, n_geese, 1)
        return probs.numpy().astype(np.float), np.expand_dims(final_values, axis=-1)

    return batch_actor_critic_func


def torch_actor_critic_factory(model: FullConvActorCriticNetwork) -> Callable:
    def torch_actor_critic_func(
            obs: torch.Tensor,
            head_locs: torch.Tensor,
            still_alive: torch.Tensor,
            rewards: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_geese = head_locs.shape[1]
        with torch.no_grad():
            logits, values = model(obs, head_locs, still_alive)
            probs = F.softmax(logits, dim=-1)
        # Score the dead geese
        dead_geese_mask = ~still_alive
        agent_rankings = rankdata_average(rewards) - 1.
        agent_rankings_rescaled = 2. * agent_rankings / (n_geese - 1.) - 1.

        final_values = torch.where(
            dead_geese_mask,
            agent_rankings_rescaled,
            values
        )
        # Logits should be of shape (n_envs, n_geese, 4)
        # Values should be of shape (n_envs, n_geese)
        return probs, final_values

    return torch_actor_critic_func


def torch_terminal_value_func(rewards: torch.Tensor) -> torch.Tensor:
    n_geese = rewards.shape[1]
    agent_rankings = rankdata_average(rewards) - 1.
    agent_rankings_rescaled = 2. * agent_rankings / (n_geese - 1.) - 1.
    return agent_rankings_rescaled


def rankdata_average(a: torch.Tensor) -> torch.Tensor:
    assert a.ndim == 2
    arr = a.clone()
    sorter = torch.argsort(arr, dim=-1)
    inv = torch.empty_like(sorter)
    inv.scatter_(-1, sorter, torch.arange(sorter.shape[-1], device=arr.device).unsqueeze(0).expand_as(sorter))

    arr = arr.gather(-1, sorter)
    obs = torch.cat([
        torch.ones((arr.shape[0], 1), dtype=torch.bool, device=arr.device),
        arr[:, 1:] != arr[:, :-1]
    ], dim=-1).to(torch.int64)
    dense = obs.cumsum(dim=-1).gather(-1, inv)
    """
    We need to take the rowwise indices of the nonzero elements. However, the number of nonzero elements may vary from
    row to row, so we pad the rows with additional elements so that the 1D nonzero indices can be reshaped to a matrix
    of shape (n_rows, n_cols). Note that this will not affect the final result, as any additional nonzero elements
    will not be gathered by the dense indices.
    """
    padding = torch.where(
        (obs == 0).sum(dim=-1, keepdim=True) > torch.arange(arr.shape[-1], device=arr.device).unsqueeze(0),
        torch.ones_like(obs),
        torch.zeros_like(obs)
    )
    # cumulative counts of each unique value
    count = torch.nonzero(torch.cat([obs, padding], dim=-1), as_tuple=True)[1].view(*arr.shape)
    count = torch.cat([count, torch.zeros((arr.shape[0], 1), dtype=count.dtype, device=arr.device)], dim=-1)
    count.scatter_(
        -1,
        (obs != 0).sum(dim=-1, keepdim=True),
        torch.zeros((arr.shape[0], 1), dtype=count.dtype, device=arr.device) + arr.shape[-1]
    )

    return 0.5 * (count.gather(-1, dense) + count.gather(-1, dense - 1) + 1)
