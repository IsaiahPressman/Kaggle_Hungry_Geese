import numpy as np
from scipy import stats
import torch
import torch.nn.functional as F
from typing import *

from ..env import goose_env as ge
from ..nns.models import FullConvActorCriticNetwork
from ..utils import STATE_TYPE, torch_rankdata_average


def terminal_value_func(state: STATE_TYPE):
    agent_rankings = stats.rankdata([agent['reward'] for agent in state], method='average') - 1.
    ranks_rescaled = 2. * agent_rankings / (len(state) - 1.) - 1.
    return ranks_rescaled


def actor_critic_factory(model: FullConvActorCriticNetwork, obs_type: ge.ObsType):
    def actor_critic_func(state: STATE_TYPE):
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
    def batch_actor_critic_func(states: List[STATE_TYPE], device: torch.device):
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
        agent_rankings = torch_rankdata_average(rewards) - 1.
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
