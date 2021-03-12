import copy
import json
import torch.multiprocessing as mp
import numpy as np
import os
from pathlib import Path
from scipy import stats
import time
import torch
from typing import *

from ...env import goose_env as ge
from ...env.lightweight_env import LightweightEnv, make
from ...mcts.basic_mcts import BasicMCTS
from ...mcts.utils import terminal_value_func, batch_actor_critic_factory
from ...models import FullConvActorCriticNetwork
from ...utils import ActionMasking


def alphagoose_data_generator_worker(
        worker_id: int,
        model_kwargs: Dict,
        device: torch.device,
        n_envs_per_worker: int,
        worker_data_dir: Path,
        weights_dir: Path,
        obs_type: ge.ObsType,
        max_saved_steps: int,
        model_reload_freq: int,
        n_iter: int,
        float_precision: torch.dtype = torch.float32,
        **mcts_kwargs
):
    # For whatever reason, this sleep statement helps prevent CUDNN_NOT_INITIALIZED errors
    time.sleep(worker_id / 5.)
    # Create environments
    envs = [make() for _ in range(n_envs_per_worker)]
    for env in envs:
        env.reset()
    search_trees = [BasicMCTS(
        action_mask_func=ActionMasking.LETHAL.get_action_mask,
        actor_critic_func=lambda x: None,
        terminal_value_func=terminal_value_func,
        **mcts_kwargs
    ) for _ in range(n_envs_per_worker)]
    available_actions_masks = [[] for _ in range(n_envs_per_worker)]
    pre_search_policies = [[] for _ in range(n_envs_per_worker)]
    post_search_policies = [[] for _ in range(n_envs_per_worker)]
    # Create model and load weights
    model = FullConvActorCriticNetwork(**model_kwargs)
    model.to(device=device, dtype=float_precision)
    current_weights_path = get_most_recent_weights_file(weights_dir)
    model.load_state_dict(torch.load(current_weights_path, map_location=device))
    model.eval()
    # Load actor_critic_func
    batch_actor_critic_func = batch_actor_critic_factory(model, obs_type, float_precision)
    n_saved_steps = 0
    while True:
        for steps_since_reload in range(model_reload_freq):
            for env_idx, env in enumerate(envs):
                if env.done:
                    save_start_time = time.time()
                    n_saved_steps = save_episode_steps(
                        env,
                        available_actions_masks[env_idx],
                        pre_search_policies[env_idx],
                        post_search_policies[env_idx],
                        worker_data_dir,
                        n_saved_steps,
                        max_saved_steps
                    )
                    print(f'{worker_id}: Saved train examples in {time.time() - save_start_time:.2} seconds')
                    env.reset()
                    available_actions_masks[env_idx] = []
                    pre_search_policies[env_idx] = []
                    post_search_policies[env_idx] = []
            step_start_time = time.time()
            # n_iter + 1 because the first iteration creates the root node
            for i in range(n_iter + 1):
                state_batch, trajectory_batch, done_batch, still_alive_batch, available_actions_batch = zip(
                    *[st.expand(env.lightweight_clone()) for st, env in zip(search_trees, envs)]
                )
                policies_batch, values_batch = batch_actor_critic_func(state_batch, device)

                for idx, (env, search_tree) in enumerate(zip(envs, search_trees)):
                    backprop_kwargs = dict(
                        trajectory=trajectory_batch[idx],
                        still_alive=still_alive_batch[idx],
                        available_actions=available_actions_batch[idx]
                    )
                    if done_batch[idx]:
                        search_tree.backpropagate(
                            policy_est=None,
                            value_est=terminal_value_func(state_batch[idx]),
                            **backprop_kwargs
                        )
                    else:
                        search_tree.backpropagate(
                            policy_est=policies_batch[idx],
                            value_est=values_batch[idx],
                            **backprop_kwargs
                        )
            for idx, (env, search_tree, available_actions_list, pre_policy_list, post_policy_list) in enumerate(zip(
                    envs, search_trees, available_actions_masks, pre_search_policies, post_search_policies
            )):
                root_node = search_tree.get_root_node(env)
                # Booleans are not JSON serializable
                available_actions_list.append(root_node.available_actions_masks.astype(np.float))
                pre_policy_list.append(root_node.initial_policies)
                post_policy_list.append(root_node.get_improved_policies(temp=1.))
                actions = root_node.get_improved_actions(temp=0.)
                env.step(actions)
                search_tree.reset()
            print(f'{worker_id}: Finished step {steps_since_reload} in {time.time() - step_start_time:.2f} seconds')
        # Reload the model weights if a new trained model is available
        latest_weights_path = get_most_recent_weights_file(weights_dir)
        if current_weights_path != latest_weights_path:
            reload_start_time = time.time()
            current_weights_path = latest_weights_path
            model.load_state_dict(torch.load(current_weights_path, map_location=device))
            model.eval()
            print(f'{worker_id}: Reloaded model in {time.time() - reload_start_time:.2f} seconds')


def save_episode_steps(
        env: LightweightEnv,
        available_actions_masks: List[np.ndarray],
        pre_search_policies: List[np.ndarray],
        post_search_policies: List[np.ndarray],
        worker_data_dir: Path,
        n_saved_steps: int,
        max_saved_steps,
) -> int:
    # Save the episode steps to disk
    game_score = np.array([agent['reward'] for agent in env.steps[-1]])
    agent_rankings = stats.rankdata(game_score, method='average') - 1.
    for step_idx, step in enumerate(env.steps[:-1]):
        for agent_idx, agent in enumerate(step):
            agent['final_rank'] = agent_rankings[agent_idx]
            if agent['status'] == 'ACTIVE':
                agent['available_actions_mask'] = list(available_actions_masks[step_idx][agent_idx])
                agent['initial_policy'] = list(pre_search_policies[step_idx][agent_idx])
                agent['policy'] = list(post_search_policies[step_idx][agent_idx])
        with open(worker_data_dir / f'{int(n_saved_steps)}.json', 'w') as f:
            json.dump(step, f)
        n_saved_steps = (n_saved_steps + 1) % max_saved_steps
    return n_saved_steps


def get_most_recent_weights_file(weights_dir: Path) -> Path:
    all_weight_files = list(weights_dir.glob('*.pt'))
    all_weight_files.sort(key=lambda f: int(f.stem))

    return all_weight_files[-1]


def multiprocess_alphagoose_data_generator(
        n_workers: int,
        device: torch.device,
        data_dir: Path,
        max_saved_steps: int,
        **data_generator_kwargs
):
    mp.set_start_method('spawn')
    os.environ['OMP_NUM_THREADS'] = '1'
    if data_dir.exists() and any(Path(data_dir).iterdir()):
        raise RuntimeError(f'data_dir already exists and is not empty: {data_dir}')
    else:
        data_dir.mkdir(exist_ok=True)
    assert max_saved_steps % n_workers == 0

    processes = []
    for rank in range(n_workers):
        worker_data_dir = data_dir / str(rank)
        worker_data_dir.mkdir()
        data_generator_kwargs['worker_id'] = rank
        data_generator_kwargs['device'] = device
        data_generator_kwargs['worker_data_dir'] = worker_data_dir
        data_generator_kwargs['max_saved_steps'] = max_saved_steps / n_workers
        p = mp.Process(
            target=alphagoose_data_generator_worker,
            kwargs=copy.deepcopy(data_generator_kwargs)
        )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
