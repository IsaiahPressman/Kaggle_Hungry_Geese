import copy
from filelock import FileLock
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
        save_episode_queue: mp.Queue,
        model_kwargs: Dict,
        device: torch.device,
        n_envs_per_worker: int,
        weights_dir: Path,
        obs_type: ge.ObsType,
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
    post_search_policies = [[] for _ in range(n_envs_per_worker)]
    # Create model and load weights
    model = FullConvActorCriticNetwork(**model_kwargs)
    model.to(device=device, dtype=float_precision)
    current_weights_path = get_most_recent_weights_file(weights_dir)
    model.load_state_dict(torch.load(current_weights_path, map_location=device))
    model.eval()
    # Load actor_critic_func
    batch_actor_critic_func = batch_actor_critic_factory(model, obs_type, float_precision)
    while True:
        for steps_since_reload in range(model_reload_freq):
            for env_idx, env in enumerate(envs):
                if env.done:
                    save_episode_steps(
                        save_episode_queue,
                        env,
                        available_actions_masks[env_idx],
                        post_search_policies[env_idx],
                    )
                    env.reset()
                    available_actions_masks[env_idx] = []
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
            for idx, (env, search_tree, available_actions_list, post_policy_list) in enumerate(zip(
                    envs, search_trees, available_actions_masks, post_search_policies
            )):
                root_node = search_tree.get_root_node(env)
                # Booleans are not JSON serializable
                available_actions_list.append(root_node.available_actions_masks.astype(np.float))
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
            print(f'{worker_id}: Reloaded model {current_weights_path.name} in '
                  f'{time.time() - reload_start_time:.2f} seconds')


def save_episode_steps(
        save_episode_queue: mp.Queue,
        env: LightweightEnv,
        available_actions_masks: List[np.ndarray],
        post_search_policies: List[np.ndarray]
) -> NoReturn:
    # Send the episode steps to the writer to be saved to disk
    game_score = np.array([agent['reward'] for agent in env.steps[-1]])
    agent_rankings = stats.rankdata(game_score, method='average') - 1.
    episode = []
    for step_idx, step in enumerate(env.steps[:-1]):
        for agent_idx, agent in enumerate(step):
            agent['final_rank'] = agent_rankings[agent_idx]
            if agent['status'] == 'ACTIVE':
                agent['available_actions_mask'] = list(available_actions_masks[step_idx][agent_idx])
                agent['policy'] = list(post_search_policies[step_idx][agent_idx])
        episode.append(step)
    save_episode_queue.put_nowait(episode)


def get_most_recent_weights_file(weights_dir: Path) -> Path:
    all_weight_files = list(weights_dir.glob('*.pt'))
    all_weight_files.sort(key=lambda f: int(f.stem))

    if len(all_weight_files) == 0:
        raise FileNotFoundError(f'No .pt weight files found in {weights_dir}')
    return all_weight_files[-1]


def multiprocess_alphagoose_data_generator(
        n_workers: int,
        dataset_path: Path,
        max_saved_steps: int,
        **data_generator_kwargs
):
    mp.set_start_method('spawn')
    os.environ['OMP_NUM_THREADS'] = '1'
    if dataset_path.exists():
        raise RuntimeError(f'data_dir already exists: {dataset_path}')

    save_episode_queue = mp.Queue()
    processes = []
    for worker_id in range(n_workers):
        p = mp.Process(
            target=alphagoose_data_generator_worker,
            args=(worker_id, save_episode_queue),
            kwargs=copy.deepcopy(data_generator_kwargs)
        )
        p.daemon = True
        p.start()
        processes.append(p)

    all_steps = []
    steps_batch = []
    while True:
        steps_batch.extend(save_episode_queue.get())
        # Save steps in batches of 1000+
        if len(steps_batch) >= 1000:
            save_start_time = time.time()
            all_steps.extend(steps_batch)
            if len(all_steps) > max_saved_steps:
                all_steps = all_steps[-max_saved_steps:]
            with FileLock(str(dataset_path) + '.lock'):
                with open(dataset_path, 'w') as f:
                    json.dump(all_steps, f)
            print(f'Saved {len(steps_batch)} train examples in {time.time() - save_start_time:.2} seconds')
            steps_batch = []
