import copy
from filelock import FileLock
try:
    import ujson as json
except ModuleNotFoundError:
    import json
import torch.multiprocessing as mp
import numpy as np
import os
from pathlib import Path
from scipy import stats
import time
import torch
from torch import nn
from typing import *

from ...env import goose_env as ge
from ...env.lightweight_env import LightweightEnv, make
from ...mcts.basic_mcts import BasicMCTS
from ...mcts.utils import terminal_value_func, batch_actor_critic_factory
from hungry_geese.nns.models import FullConvActorCriticNetwork
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
    current_weights_path = get_latest_weights_file(weights_dir)
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
        reload_model_weights(model, weights_dir, current_weights_path, device)


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


def reload_model_weights(
        model: nn.Module,
        weights_dir: Path,
        current_weights_path: Optional[Path],
        device: torch.device
) -> Path:
    # Reload the model weights if a new trained model is available
    latest_weights_path = get_latest_weights_file(weights_dir)
    if current_weights_path != latest_weights_path:
        reload_start_time = time.time()
        try:
            model.load_state_dict(torch.load(latest_weights_path, map_location=device))
        # In case the model weights are being saved at the same moment that they are being reloaded
        except EOFError:
            time.sleep(0.5)
            model.load_state_dict(torch.load(latest_weights_path, map_location=device))
        model.eval()
        print(f'Loaded model weights from {latest_weights_path.name} in '
              f'{time.time() - reload_start_time:.2f} seconds')
    return latest_weights_path


def get_latest_weights_file(weights_dir: Path) -> Path:
    all_weight_files = list(weights_dir.glob('*.pt'))
    all_weight_files.sort(key=lambda f: int(f.stem))

    if len(all_weight_files) == 0:
        raise FileNotFoundError(f'No .pt weight files found in {weights_dir}')
    return all_weight_files[-1]


def save_episodes_worker(
        dataset_dir: Path,
        save_episode_queue: mp.Queue,
        max_saved_episodes: int,
        start_idx: int = 0
) -> NoReturn:
    saved_episode_counter = start_idx
    episode_batch = []
    while True:
        episode_batch.append(save_episode_queue.get())
        if len(episode_batch) >= 1:
            with FileLock(str(dataset_dir) + '.lock'):
                save_start_time = time.time()
                # Empty queue items that arrived while waiting for the lock
                while not save_episode_queue.empty():
                    episode_batch.append(save_episode_queue.get())
                for episode in episode_batch:
                    with open(dataset_dir / f'{saved_episode_counter}.ljson', 'w') as f:
                        f.writelines([json.dumps(step) + '\n' for step in episode])
                    saved_episode_counter = (saved_episode_counter + 1) % max_saved_episodes
            print(f'Saved {len(episode_batch)} batch{"es" if len(episode_batch) != 1 else ""} in '
                  f'{time.time() - save_start_time:.2} seconds')
            episode_batch = []


def multiprocess_alphagoose_data_generator(
        n_workers: int,
        dataset_dir: Path,
        max_saved_episodes: int,
        **data_generator_kwargs
):
    mp.set_start_method('spawn')
    os.environ['OMP_NUM_THREADS'] = '1'
    if dataset_dir.exists() and any(list(dataset_dir.iterdir())):
        raise RuntimeError(f'dataset_dir already exists and is not empty: {dataset_dir}')
    dataset_dir.mkdir(exist_ok=True)

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

    save_episodes_worker(
        dataset_dir,
        save_episode_queue,
        max_saved_episodes
    )
