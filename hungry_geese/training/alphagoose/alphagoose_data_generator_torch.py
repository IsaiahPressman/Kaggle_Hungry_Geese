from pathlib import Path
import time
import torch
import torch.multiprocessing as mp
from typing import *

from ...env.torch_env import TorchEnv
from ...mcts.torch_mcts import TorchMCTS
from ...mcts.utils import torch_actor_critic_factory, torch_terminal_value_func, rankdata_average
from hungry_geese.nns.models import FullConvActorCriticNetwork
from .alphagoose_data_generator import reload_model_weights, save_episodes_worker


def main_env_worker(
        shared_env: TorchEnv,
        shared_policies: torch.Tensor,
        save_episode_queue: mp.Queue,
        search_finished: mp.Barrier,
        policy_updated: mp.Barrier,
        step_taken: mp.Barrier,
) -> NoReturn:
    try:
        all_episodes = [[] for _ in range(shared_env.n_envs)]
        finished_episodes = [[] for _ in range(shared_env.n_envs)]
        dones_before_reset = shared_env.dones.cpu()
        while True:
            obs_dict_start_time = time.time()
            all_obs_dicts = shared_env.generate_obs_dicts()
            all_policies = shared_policies.cpu()
            for env_idx in range(shared_env.n_envs):
                # Get pointer to the last step to add the policies to
                if len(all_episodes[env_idx]) > 0:
                    # Usually, it is a step in the current environment
                    step = all_episodes[env_idx][-1]
                elif finished_episodes[env_idx]:
                    # Sometimes, it's an episode that just finished during the last step
                    step = finished_episodes[env_idx][-1]
                else:
                    # For the very first step of the worker, there is no last environment, so we just pass an
                    # empty list that will be discarded
                    step = [{} for _ in range(shared_env.n_geese)]
                for g in range(shared_env.n_geese):
                    if step[g].get('status') == 'ACTIVE':
                        step[g].update({
                            'available_actions_mask': [1] * 4,
                            'policy': all_policies[env_idx, g].tolist()
                        })
                all_episodes[env_idx].append(all_obs_dicts[env_idx])
            for env_idx in torch.arange(shared_env.n_envs)[dones_before_reset]:
                save_episode_queue.put_nowait(finished_episodes[env_idx])
                finished_episodes[env_idx] = []
            print(f'Finished generating obs_dict in {time.time() - obs_dict_start_time:.2f} seconds')
            torch.cuda.synchronize(shared_env.device)
            search_finished.wait()

            policy_updated.wait()

            policy_max = shared_policies.max(dim=-1, keepdim=True).values
            actions = torch.multinomial(
                (shared_policies == policy_max).view(-1, 4).to(torch.float32),
                1
            ).view(-1, shared_env.n_geese)
            shared_env.step(torch.where(
                shared_env.alive,
                actions,
                torch.zeros_like(actions)
            ))
            dones_before_reset = shared_env.dones.cpu().clone()
            if dones_before_reset.any():
                all_agent_rankings = rankdata_average(shared_env.rewards).cpu()
                for env_idx in torch.arange(shared_env.n_envs)[dones_before_reset]:
                    finished_episodes[env_idx] = all_episodes[env_idx]
                    all_episodes[env_idx] = []
                    for step_idx, step in enumerate(finished_episodes[env_idx]):
                        for goose_idx, goose in enumerate(step):
                            goose['final_rank'] = all_agent_rankings[env_idx, goose_idx].item() - 1.
                shared_env.reset()
            torch.cuda.synchronize(shared_env.device)
            step_taken.wait()
    finally:
        # Delete references to shared CUDA tensors before exiting
        del shared_env, shared_policies
        print('main_env_worker exited gracefully')


def mcts_worker(
        shared_env: TorchEnv,
        shared_policies: torch.Tensor,
        search_finished: mp.Barrier,
        policy_updated: mp.Barrier,
        step_taken: mp.Barrier,
        model_kwargs: Dict,
        weights_dir: Path,
        update_model_freq: int,
        device: torch.device,
        env_kwargs: Dict,
        mcts_kwargs: Dict
) -> NoReturn:
    try:
        # Create model and load weights
        model = FullConvActorCriticNetwork(**model_kwargs)
        model.to(device=device)
        model.eval()
        current_weights_path = reload_model_weights(model, weights_dir, None, device)

        # Duplicate environment
        local_env = TorchEnv(**env_kwargs)
        local_env.copy_data_from(shared_env)

        # Create MCTS object
        mcts = TorchMCTS(
            torch_actor_critic_factory(model),
            torch_terminal_value_func,
            **mcts_kwargs
        )

        step_counter = 0
        while True:
            step_start_time = time.time()
            # Perform search
            mcts.reset()
            mcts.run_mcts(shared_env, local_env)
            print(f'Finished MCTS in {time.time() - step_start_time:.2f} seconds')
            torch.cuda.synchronize(shared_env.device)
            search_finished.wait()

            # Update policies
            shared_policies[:] = mcts.tree.get_improved_policies()
            torch.cuda.synchronize(shared_env.device)
            policy_updated.wait()

            # Update model every update_model_freq iterations
            step_counter = (step_counter + 1) % update_model_freq
            if step_counter == 0:
                current_weights_path = reload_model_weights(model, weights_dir, current_weights_path, device)
            torch.cuda.synchronize(shared_env.device)
            step_taken.wait()
            print(f'Finished step {step_counter} in {time.time() - step_start_time:.2f} seconds')
    finally:
        # Delete references to shared CUDA tensors before exiting
        del shared_env, shared_policies
        print('mcts_search_worker exited gracefully')


def start_alphagoose_data_generator(
        dataset_dir: Path,
        weights_dir: Path,
        allow_resume: bool,
        max_saved_batches: int,
        update_model_freq: int,
        env_kwargs: Dict,
        model_kwargs: Dict,
        mcts_kwargs: Dict
) -> NoReturn:
    # Check that dataset_dir exists and is empty
    if dataset_dir.exists() and any(list(dataset_dir.iterdir())):
        if not allow_resume:
            raise RuntimeError(f'dataset_dir already exists and is not empty: {dataset_dir}')
        # Check that the directory only contains replay data files
        all_files = [f for f in dataset_dir.iterdir() if not f.stem.startswith('.')]
        if not all([f.suffix == '.ljson' for f in all_files]):
            raise RuntimeError(f'dataset_dir already exists and contains non-".ljson" files: {dataset_dir}')
        # Check that the directory does not contain more replay data files than allowed by max_saved_batches
        all_files.sort(key=lambda f: int(f.stem))
        if int(all_files[-1].stem) >= max_saved_batches:
            raise RuntimeError(f'dataset_dir already exists and contains files with id > max_saved_batches: '
                               f'{dataset_dir} - {all_files[-1].stem} > {max_saved_batches}')
        # Find the index to restart at
        all_files.sort(key=lambda f: f.stat().st_mtime)
        start_idx = (int(all_files[-1].stem) + 1) % max_saved_batches
        print(f'Resuming data generation. Latest replay file: {all_files[-1].name}')
    else:
        start_idx = 0
    dataset_dir.mkdir(exist_ok=True)

    shared_env = TorchEnv(**env_kwargs)
    shared_policies = torch.zeros(
        (shared_env.n_envs, shared_env.n_geese, 4),
        dtype=torch.float32,
        device=shared_env.device
    )

    save_episode_queue = mp.Queue()
    search_finished = mp.Barrier(2)
    policy_updated = mp.Barrier(2)
    step_taken = mp.Barrier(2)

    # Start the main environment process
    main_env_proc = mp.Process(
        target=main_env_worker,
        kwargs=dict(
            shared_env=shared_env,
            shared_policies=shared_policies,
            save_episode_queue=save_episode_queue,
            search_finished=search_finished,
            policy_updated=policy_updated,
            step_taken=step_taken
        )
    )
    main_env_proc.start()

    # Start the MCTS process
    mcts_proc = mp.Process(
        target=mcts_worker,
        kwargs=dict(
            shared_env=shared_env,
            shared_policies=shared_policies,
            search_finished=search_finished,
            policy_updated=policy_updated,
            step_taken=step_taken,
            model_kwargs=model_kwargs,
            weights_dir=weights_dir,
            update_model_freq=update_model_freq,
            device=shared_env.device,
            env_kwargs=env_kwargs,
            mcts_kwargs=mcts_kwargs
        )
    )
    mcts_proc.start()

    # Start the episode saving process
    save_steps_proc = mp.Process(
        target=save_episodes_worker,
        args=(dataset_dir, save_episode_queue, max_saved_batches, start_idx)
    )
    save_steps_proc.daemon = True
    save_steps_proc.start()

    n_iter = mcts_kwargs.get('n_iter')
    print(f'Running {shared_env.n_envs} environments with {n_iter} iterations of MCTS on device: {shared_env.device}.')

    try:
        main_env_proc.join()
        mcts_proc.join()
    except KeyboardInterrupt:
        print('KeyboardInterrupt...')
    finally:
        search_finished.abort()
        policy_updated.abort()
        step_taken.abort()
        main_env_proc.join()
        mcts_proc.join()
