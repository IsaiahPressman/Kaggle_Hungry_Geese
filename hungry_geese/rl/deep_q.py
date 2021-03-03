import base64
import copy
import numpy as np
from pathlib import Path
import pickle
import shutil
import time
import torch
from torch.utils.tensorboard import SummaryWriter
import tqdm
from typing import *

# Custom imports
from ..env import goose_env as ge
from .replay_buffers import BasicReplayBuffer
from hungry_geese.models import DeepQNetwork


class DeepQ:
    def __init__(
            self,
            q_model: DeepQNetwork,
            env,
            replay_buffer: BasicReplayBuffer,
            validation_kwargs_dicts: Sequence[Dict] = (),
            device: torch.device = torch.device('cuda'),
            exp_folder: Path = Path('runs/awac/TEMP'),
            use_action_masking: bool = True,
            clip_grads: float = 10.,
            checkpoint_freq: int = 10,
            checkpoint_render_n_games: int = 10,
    ):
        self.q_model = q_model
        self.q_model.train()
        self.replay_buffer = replay_buffer
        self.validation_kwargs_dicts = validation_kwargs_dicts
        self.device = device
        self.exp_folder = exp_folder.absolute()
        if self.exp_folder.name == 'TEMP':
            print('WARNING: Using TEMP exp_folder')
            if self.exp_folder.exists():
                shutil.rmtree(self.exp_folder)
        elif self.exp_folder.exists() and any(Path(self.exp_folder).iterdir()):
            raise RuntimeError(f'Experiment folder {self.exp_folder} already exists and is not empty')
        else:
            print(f'Saving results to {self.exp_folder}')
        self.exp_folder.mkdir(exist_ok=True)
        self.use_action_masking = use_action_masking
        self.clip_grads = clip_grads
        self.checkpoint_freq = checkpoint_freq
        self.checkpoint_render_n_games = checkpoint_render_n_games

        self.env = env
        self.epoch_counter = 0
        self.batch_counter = 0
        self.train_step_counter = 0
        self.validation_counter = 0
        self.validation_step_counters = [0] * len(validation_kwargs_dicts)
        self.summary_writer = SummaryWriter(str(self.exp_folder))

        if self.clip_grads is not None:
            if self.clip_grads <= 0:
                raise ValueError(f'Should not clip gradients to <= 0, was {self.clip_grads} - '
                                 'pass None to clip_grads to not clip gradients')
            for p in self.q_model.parameters():
                if p.requires_grad:
                    p.register_hook(lambda grad: torch.clamp(grad, -self.clip_grads, self.clip_grads))

        dummy_input, *_ = self.env.hard_reset()
        dummy_input = torch.from_numpy(dummy_input).to(device=self.device)[0, 0].unsqueeze(0)
        self.summary_writer.add_graph(self.q_model, dummy_input)

    def train(self, batch_size, n_epochs, n_steps_per_epoch,
              train_frequency, n_train_batches, gamma=0.99, epsilon_scheduler=None):
        self.env.hard_reset()
        for epoch in range(n_epochs):
            print(f'Epoch #{epoch}:')
            print(f'Sampling {n_steps_per_epoch} time-steps from the environment and '
                  f'training on {n_train_batches} batches every {train_frequency} steps.')
            epoch_start_time = time.time()
            finished_episode_infos = []
            self.q_model.eval()
            s, r, done, info_dict = self.env.soft_reset()
            s = torch.from_numpy(s)
            done = torch.from_numpy(done)
            for step in tqdm.trange(n_steps_per_epoch):
                start_time = time.time()
                still_playing_mask = torch.logical_not(done)
                s_shape = s.shape
                if epsilon_scheduler is not None:
                    epsilon = epsilon_scheduler(self.train_step_counter)
                    epsilon_expanded = epsilon.expand(
                        self.env.n_envs,
                        self.env.n_players
                    ).reshape(-1, 1).to(device=self.device)
                else:
                    epsilon_expanded = None
                if self.use_action_masking:
                    available_actions_mask = torch.from_numpy(
                        info_dict['available_actions_mask']
                    ).to(device=self.device).view(-1, 4)
                else:
                    available_actions_mask = None
                a = self.q_model.sample_action(
                    s.to(device=self.device).view(-1, *s_shape[-3:]),
                    epsilon=epsilon_expanded,
                    available_actions_mask=available_actions_mask
                ).detach().cpu()
                # a = self.q_model.sample_action(s.to(device=self.device).view(-1, *s_shape[-3:])).detach().cpu()
                a = a.view(*s_shape[:-3])
                next_s, r, done, info_dict = self.env.step(a.numpy())
                next_s = torch.from_numpy(next_s)
                r = torch.from_numpy(r)
                done = torch.from_numpy(done)
                self.replay_buffer.append_samples_batch(
                    s[still_playing_mask].reshape(-1, *s.shape[-3:]).clone(),
                    a[still_playing_mask].reshape(-1).clone(),
                    r[still_playing_mask].reshape(-1).clone(),
                    done[still_playing_mask].reshape(-1).clone(),
                    next_s[still_playing_mask].reshape(-1, *next_s.shape[-3:]).clone()
                )
                if epsilon_expanded is not None:
                    for eps_idx, eps in enumerate(epsilon.view(-1)):
                        self.summary_writer.add_scalar(f'Info/epsilon_{eps_idx}',
                                                       eps,
                                                       self.train_step_counter)
                self.summary_writer.add_scalar(f'Info/replay_buffer__top',
                                               self.replay_buffer._top,
                                               self.train_step_counter)
                s = next_s
                if info_dict['episodes_finished_last_turn'].any():
                    for env_idx, did_episode_finish in enumerate(info_dict['episodes_finished_last_turn']):
                        if did_episode_finish:
                            finished_episode_infos.append(copy.deepcopy(
                                info_dict['episodes_finished_last_turn_info'][env_idx]
                            ))
                self.summary_writer.add_scalar('Time/exploration_step_time_ms',
                                               (time.time() - start_time) * 1000,
                                               self.train_step_counter)
                self.train_step_counter += 1

                if step % train_frequency == 0:
                    self.q_model.train()
                    for _ in range(n_train_batches):
                        self.train_on_batch(batch_size, gamma)
                    self.q_model.eval()

            self.log_epoch(finished_episode_infos)
            self.epoch_counter += 1
            if self.epoch_counter % self.checkpoint_freq == 0:
                self.checkpoint()
            self.summary_writer.add_scalar('Time/epoch_time_minutes',
                                           (time.time() - epoch_start_time) / 60.,
                                           self.epoch_counter - 1)
            print()
        self.save(self.exp_folder, finished=True)

    def train_on_batch(self, batch_size, gamma):
        start_time = time.time()
        s_batch, a_batch, r_batch, d_batch, next_s_batch = self.replay_buffer.get_samples_batch(batch_size)
        s_batch = s_batch.to(device=self.device)
        a_batch = a_batch.to(device=self.device)
        r_batch = r_batch.to(device=self.device)
        d_batch = d_batch.to(device=self.device)
        next_s_batch = next_s_batch.to(device=self.device)
        losses_dict = self.q_model.train_on_batch(s_batch, a_batch, r_batch, d_batch, next_s_batch, gamma)
        self.log_batch(losses_dict)
        self.summary_writer.add_scalar('Time/batch_time_ms',
                                       (time.time() - start_time) * 1000,
                                       self.batch_counter)
        self.batch_counter += 1

    def log_batch(self, losses_dict: Dict):
        for loss_name, loss_val in losses_dict.items():
            self.summary_writer.add_scalar(f'Batch/{loss_name}',
                                           loss_val.detach().cpu().numpy().item(),
                                           self.batch_counter)

    def log_epoch(self, finished_episode_infos: Sequence[Dict]):
        all_n_steps = np.array([fei['n_steps'] for fei in finished_episode_infos])
        all_goose_death_times = np.array([fei['goose_death_times'] for fei in finished_episode_infos]).ravel()
        all_goose_lengths = np.array([fei['goose_lengths'] for fei in finished_episode_infos]).ravel()
        _all_goose_rankings = np.array([fei['goose_rankings'] for fei in finished_episode_infos]).ravel()
        all_winning_goose_lengths = all_goose_lengths[_all_goose_rankings == self.env.n_players]

        self.summary_writer.add_scalar(f'Epoch/mean_n_steps',
                                       all_n_steps.mean(),
                                       self.epoch_counter)
        self.summary_writer.add_scalar('Epoch/mean_goose_death_times',
                                       all_goose_death_times.mean(),
                                       self.epoch_counter)
        self.summary_writer.add_scalar('Epoch/mean_goose_lengths',
                                       all_goose_lengths.mean(),
                                       self.epoch_counter)
        self.summary_writer.add_scalar('Epoch/mean_winning_goose_lengths',
                                       all_winning_goose_lengths.mean(),
                                       self.epoch_counter)

        self.summary_writer.add_histogram(f'Epoch/n_steps',
                                          all_n_steps,
                                          self.epoch_counter)
        self.summary_writer.add_histogram('Epoch/goose_death_times',
                                          all_goose_death_times,
                                          self.epoch_counter)
        self.summary_writer.add_histogram('Epoch/goose_lengths',
                                          all_goose_lengths,
                                          self.epoch_counter)
        self.summary_writer.add_histogram('Epoch/winning_goose_lengths',
                                          all_winning_goose_lengths,
                                          self.epoch_counter)

    def checkpoint(self):
        checkpoint_start_time = time.time()
        for name, param in self.q_model.named_parameters():
            if param.requires_grad:
                self.summary_writer.add_histogram(
                    f'Parameters/{name}',
                    param.detach().cpu().clone().numpy(),
                    int(self.epoch_counter / self.checkpoint_freq)
                )
        checkpoint_dir = self.exp_folder / f'{self.epoch_counter:04}'
        checkpoint_dir.mkdir()
        self.run_validation()
        self.render_n_games(checkpoint_dir)
        self.save(checkpoint_dir)
        self.summary_writer.add_scalar('Time/checkpoint_time_s',
                                       (time.time() - checkpoint_start_time),
                                       int(self.epoch_counter / self.checkpoint_freq))

    def run_validation(self):
        return None
        # TODO: Need to reimplement
        self.model.eval()
        if len(self.validation_kwargs_dicts) > 0:
            print(f'Validating model performance in {len(self.validation_kwargs_dicts)} environments')
            episode_reward_sums = []
            final_info_dicts = []
            for i in tqdm.trange(len(self.validation_kwargs_dicts)):
                if 'opponent' in self.validation_kwargs_dicts[i].keys():
                    self.validation_kwargs_dicts[i]['opponent'].soft_reset()
                # Lazily construct validation envs to conserve GPU memory
                val_env = ve.KaggleMABEnvTorchVectorized(**self.validation_kwargs_dicts[i])
                s, r, done, info_dict = val_env.soft_reset()
                episode_reward_sums.append(r)
                while not done:
                    start_time = time.time()
                    if self.deterministic_validation_policy:
                        a = self.model.choose_best_action(s.to(device=self.device).unsqueeze(0))
                    else:
                        a = self.model.sample_action(s.to(device=self.device).unsqueeze(0))
                    next_s, r, done, info_dict = val_env.step(a.squeeze(0))
                    s = next_s
                    episode_reward_sums[-1] += r
                    self.summary_writer.add_scalar(f'Time/val_env{i}_step_time_ms',
                                                   (time.time() - start_time) * 1000,
                                                   self.validation_step_counters[i])
                    self.validation_step_counters[i] += 1
                episode_reward_sums[-1] = episode_reward_sums[-1].mean(dim=-1).cpu().clone() / val_env.r_norm
                final_info_dicts.append(info_dict)
            self.log_validation_episodes(
                episode_reward_sums,
                final_info_dicts
            )
        self.validation_counter += 1

    def log_validation_episodes(self, episode_reward_sums, final_info_dicts):
        return None
        # TODO: Need to reimplement
        assert len(episode_reward_sums) == len(final_info_dicts)
        n_val_envs = len(episode_reward_sums)
        for i, ers, fid in zip(range(n_val_envs), episode_reward_sums, final_info_dicts):
            opponent = self.validation_kwargs_dicts[i].get('opponent')
            if opponent is not None:
                opponent = opponent.name
            env_name = opponent

            self.summary_writer.add_histogram(
                f'Validation/{env_name}_game_results',
                ers.numpy(),
                self.validation_counter
            )
            self.summary_writer.add_histogram(
                f'Validation/{env_name}_hero_pull_rewards',
                fid['player_rewards_sums'].sum(dim=-1)[:, 0].cpu().numpy(),
                self.validation_counter
            )
            self.summary_writer.add_histogram(
                f'Validation/{env_name}_villain_pull_rewards',
                fid['player_rewards_sums'].sum(dim=-1)[:, 1].cpu().numpy(),
                self.validation_counter
            )
            self.summary_writer.add_scalar(
                f'Validation/{env_name}_win_percent',
                # Rank values are in the range [-1, 1], so they need to be scaled to 0-1 to be represented as a percent
                (ers.mean().numpy().item() + 1) / 2. * 100.,
                self.validation_counter
            )
            self.summary_writer.add_scalar(
                f'Validation/{env_name}_mean_hero_pull_rewards',
                fid['player_rewards_sums'].sum(dim=-1)[:, 0].mean().cpu().numpy().item(),
                self.validation_counter
            )
            self.summary_writer.add_scalar(
                f'Validation/{env_name}_mean_villain_pull_rewards',
                fid['player_rewards_sums'].sum(dim=-1)[:, 1].mean().cpu().numpy().item(),
                self.validation_counter
            )

    def render_n_games(self, checkpoint_dir: Path):
        if self.checkpoint_render_n_games > 0:
            save_dir = checkpoint_dir / 'replays'
            save_dir.mkdir()
            rendering_env = ge.VectorizedEnv(self.env.obs_type, self.env.reward_type, self.env.action_masking,
                                             n_envs=min(self.env.n_envs, self.checkpoint_render_n_games),
                                             silent_reset=False)
            n_envs_rendered = 0
            s, _, _, info_dict = rendering_env.hard_reset()
            s = torch.from_numpy(s)
            while n_envs_rendered < self.checkpoint_render_n_games:
                s_shape = s.shape
                if self.use_action_masking:
                    available_actions_mask = torch.from_numpy(
                        info_dict['available_actions_mask']
                    ).to(device=self.device).view(-1, 4)
                else:
                    available_actions_mask = None
                a = self.q_model.choose_best_action(
                    s.to(device=self.device).view(-1, *s_shape[-3:]),
                    available_actions_mask=available_actions_mask
                ).detach().cpu()
                # a = self.q_model.choose_best_action(s.to(device=self.device).view(-1, *s_shape[-3:])).detach().cpu()
                a = a.view(*s_shape[:-3])
                next_s, _, _, info_dict = rendering_env.step(a.numpy())
                next_s = torch.from_numpy(next_s)

                s = next_s
                if rendering_env.episodes_done.any():
                    for env_idx, did_episode_finish in enumerate(rendering_env.episodes_done):
                        if did_episode_finish:
                            rendered_html = rendering_env.wrapped_envs[env_idx].render(mode='html')
                            ep_length = rendering_env.wrapped_envs[env_idx].steps[-1][0]['observation']['step']
                            with open(save_dir / f'replay_{n_envs_rendered}_{ep_length}_steps.html', 'w') as f:
                                f.write(rendered_html)
                            n_envs_rendered += 1
                            if n_envs_rendered >= self.checkpoint_render_n_games:
                                break
                    s, _, _, info_dict = rendering_env.soft_reset()
                    s = torch.from_numpy(s)

    def save(self, save_dir: Path, finished: bool = False):
        if finished:
            save_dir = save_dir / f'final_{self.epoch_counter}'
            save_dir.mkdir()
        # Save model params
        self.q_model.cpu()
        state_dict_bytes = pickle.dumps(self.q_model.state_dict())
        serialized_string = base64.b64encode(state_dict_bytes)
        with open(save_dir / 'cp.txt', 'w') as f:
            f.write(str(serialized_string))
        q_state_dict_bytes = pickle.dumps(self.q_model.q_1.state_dict())
        q_serialized_string = base64.b64encode(q_state_dict_bytes)
        with open(save_dir / 'Q_1_cp.txt', 'w') as f:
            f.write(str(q_serialized_string))
        self.q_model.to(device=self.device)
