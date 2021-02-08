import base64
import copy
import numpy as np
from pathlib import Path
import pickle
import shutil
import time
import torch
from torch import distributions
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import tqdm
from typing import *

# Custom imports
from . import goose_env as ge


class AWAC:
    def __init__(
            self,
            model: torch.nn.Module,
            optimizer: torch.optim,
            env,
            replay_buffer,
            validation_kwargs_dicts: Sequence[Dict] = (),
            deterministic_validation_policy: bool = True,
            device: torch.device = torch.device('cuda'),
            exp_folder: Path = Path('runs/awac/TEMP'),
            q_val_clamp: Tuple[Union[float, None], Union[float, None]] = (None, None),
            clip_grads: float = 10.,
            checkpoint_freq: int = 10,
            checkpoint_render_n_games: int = 10,
    ):
        self.model = model
        self.model.train()
        self.optimizer = optimizer
        self.replay_buffer = replay_buffer
        self.validation_kwargs_dicts = validation_kwargs_dicts
        self.deterministic_validation_policy = deterministic_validation_policy
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

        self.q_val_clamp = q_val_clamp
        if None not in self.q_val_clamp:
            assert self.q_val_clamp[0] < self.q_val_clamp[1]
        if self.clip_grads is not None:
            if self.clip_grads <= 0:
                raise ValueError(f'Should not clip gradients to <= 0, was {self.clip_grads} - '
                                 'pass None to clip_grads to not clip gradients')
            for p in self.model.parameters():
                if p.requires_grad:
                    p.register_hook(lambda grad: torch.clamp(grad, -self.clip_grads, self.clip_grads))

        dummy_input, *_ = self.env.hard_reset()
        dummy_input = torch.from_numpy(dummy_input).to(device=self.device)[0, 0].unsqueeze(0)
        self.summary_writer.add_graph(self.model, dummy_input)

    def train(self, batch_size, n_pretrain_batches, n_epochs, n_steps_per_epoch,
              n_train_batches_per_epoch=None, gamma=0.99, lagrange_multiplier=1.):
        if n_train_batches_per_epoch is None:
            n_train_batches_per_epoch = n_steps_per_epoch

        self.model.train()
        print(f'Pre-training on {n_pretrain_batches} batches')
        for batch in tqdm.trange(n_pretrain_batches):
            self.train_on_batch(batch_size, gamma, lagrange_multiplier)

        self.run_validation()
        print(f'\nRunning main training loop with {n_epochs} epochs')
        for epoch in range(n_epochs):
            print(f'Epoch #{epoch}:')
            print(f'Sampling {n_steps_per_epoch} time-steps from the environment')
            epoch_start_time = time.time()
            finished_episode_infos = []
            self.model.eval()
            s, r, done, info_dict = self.env.hard_reset()
            s = torch.from_numpy(s)
            done = torch.from_numpy(done)
            for step in tqdm.trange(n_steps_per_epoch):
                start_time = time.time()
                still_playing_mask = torch.logical_not(done)
                s_shape = s.shape
                a = self.model.sample_action(s.to(device=self.device).view(-1, *s_shape[-3:])).detach().cpu()
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
                self.summary_writer.add_scalar(f'Info/replay_buffer_size',
                                               len(self.replay_buffer),
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

            self.model.train()
            print(f'Training on {n_train_batches_per_epoch} batches from the replay buffer')
            for batch in tqdm.trange(n_train_batches_per_epoch):
                self.train_on_batch(batch_size, gamma, lagrange_multiplier)

            self.log_epoch(finished_episode_infos)
            self.epoch_counter += 1
            if self.epoch_counter % self.checkpoint_freq == 0:
                self.checkpoint()
            self.summary_writer.add_scalar('Time/epoch_time_minutes',
                                           (time.time() - epoch_start_time) / 60.,
                                           self.epoch_counter - 1)
            print()
        self.save(self.exp_folder, finished=True)

    def train_on_batch(self, batch_size, gamma, lagrange_multiplier):
        start_time = time.time()
        s_batch, a_batch, r_batch, d_batch, next_s_batch = self.replay_buffer.get_samples_batch(batch_size)
        s_batch = s_batch.to(device=self.device)
        a_batch = a_batch.to(device=self.device)
        r_batch = r_batch.to(device=self.device)
        d_batch = d_batch.to(device=self.device)
        next_s_batch = next_s_batch.to(device=self.device)
        logits, values = self.model(s_batch.to(device=self.device))
        _, v_next_s = self.model(next_s_batch.to(device=self.device))
        # Reward of 0 for terminal states
        v_t = (r_batch + gamma * v_next_s * (1. - d_batch)).detach()
        if self.q_val_clamp[0] is not None or self.q_val_clamp[1] is not None:
            v_t = torch.clamp(v_t, min=self.q_val_clamp[0], max=self.q_val_clamp[1])
        # Huber loss for critic
        critic_loss = F.smooth_l1_loss(values, v_t, reduction='none').view(-1)

        td = v_t - values
        log_probs = distributions.Categorical(F.softmax(logits, dim=-1)).log_prob(a_batch.view(-1))
        weights = torch.exp(td / lagrange_multiplier).view(-1)
        # weights = F.softmax(td.view(-1) / lagrange_multiplier, dim=-1)
        actor_loss = -(log_probs * weights.detach())

        total_loss = (critic_loss + actor_loss).mean()
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        self.log_batch(actor_loss.mean(), critic_loss.mean(), total_loss)
        self.summary_writer.add_scalar('Time/batch_time_ms',
                                       (time.time() - start_time) * 1000,
                                       self.batch_counter)
        self.batch_counter += 1

    def log_batch(self, actor_loss, critic_loss, total_loss):
        self.summary_writer.add_scalar('Batch/actor_loss', actor_loss.detach().cpu().numpy().item(),
                                       self.batch_counter)
        self.summary_writer.add_scalar('Batch/critic_loss', critic_loss.detach().cpu().numpy().item(),
                                       self.batch_counter)
        self.summary_writer.add_scalar('Batch/total_loss', total_loss.detach().cpu().numpy().item(),
                                       self.batch_counter)

    def log_epoch(self, finished_episode_infos: Sequence[Dict]):
        all_n_steps = np.array([fei['n_steps'] for fei in finished_episode_infos])
        all_goose_death_times = np.array([fei['goose_death_times'] for fei in finished_episode_infos]).ravel()
        all_goose_lengths = np.array([fei['goose_lengths'] for fei in finished_episode_infos]).ravel()
        self.summary_writer.add_scalar(f'Epoch/mean_n_steps',
                                       all_n_steps.mean(),
                                       self.epoch_counter)
        self.summary_writer.add_scalar('Epoch/mean_goose_death_times',
                                       all_goose_death_times.mean(),
                                       self.epoch_counter)
        self.summary_writer.add_scalar('Epoch/mean_goose_lengths',
                                       all_goose_lengths.mean(),
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
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.summary_writer.add_histogram(
                    f'params/{name}',
                    param.detach().cpu().clone().numpy(),
                    self.epoch_counter
                )

    def checkpoint(self):
        checkpoint_dir = self.exp_folder / f'{self.epoch_counter:04}'
        checkpoint_dir.mkdir()
        self.run_validation()
        self.render_n_games(checkpoint_dir)
        self.save(checkpoint_dir)

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
            rendering_env = ge.GooseEnvVectorized(self.env.obs_type, self.env.reward_type,
                                                  n_envs=min(self.env.n_envs, self.checkpoint_render_n_games),
                                                  silent_reset=False)
            n_envs_rendered = 0
            s, *_ = rendering_env.hard_reset()
            s = torch.from_numpy(s)
            while n_envs_rendered < self.checkpoint_render_n_games:
                s_shape = s.shape
                a = self.model.sample_action(s.to(device=self.device).view(-1, *s_shape[-3:])).detach().cpu()
                a = a.view(*s_shape[:-3])
                next_s, *_ = rendering_env.step(a.numpy())
                next_s = torch.from_numpy(next_s)

                s = next_s
                if rendering_env.episodes_done.any():
                    for env_idx, did_episode_finish in enumerate(rendering_env.episodes_done):
                        if did_episode_finish:
                            n_envs_rendered += 1
                            rendered_html = rendering_env.wrapped_envs[env_idx].render(mode='html')
                            with open(save_dir / f'replay_{n_envs_rendered}.html', 'w') as f:
                                f.write(rendered_html)
                            if n_envs_rendered >= self.checkpoint_render_n_games:
                                break
                    s, *_ = rendering_env.soft_reset()
                    s = torch.from_numpy(s)

    def save(self, save_dir: Path, finished: bool = False):
        if finished:
            save_dir = save_dir / f'final_{self.epoch_counter}'
            save_dir.mkdir()
        # Save model params
        self.model.cpu()
        state_dict_bytes = pickle.dumps(self.model.state_dict())
        serialized_string = base64.b64encode(state_dict_bytes)
        with open(save_dir / 'cp.txt', 'w') as f:
            f.write(str(serialized_string))
        self.model.to(device=self.device)


class BasicReplayBuffer:
    def __init__(self, s_shape, max_len=1e6, starting_s_a_r_d_s=None):
        self.max_len = int(max_len)
        self._s_buffer = torch.zeros(self.max_len, *s_shape)
        self._a_buffer = torch.zeros(self.max_len)
        self._r_buffer = torch.zeros(self.max_len)
        self._d_buffer = torch.zeros(self.max_len)
        self._next_s_buffer = torch.zeros(self.max_len, *s_shape)
        self.current_size = 0
        self._top = 0
        if starting_s_a_r_d_s is not None:
            self.append_samples_batch(*starting_s_a_r_d_s)
            # Randomly shuffle initial experiences
            shuffled_idxs = np.arange(self.current_size)
            np.random.shuffle(shuffled_idxs)
            shuffled_idxs = np.append(shuffled_idxs, np.arange(self.current_size, self.max_len))
            self._s_buffer = self._s_buffer[torch.from_numpy(shuffled_idxs)]
            self._a_buffer = self._a_buffer[torch.from_numpy(shuffled_idxs)]
            self._r_buffer = self._r_buffer[torch.from_numpy(shuffled_idxs)]
            self._d_buffer = self._d_buffer[torch.from_numpy(shuffled_idxs)]
            self._next_s_buffer = self._next_s_buffer[torch.from_numpy(shuffled_idxs)]

    def get_samples_batch(self, sample_size):
        # Sampling with replacement
        idxs = torch.randint(self.current_size, size=(sample_size,))
        # Sampling without replacement is possible, but quite a bit slower:
        # idxs = np.random.choice(self.current_size, size=sample_size, replace=(self.current_size < sample_size))
        return (self._s_buffer[idxs],
                self._a_buffer[idxs],
                self._r_buffer[idxs],
                self._d_buffer[idxs],
                self._next_s_buffer[idxs])

    def append_samples_batch(self, s_batch, a_batch, r_batch, d_batch, next_s_batch):
        batch_len = s_batch.shape[0]
        assert a_batch.shape[0] == batch_len
        assert r_batch.shape[0] == batch_len
        assert d_batch.shape[0] == batch_len
        assert next_s_batch.shape[0] == batch_len
        new_len = self._top + batch_len
        if new_len <= self.max_len:
            self._s_buffer[self._top:new_len] = s_batch
            self._a_buffer[self._top:new_len] = a_batch
            self._r_buffer[self._top:new_len] = r_batch
            self._d_buffer[self._top:new_len] = d_batch
            self._next_s_buffer[self._top:new_len] = next_s_batch
            self._top = new_len % self.max_len
            self.current_size = max(new_len, self.current_size)
        else:
            leftover_batch = new_len % self.max_len
            s_batch_split = s_batch.split((batch_len - leftover_batch, leftover_batch))
            a_batch_split = a_batch.split((batch_len - leftover_batch, leftover_batch))
            r_batch_split = r_batch.split((batch_len - leftover_batch, leftover_batch))
            d_batch_split = d_batch.split((batch_len - leftover_batch, leftover_batch))
            next_s_batch_split = next_s_batch.split((batch_len - leftover_batch, leftover_batch))
            self.append_samples_batch(s_batch_split[0],
                                      a_batch_split[0],
                                      r_batch_split[0],
                                      d_batch_split[0],
                                      next_s_batch_split[0])
            self.append_samples_batch(s_batch_split[1],
                                      a_batch_split[1],
                                      r_batch_split[1],
                                      d_batch_split[1],
                                      next_s_batch_split[1])

    def __len__(self):
        return self.current_size
