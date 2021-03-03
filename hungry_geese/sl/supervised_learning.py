import base64
from kaggle_environments import make
import numpy as np
from pathlib import Path
import pickle
import shutil
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.jit import TracerWarning
from tqdm import tqdm
from typing import *
import warnings

from ..models import FullConvActorCriticNetwork
from ..env import goose_env as ge


class SupervisedLearning:
    def __init__(
            self,
            model: FullConvActorCriticNetwork,
            optimizer: torch.optim,
            lr_scheduler: torch.optim.lr_scheduler,
            train_dataloader: DataLoader,
            test_dataloader: DataLoader,
            device: torch.device = torch.device('cuda'),
            exp_folder: Path = Path('runs/supervised_learning/TEMP'),
            checkpoint_freq: int = 10,
            checkpoint_render_n_games: int = 10
    ):
        self.model = model
        self.model.train()
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
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
        self.checkpoint_freq = checkpoint_freq
        self.checkpoint_render_n_games = checkpoint_render_n_games
        if self.checkpoint_render_n_games > 0:
            self.rendering_env_kwargs = dict(
                obs_type=self.train_dataloader.dataset.obs_type,
                reward_type=ge.RewardType.RANK_ON_DEATH,
                action_masking=ge.ActionMasking.LETHAL,
                n_envs=max(self.checkpoint_render_n_games, 20),
                silent_reset=False,
                make_fn=make
            )
        else:
            self.rendering_env_kwargs = None

        self.epoch_counter = 0
        self.summary_writer = SummaryWriter(str(self.exp_folder))

        dummy_state, _, _, dummy_head_loc, still_alive = self.train_dataloader.dataset[0]
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=TracerWarning)
            self.summary_writer.add_graph(
                self.model,
                [dummy_state.unsqueeze(0).to(device=self.device),
                 dummy_head_loc.unsqueeze(0).to(device=self.device),
                 still_alive.unsqueeze(0).to(device=self.device)]
            )

    def train(self, n_epochs: int) -> NoReturn:
        for epoch in range(n_epochs):
            epoch_start_time = time.time()
            self.model.train()
            train_metrics = {
                'policy_loss': [],
                'value_loss': [],
                'combined_loss': []
            }
            for state, action, result, head_locs, still_alive in tqdm(self.train_dataloader,
                                                                      desc=f'Epoch #{epoch} train'):
                state = state.to(device=self.device)
                action = action.to(device=self.device)
                result = result.to(device=self.device)
                head_locs = head_locs.to(device=self.device)
                still_alive = still_alive.to(device=self.device)

                self.optimizer.zero_grad()
                logits, value = self.model(state, head_locs, still_alive)

                logits = logits.view(-1, 4)
                action = action.view(-1)
                policy_loss = F.cross_entropy(logits, action, ignore_index=-1)

                value_masked = value.view(-1)[still_alive.view(-1)]
                result_masked = result.view(-1)[still_alive.view(-1)]
                value_loss = F.mse_loss(value_masked, result_masked)

                combined_loss = policy_loss + value_loss
                combined_loss.backward()
                self.optimizer.step()

                train_metrics['policy_loss'].append(policy_loss.detach().cpu().item())
                train_metrics['value_loss'].append(value_loss.detach().cpu().item())
                train_metrics['combined_loss'].append(combined_loss.detach().cpu().item())
            self.log_train(train_metrics)

            self.model.eval()
            test_metrics = {
                'policy_loss': 0.,
                'value_loss': 0.,
                'combined_loss': 0.,
                'policy_accuracy': 0.
            }
            n_test_samples = 0.
            with torch.no_grad():
                for state, action, result, head_locs, still_alive in tqdm(self.test_dataloader,
                                                                          desc=f'Epoch #{epoch} test'):
                    state = state.to(device=self.device)
                    action = action.to(device=self.device)
                    result = result.to(device=self.device)
                    head_locs = head_locs.to(device=self.device)
                    still_alive = still_alive.to(device=self.device)

                    logits, value = self.model(state, head_locs, still_alive)

                    logits_masked = logits.view(-1, 4)[still_alive.view(-1, 1).expand(-1, 4)].view(-1, 4)
                    action_masked = action.view(-1)[still_alive.view(-1)]
                    policy_loss = F.cross_entropy(logits_masked, action_masked, reduction='sum')

                    value_masked = value.view(-1)[still_alive.view(-1)]
                    result_masked = result.view(-1)[still_alive.view(-1)]
                    value_loss = F.mse_loss(value_masked, result_masked, reduction='sum')

                    combined_loss = policy_loss + value_loss

                    preds = logits_masked.argmax(dim=-1)

                    test_metrics['policy_loss'] += policy_loss.detach().cpu().item()
                    test_metrics['value_loss'] += value_loss.detach().cpu().item()
                    test_metrics['combined_loss'] += combined_loss.detach().cpu().item()
                    test_metrics['policy_accuracy'] += preds.eq(action_masked).sum().cpu().item()
                    n_test_samples += still_alive.sum().cpu().item()

                for key, metric in test_metrics.items():
                    test_metrics[key] = metric / n_test_samples
            self.log_test(test_metrics)
            if self.epoch_counter % self.checkpoint_freq == 0 and self.epoch_counter > 0:
                self.checkpoint()
            self.lr_scheduler.step()
            self.summary_writer.add_scalar('Time/epoch_time_minutes',
                                           (time.time() - epoch_start_time) / 60.,
                                           self.epoch_counter)
            self.epoch_counter += 1

    def log_train(self, train_metrics: Dict[str, List]) -> NoReturn:
        n_batches = len(self.train_dataloader)
        for metric_name, metric in train_metrics.items():
            self.summary_writer.add_scalar(f'Epoch/train_{metric_name}',
                                           np.mean(metric),
                                           self.epoch_counter)
            for i, m in enumerate(metric):
                self.summary_writer.add_scalar(f'Batch/train_{metric_name}',
                                               m,
                                               (self.epoch_counter + 1) * n_batches - (n_batches - i))
        last_lr = self.lr_scheduler.get_last_lr()
        assert len(last_lr) == 1, 'Logging per-parameter LR still needs support'
        self.summary_writer.add_scalar(f'Epoch/learning_rate',
                                       last_lr[0],
                                       self.epoch_counter)

    def log_test(self, test_metrics: Dict[str, float]) -> NoReturn:
        for metric_name, metric in test_metrics.items():
            self.summary_writer.add_scalar(f'Epoch/test_{metric_name}',
                                           metric,
                                           self.epoch_counter)

    def checkpoint(self):
        checkpoint_start_time = time.time()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.summary_writer.add_histogram(
                    f'Parameters/{name}',
                    param.detach().cpu().clone().numpy(),
                    self.epoch_counter // self.checkpoint_freq
                )
        checkpoint_dir = self.exp_folder / f'{self.epoch_counter:04}'
        checkpoint_dir.mkdir()
        self.render_n_games(checkpoint_dir)
        self.save(checkpoint_dir)
        self.summary_writer.add_scalar('Time/checkpoint_time_s',
                                       (time.time() - checkpoint_start_time),
                                       int(self.epoch_counter / self.checkpoint_freq))

    def render_n_games(self, checkpoint_dir: Path):
        if self.checkpoint_render_n_games > 0:
            save_dir = checkpoint_dir / 'replays'
            save_dir.mkdir()
            rendering_env = ge.VectorizedEnv(**self.rendering_env_kwargs)
            n_envs_rendered = 0
            s, _, agent_dones, info_dict = rendering_env.hard_reset()
            s = torch.from_numpy(s)
            while n_envs_rendered < self.checkpoint_render_n_games:
                s_shape = s.shape
                head_locs = torch.from_numpy(
                    info_dict['head_locs']
                ).to(device=self.device).view(-1, 4)
                still_alive = torch.from_numpy(
                    ~agent_dones
                ).to(device=self.device).view(-1, 4)
                available_actions_mask = torch.from_numpy(
                    info_dict['available_actions_mask']
                ).to(device=self.device).view(-1, rendering_env.n_players, 4)
                a = self.model.choose_best_action(
                    s.to(device=self.device).view(-1, *s_shape[-3:]),
                    head_locs=head_locs,
                    still_alive=still_alive,
                    available_actions_mask=available_actions_mask,
                ).detach().cpu()
                a = a.view(-1, rendering_env.n_players)
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
        self.model.cpu()
        state_dict_bytes = pickle.dumps(self.model.state_dict())
        serialized_string = base64.b64encode(state_dict_bytes)
        with open(save_dir / 'cp.txt', 'w') as f:
            f.write(str(serialized_string))
        self.model.to(device=self.device)
