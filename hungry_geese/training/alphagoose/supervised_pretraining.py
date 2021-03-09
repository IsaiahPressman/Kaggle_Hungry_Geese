import base64
from kaggle_environments import make
import numpy as np
from pathlib import Path
import pickle
import shutil
import time
import torch
import torch.nn.functional as F
from torch.cuda import amp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.jit import TracerWarning
from tqdm import tqdm
from typing import *
import warnings

from ...models import FullConvActorCriticNetwork
from ...env import goose_env as ge


class SupervisedPretraining:
    def __init__(
            self,
            model: FullConvActorCriticNetwork,
            optimizer: torch.optim,
            lr_scheduler: torch.optim.lr_scheduler,
            train_dataloader: DataLoader,
            test_dataloader: DataLoader,
            policy_weight: float = 1.,
            value_weight: float = 1.,
            entropy_weight: float = 0.05,
            device: torch.device = torch.device('cuda'),
            use_mixed_precision: bool = True,
            grad_scaler: amp.grad_scaler = amp.GradScaler(),
            exp_folder: Path = Path('runs/supervised_pretraining/TEMP'),
            checkpoint_freq: int = 10,
            checkpoint_render_n_games: int = 10
    ):
        self.model = model
        self.model.train()
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.policy_weight = policy_weight
        assert self.policy_weight >= 0.
        self.value_weight = value_weight
        assert self.value_weight >= 0.
        self.entropy_weight = entropy_weight
        assert self.entropy_weight >= 0.
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.device = device
        self.use_mixed_precision = use_mixed_precision
        self.grad_scaler = grad_scaler
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

        dummy_state = torch.zeros(self.train_dataloader.dataset.obs_type.get_obs_spec()[1:])
        dummy_head_loc = torch.arange(4).to(dtype=torch.int64)
        still_alive = torch.ones(4).to(dtype=torch.bool)
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
                'entropy_loss': [],
                'combined_loss': []
            }
            for train_tuple in tqdm(self.train_dataloader, desc=f'Epoch #{epoch} train'):
                self.optimizer.zero_grad()
                policy_loss, value_loss, entropy_loss, combined_loss = self.compute_losses(
                    *[t.to(device=self.device, non_blocking=True) for t in train_tuple],
                    reduction='mean'
                )
                if self.use_mixed_precision:
                    self.grad_scaler.scale(combined_loss).backward()
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
                else:
                    combined_loss.backward()
                    self.optimizer.step()
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=UserWarning)
                    self.lr_scheduler.step()

                train_metrics['policy_loss'].append(policy_loss.detach().cpu().item())
                train_metrics['value_loss'].append(value_loss.detach().cpu().item())
                train_metrics['entropy_loss'].append(entropy_loss.detach().cpu().item())
                train_metrics['combined_loss'].append(combined_loss.detach().cpu().item())
            self.log_train(train_metrics)

            self.model.eval()
            test_metrics = {
                'policy_loss': 0.,
                'value_loss': 0.,
                'entropy_loss': 0.,
                'combined_loss': 0.,
                'policy_accuracy': 0.
            }
            n_test_samples = 0.
            with torch.no_grad():
                for test_tuple in tqdm(self.test_dataloader, desc=f'Epoch #{epoch} test'):
                    test_tuple = [t.to(device=self.device, non_blocking=True) for t in test_tuple]
                    state, action, result, head_locs, still_alive = test_tuple
                    policy_loss, value_loss, entropy_loss, combined_loss, preds = self.compute_losses(
                        *test_tuple,
                        reduction='sum',
                        get_preds=True
                    )
                    action_masked = action.view(-1)[still_alive.view(-1)]

                    test_metrics['policy_loss'] += policy_loss.detach().cpu().item()
                    test_metrics['value_loss'] += value_loss.detach().cpu().item()
                    test_metrics['entropy_loss'] += entropy_loss.detach().cpu().item()
                    test_metrics['combined_loss'] += combined_loss.detach().cpu().item()
                    test_metrics['policy_accuracy'] += preds.eq(action_masked).sum().cpu().item()
                    n_test_samples += still_alive.sum().cpu().item()

                for key, metric in test_metrics.items():
                    test_metrics[key] = metric / n_test_samples
            self.log_test(test_metrics)
            if self.epoch_counter % self.checkpoint_freq == 0 and self.epoch_counter > 0:
                self.checkpoint()
            epoch_time = time.time() - epoch_start_time
            self.summary_writer.add_scalar('Time/epoch_time_minutes',
                                           epoch_time / 60.,
                                           self.epoch_counter)
            self.summary_writer.add_scalar(
                'Time/batch_time_ms',
                1000. * epoch_time / (len(self.train_dataloader) + len(self.test_dataloader)),
                self.epoch_counter
            )
            self.summary_writer.add_scalar(
                'Time/sample_time_ms',
                1000. * epoch_time / (len(self.train_dataloader.dataset) + len(self.test_dataloader.dataset)),
                self.epoch_counter
            )
            self.epoch_counter += 1

    def compute_losses(
            self,
            state: torch.Tensor,
            action: torch.Tensor,
            result: torch.Tensor,
            head_locs: torch.Tensor,
            still_alive: torch.Tensor,
            reduction: str = 'mean',
            get_preds: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        with amp.autocast(enabled=self.use_mixed_precision):
            logits, value = self.model(state, head_locs, still_alive)

            logits_masked = logits.view(-1, 4)[still_alive.view(-1, 1).expand(-1, 4)].view(-1, 4)
            action_masked = action.view(-1)[still_alive.view(-1)]
            policy_loss = F.cross_entropy(logits_masked, action_masked, reduction=reduction)

            value_masked = value.view(-1)[still_alive.view(-1)]
            result_masked = result.view(-1)[still_alive.view(-1)]
            value_loss = F.mse_loss(value_masked, result_masked, reduction=reduction)

            probs_masked = F.softmax(logits_masked, dim=-1)
            entropy_loss = torch.sum(probs_masked * torch.log(probs_masked), dim=-1)
            if reduction == 'none':
                pass
            elif reduction == 'mean':
                entropy_loss = entropy_loss.mean()
            elif reduction == 'sum':
                entropy_loss = entropy_loss.sum()
            else:
                raise ValueError(f'Unrecognized reduction: {reduction}')

            combined_loss = (self.policy_weight * policy_loss +
                             self.value_weight * value_loss +
                             self.entropy_weight * entropy_loss)

        if get_preds:
            return policy_loss, value_loss, entropy_loss, combined_loss, logits_masked.argmax(dim=-1)
        else:
            return policy_loss, value_loss, entropy_loss, combined_loss

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
