import base64
from kaggle_environments import make
from pathlib import Path
import pickle
import shutil
import time
import torch
import torch.nn.functional as F
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter
from torch.jit import TracerWarning
import tqdm
from typing import *
import warnings

# Custom imports
from ..env import goose_env as ge
from ..env.torch_env import TorchEnv
from ..nns.models import FullConvActorCriticNetwork


class A2C:
    def __init__(
            self,
            model: FullConvActorCriticNetwork,
            optimizer: torch.optim,
            lr_scheduler: torch.optim.lr_scheduler,
            env: TorchEnv,
            policy_weight: float = 1.,
            value_weight: float = 1.,
            entropy_weight: float = 0.1,
            use_action_masking: bool = True,
            use_mixed_precision: bool = True,
            grad_scaler: amp.grad_scaler = amp.GradScaler(),
            clip_grads: Optional[float] = 10.,
            exp_folder: Path = Path('runs/a2c/TEMP'),
            checkpoint_freq: int = 10,
            checkpoint_render_n_games: int = 10,
    ):
        self.model = model
        self.model.train()
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.env = env
        self.policy_weight = policy_weight
        assert self.policy_weight >= 0.
        self.value_weight = value_weight
        assert self.value_weight >= 0.
        self.entropy_weight = entropy_weight
        assert self.entropy_weight >= 0.
        self.use_action_masking = use_action_masking
        self.use_mixed_precision = use_mixed_precision
        self.grad_scaler = grad_scaler
        self.clip_grads = clip_grads
        if self.clip_grads is not None and self.clip_grads < 1.:
            raise ValueError(f'Clip_grads should be None for no clipping or >= 1, was {self.clip_grads:.2f}')
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
                obs_type=self.env.obs_type,
                reward_type=ge.RewardType.RANK_ON_DEATH,
                action_masking=ge.ActionMasking.LETHAL,
                n_envs=max(self.checkpoint_render_n_games, 20),
                silent_reset=False,
                make_fn=make
            )
        else:
            self.rendering_env_kwargs = None

        self.batch_counter = 0
        self.train_step_counter = 0
        self.summary_writer = SummaryWriter(str(self.exp_folder))

        dummy_state = torch.zeros(self.env.obs_type.get_obs_spec()[1:])
        dummy_head_loc = torch.arange(4).to(dtype=torch.int64)
        still_alive = torch.ones(4).to(dtype=torch.bool)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=TracerWarning)
            self.summary_writer.add_graph(
                self.model,
                [dummy_state.unsqueeze(0).to(device=self.env.device),
                 dummy_head_loc.unsqueeze(0).to(device=self.env.device),
                 still_alive.unsqueeze(0).to(device=self.env.device)]
            )

    def train(self, n_batches, batch_len=50, gamma=0.99) -> NoReturn:
        self.model.train()
        self.env.force_reset()

        print(f'\nRunning main training loop for {n_batches} batches of {batch_len} steps each.')
        for _ in tqdm.trange(n_batches):
            print('TODO: Compute entropy loss')
            batch_start_time = time.time()
            # alive_buffer tracks whether an agent was alive to act in a given state
            # dead_buffer tracks whether an agent dies after its action
            a_buffer, r_buffer, l_buffer, v_buffer, alive_buffer, dead_buffer = [], [], [], [], [], []
            for _ in range(batch_len):
                step_start_time = time.time()
                s, _, dead = self.env.reset(get_reward_and_dead=True)
                available_actions_mask = ~self.env.get_illegal_action_masks()
                a, (l, v) = self.model.sample_action(
                    states=s,
                    head_locs=self.env.head_locs,
                    still_alive=~dead,
                    available_actions_mask=available_actions_mask,
                    train=True
                )
                # The dead tensor is modified in-place by the environment
                alive_buffer.append(~dead.clone())
                _, r, dead = self.env.step(a, get_reward_and_dead=True)
                a_buffer.append(a)
                r_buffer.append(r)
                l_buffer.append(l)
                v_buffer.append(v)
                # The dead tensor is modified in-place by the environment
                dead_buffer.append(dead.clone())

                self.summary_writer.add_scalar('Time/step_time_ms',
                                               (time.time() - step_start_time) * 1000.,
                                               self.train_step_counter)
                self.train_step_counter += 1
            self.summary_writer.add_scalar('Time/batch_step_time_ms',
                                           (time.time() - batch_start_time) * 1000.,
                                           self.batch_counter)

            a_tensor = torch.stack(a_buffer, dim=0)
            l_tensor = torch.stack(l_buffer, dim=0)
            v_tensor = torch.stack(v_buffer, dim=0)
            alive_tensor = torch.stack(alive_buffer, dim=0)
            train_start_time = time.time()
            _, v_final = self.model(
                states=self.env.obs,
                head_locs=self.env.head_locs,
                still_alive=self.env.alive,
            ).detach()
            td_target = compute_td_target(v_final, r_buffer, dead_buffer, gamma)
            td_target_masked = td_target[alive_tensor]
            value_masked = v_tensor[alive_tensor]
            advantage = td_target_masked - value_masked
            critic_loss = F.smooth_l1_loss(value_masked, td_target_masked)
            weighted_critic_loss = critic_loss * self.value_weight

            log_probs = l_tensor[alive_tensor].gather(-1, a_tensor[alive_tensor])
            actor_loss = -(log_probs * advantage.detach()).mean()
            weighted_actor_loss = actor_loss * self.policy_weight

            # TODO: Compute entropy loss in a stable fashion
            entropy_loss = torch.zeros(1).to(self.env.device)
            weighted_entropy_loss = entropy_loss * self.entropy_weight

            total_loss = weighted_critic_loss + weighted_actor_loss + weighted_entropy_loss
            self.log_batch({
                'actor_loss': actor_loss.detach().cpu().numpy().item(),
                'weighted_actor_loss': weighted_actor_loss.detach().cpu().numpy().item(),
                'critic_loss': critic_loss.detach().cpu().numpy().item(),
                'weighted_critic_loss': weighted_critic_loss.detach().cpu().numpy().item(),
                'entropy_loss': entropy_loss.detach().cpu().numpy().item(),
                'weighted_entropy_loss': entropy_loss.detach().cpu().numpy().item(),
                'total_loss': entropy_loss.detach().cpu().numpy().item(),
            })

            self.optimizer.zero_grad()
            if self.use_mixed_precision:
                self.grad_scaler.scale(total_loss).backward()
                if self.clip_grads is not None:
                    self.grad_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grads)
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                total_loss.backward()
                if self.clip_grads is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grads)
                self.optimizer.step()
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning)
                self.lr_scheduler.step()

            if self.batch_counter % self.checkpoint_freq == 0 and self.batch_counter != 0:
                self.checkpoint()
            self.summary_writer.add_scalar('Time/batch_train_time_ms',
                                           (time.time() - train_start_time) * 1000.,
                                           self.batch_counter)
            self.summary_writer.add_scalar('Time/batch_total_time_ms',
                                           (time.time() - batch_start_time) * 1000.,
                                           self.batch_counter)
            self.batch_counter += 1
        self.save(self.exp_folder, finished=True)

    def log_batch(self, log_dict: Dict[str, float]) -> NoReturn:
        for key, val in log_dict.items():
            self.summary_writer.add_scalar(f'Batch/{key}', val, self.batch_counter)

    def checkpoint(self) -> NoReturn:
        checkpoint_start_time = time.time()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.summary_writer.add_histogram(
                    f'params/{name}',
                    param.detach().cpu().clone().numpy(),
                    self.batch_counter
                )
        checkpoint_dir = self.exp_folder / f'{self.batch_counter:04}'
        checkpoint_dir.mkdir()
        self.render_n_games(checkpoint_dir)
        self.save(checkpoint_dir)
        self.summary_writer.add_scalar('Time/checkpoint_time_s',
                                       (time.time() - checkpoint_start_time),
                                       int(self.batch_counter / self.checkpoint_freq))

    def render_n_games(self, checkpoint_dir: Path) -> NoReturn:
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
                ).to(device=self.env.device).view(-1, 4)
                still_alive = torch.from_numpy(
                    ~agent_dones
                ).to(device=self.env.device).view(-1, 4)
                available_actions_mask = torch.from_numpy(
                    info_dict['available_actions_mask']
                ).to(device=self.env.device).view(-1, rendering_env.n_players, 4)
                a = self.model.choose_best_action(
                    s.to(device=self.env.device).view(-1, *s_shape[-3:]),
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

    def save(self, save_dir: Path, finished: bool = False) -> NoReturn:
        if finished:
            save_dir = save_dir / f'final_{self.batch_counter}'
            save_dir.mkdir()
        # Save model params
        self.model.cpu()
        state_dict_bytes = pickle.dumps(self.model.state_dict())
        serialized_string = base64.b64encode(state_dict_bytes)
        with open(save_dir / 'cp.txt', 'w') as f:
            f.write(str(serialized_string))
        self.model.to(device=self.env.device)


def compute_td_target(
    v_final: torch.Tensor,
    r_buffer: List[torch.Tensor],
    dead_buffer: List[torch.Tensor],
    gamma: float
) -> torch.Tensor:
    td_target = []
    v_next_s = v_final

    for r, dead in zip(r_buffer[::-1], dead_buffer[::-1]):
        v_next_s = r + gamma * v_next_s * (~dead).float()
        td_target.append(v_next_s)

    return torch.cat(td_target[::-1])
