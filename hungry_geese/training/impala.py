from kaggle_environments import make
import os
from pathlib import Path
import shutil
import time
import torch
from torch import multiprocessing as mp
import torch.nn.functional as F
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter
from torch.jit import TracerWarning
import tqdm
from typing import *
import warnings

# Custom imports
from ..config import N_PLAYERS
from ..env import goose_env as ge
from ..env.torch_env import TorchEnv
from ..nns.models import FullConvActorCriticNetwork
from . import vtrace

Buffers = Dict[str, torch.Tensor]


class Batch(NamedTuple):
    # Env outputs
    states: torch.Tensor
    head_locs: torch.Tensor
    alive_before_act: torch.Tensor
    available_actions_mask: torch.Tensor
    reward: torch.Tensor
    alive_after_act: torch.Tensor

    # Agent outputs
    actions: torch.Tensor
    log_probs: torch.Tensor
    baseline: torch.Tensor


class Flags(NamedTuple):
    # env params
    n_envs: int
    obs_type: ge.ObsType

    # actor params
    batch_len: int
    use_action_masking: bool
    actor_device: torch.device
    num_buffers: int
    max_queue_len: int

    # learner params
    n_batches: int
    batch_size: int
    gamma: float
    baseline_cost: float
    entropy_cost: float
    linear_entropy_decay_target: float
    use_mixed_precision: bool
    reduction: str
    clip_grads: Optional[float]
    learner_device: torch.device

    def validate_flags(self) -> NoReturn:
        assert self.n_envs % self.batch_size == 0
        assert self.baseline_cost >= 0
        assert self.entropy_cost >= 0
        if self.clip_grads is not None and self.clip_grads < 1.:
            raise ValueError(f'Clip_grads should be None for no clipping or >= 1, was {self.clip_grads:.2f}')
        assert 0 <= self.linear_entropy_decay_target <= 1.


def reduce(losses: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == 'mean':
        return losses.mean()
    elif reduction == 'sum':
        return losses.sum()
    else:
        raise ValueError(f'Reduction must be one of "sum" or "mean", was: {reduction}')


def compute_baseline_loss(advantages: torch.Tensor, reduction: str):
    return reduce(advantages ** 2, reduction=reduction)


def compute_entropy_loss(log_probs: torch.Tensor, reduction: str):
    log_probs_masked_zeroed = torch.where(
        log_probs.detach().isneginf(),
        torch.zeros_like(log_probs),
        log_probs
    )
    """Return the entropy loss, i.e., the negative entropy of the policy."""
    policy = F.softmax(log_probs, dim=-1)
    return reduce((policy * log_probs_masked_zeroed).sum(dim=-1), reduction)


def compute_policy_gradient_loss(
        log_probs: torch.Tensor,
        actions: torch.Tensor,
        advantages: torch.Tensor,
        reduction: str
):
    cross_entropy = F.nll_loss(
        log_probs,
        target=actions,
        reduction="none",
    )
    cross_entropy = cross_entropy.view_as(advantages)
    return reduce(cross_entropy * advantages.detach(), reduction)


def env_out_to_dict(
        env: TorchEnv,
        alive_before_act: torch.Tensor,
        env_out: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
) -> dict[str, torch.Tensor]:
    return {
        'states': env_out[0],
        'head_locs': env.head_locs,
        'alive_before_act': alive_before_act,
        'available_actions_mask': env.get_available_action_masks(),
        'reward': env_out[1],
        'alive_after_act': ~env_out[2]
    }


def agent_out_to_dict(model: FullConvActorCriticNetwork, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    try:
        actions, (log_probs, values) = model.sample_action(train=True, **obs)
    except ValueError as e:
        logits, values = model.forward(
            states=obs['states'],
            head_locs=obs['head_locs'],
            still_alive=obs['still_alive']
        )
        print(f'Nan logits: {torch.isnan(logits).any()}')
        print(f'Nan probs: {torch.isnan(F.softmax(logits, -1)).any()}')
        nan_mask = torch.isnan(F.softmax(logits, -1)).any(dim=-1).any(dim=-1)
        print(obs['head_locs'][nan_mask])
        print(obs['still_alive'][nan_mask])
        print(obs['states'][nan_mask])
        print(obs['states'].shape)
        print(torch.isneginf(obs['states']).any(dim=0).any(dim=-1).any(dim=-1))
        raise e
    return {
        'actions': actions,
        'log_probs': log_probs,
        'baseline': values
    }


def extract_obs(tensor_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        'states': tensor_dict['states'],
        'head_locs': tensor_dict['head_locs'],
        # model.sample_action() takes still_alive as a kwarg, not alive_before_act
        'still_alive': tensor_dict['alive_before_act'],
        'available_actions_mask': tensor_dict['available_actions_mask']
    }


def to_cpu(t: torch.Tensor) -> torch.Tensor:
    return t.to(device=torch.device('cpu'), non_blocking=True)


def create_buffers(flags: Flags) -> tuple[Buffers, Buffers]:
    t = (flags.batch_len + 1) * flags.num_buffers
    n = flags.n_envs
    g = N_PLAYERS
    train_buffer_specs = dict(
        # Env outputs
        states=dict(size=(t, n, *flags.obs_type.get_obs_spec()[1:]), dtype=torch.float32),
        head_locs=dict(size=(t, n, g), dtype=torch.int64),
        alive_before_act=dict(size=(t, n, g), dtype=torch.bool),
        available_actions_mask=dict(size=(t, n, g, 4), dtype=torch.bool),
        reward=dict(size=(t, n, g), dtype=torch.float32),
        alive_after_act=dict(size=(t, n, g), dtype=torch.bool),

        # Agent outputs
        actions=dict(size=(t, n, g), dtype=torch.int64),
        log_probs=dict(size=(t, n, g, 4), dtype=torch.float32),
        baseline=dict(size=(t, n, g), dtype=torch.float32),

        # Misc
        write_index=dict(size=(1,), dtype=torch.int64),
        read_index=dict(size=(1,), dtype=torch.int64),
    )
    log_buffer_specs = dict(
        game_counter=dict(size=(flags.num_buffers,), dtype=torch.int64),
    )
    for val in ['game_length', 'final_goose_length', 'winning_goose_length']:
        log_buffer_specs[f'Results/median_{val}'] = dict(size=(flags.num_buffers,), dtype=torch.float)
        log_buffer_specs[f'Results/mean_{val}'] = dict(size=(flags.num_buffers,), dtype=torch.float)

    train_buffers = {key: torch.zeros(**spec).share_memory_() for key, spec in train_buffer_specs.items()}
    log_buffers = {key: torch.zeros(**spec).share_memory_() for key, spec in log_buffer_specs.items()}
    return train_buffers, log_buffers


def get_batches(
        flags: Flags,
        train_buffers: Buffers
) -> list[Batch]:
    while train_buffers['read_index'] >= train_buffers['write_index']:
        time.sleep(0.25)

    batches = []
    read_idx = train_buffers['read_index'] % flags.num_buffers
    time_slice_start = read_idx * (flags.batch_len + 1)
    time_slice_end = (read_idx + 1) * (flags.batch_len + 1)
    for batch_slice in range(0, flags.n_envs, flags.batch_size):
        batches.append(Batch(
            **{
                key: train_buffers[key][
                     time_slice_start:time_slice_end,
                     batch_slice:(batch_slice + flags.batch_size),
                     ...
                     ] for key in Batch._fields
            }
        ))
    train_buffers['read_index'] += 1
    return batches


def actor(
        flags: Flags,
        model: FullConvActorCriticNetwork,
        train_buffers: Buffers,
        log_buffers: Buffers,
        **env_kwargs
):
    try:
        model.eval()
        env = TorchEnv(n_envs=flags.n_envs, obs_type=flags.obs_type, device=flags.actor_device, **env_kwargs)
        alive_before_act = env.alive.clone()
        env_output = env_out_to_dict(env, alive_before_act, env.force_reset(get_reward_and_dead=True))
        agent_output = agent_out_to_dict(model, extract_obs(env_output))

        while True:
            if train_buffers['write_index'] - train_buffers['read_index'] > flags.max_queue_len:
                time.sleep(1.)
                continue

            write_index = (train_buffers['write_index'].item() % flags.num_buffers) * (flags.batch_len + 1)
            log_write_index = train_buffers['write_index'].item() % flags.num_buffers
            if write_index < 0:
                break

            # Write old rollout end
            for key in env_output:
                train_buffers[key][write_index, ...] = to_cpu(env_output[key])
            for key in agent_output:
                train_buffers[key][write_index, ...] = to_cpu(agent_output[key])

            # Do new rollout
            game_length_buffer, final_goose_length_buffer, winning_goose_length_buffer = [], [], []
            for t in range(flags.batch_len):
                with torch.no_grad():
                    agent_output = agent_out_to_dict(model, extract_obs(env_output))

                alive_before_act = env.alive.clone()
                env_output = env_out_to_dict(
                    env,
                    alive_before_act,
                    env.step(agent_output['actions'], get_reward_and_dead=True)
                )
                torch.cuda.synchronize(flags.actor_device)
                if env.dones.any():
                    game_length_buffer.append(to_cpu(env.step_counters[env.dones].view(-1)))
                    final_goose_lengths = env.rewards[env.dones] % (env.max_len + 1)
                    final_goose_lengths = torch.maximum(final_goose_lengths, torch.ones_like(final_goose_lengths))
                    final_goose_length_buffer.append(to_cpu(final_goose_lengths.view(-1)))
                    winning_geese_idxs = env.rewards[env.dones].argmax(dim=-1, keepdim=True)
                    winning_goose_lengths = final_goose_lengths.gather(-1, winning_geese_idxs)
                    winning_goose_length_buffer.append(to_cpu(winning_goose_lengths.view(-1)))
                    log_buffers['game_counter'][log_write_index] = log_buffers['game_counter'][log_write_index - 1] + \
                                                                   env.dones.sum().cpu()
                    env.reset()
                    torch.cuda.synchronize(flags.actor_device)
                    env_output['states'] = env.obs
                    env_output['head_locs'] = env.head_locs
                    env_output['alive_before_act'] = env.alive
                    env_output['available_actions_mask'] = env.get_available_action_masks()

                for key in env_output:
                    train_buffers[key][write_index + t + 1, ...] = to_cpu(env_output[key])
                for key in agent_output:
                    train_buffers[key][write_index + t + 1, ...] = to_cpu(agent_output[key])

            for name, value in [
                ('game_length', game_length_buffer),
                ('final_goose_length', final_goose_length_buffer),
                ('winning_goose_length', winning_goose_length_buffer)
            ]:
                if len(game_length_buffer) > 0:
                    value = torch.cat(value).view(-1).float()
                    log_buffers[f'Results/median_{name}'][log_write_index] = torch.quantile(value, q=0.5)
                    log_buffers[f'Results/mean_{name}'][log_write_index] = value.mean()
                else:
                    log_buffers[f'Results/median_{name}'][log_write_index] = float('nan')
                    log_buffers[f'Results/mean_{name}'][log_write_index] = float('nan')

            train_buffers['write_index'][0] += 1
    except KeyboardInterrupt:
        pass
    finally:
        # Delete references to shared CUDA tensors before exiting
        for key in dir():
            if key != 'key' and not key.startswith('_'):
                del globals()[key]
        print('\n\nactor exited gracefully\n\n')


class Impala:
    def __init__(
            self,
            flags: Flags,
            model_kwargs: dict,
            optimizer_class: Callable,
            optimizer_kwargs: dict,
            lr_scheduler_class: Optional[Callable],
            lr_scheduler_kwargs: dict,
            grad_scaler: amp.grad_scaler = amp.GradScaler(),
            exp_folder: Path = Path('runs/impala/TEMP'),
            checkpoint_freq: float = 20.,
            checkpoint_render_n_games: int = 5
    ):
        flags.validate_flags()
        self.flags = flags
        self.learner_model = FullConvActorCriticNetwork(**model_kwargs).to(flags.learner_device)
        self.learner_model.train()
        self.actor_model = FullConvActorCriticNetwork(**model_kwargs).to(flags.actor_device)
        self.actor_model.eval()
        self.actor_model.share_memory()
        self.optimizer = optimizer_class(
            self.learner_model.parameters(),
            **optimizer_kwargs
        )
        if lr_scheduler_class is not None:
            self.lr_scheduler = lr_scheduler_class(
                self.optimizer,
                **lr_scheduler_kwargs
            )
        else:
            self.lr_scheduler = None
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
                obs_type=flags.obs_type,
                reward_type=ge.RewardType.RANK_ON_DEATH,
                action_masking=ge.ActionMasking.LETHAL,
                n_envs=min(self.checkpoint_render_n_games, 20),
                silent_reset=False,
                make_fn=make
            )
        else:
            self.rendering_env_kwargs = None
        self.batch_counter = 0
        self.summary_writer = SummaryWriter(str(self.exp_folder))

        dummy_state = torch.zeros(flags.obs_type.get_obs_spec()[1:])
        dummy_head_loc = torch.arange(4).to(dtype=torch.int64)
        alive_before_act = torch.ones(4).to(dtype=torch.bool)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=TracerWarning)
            self.summary_writer.add_graph(
                self.learner_model,
                [dummy_state.unsqueeze(0).to(device=flags.learner_device),
                 dummy_head_loc.unsqueeze(0).to(device=flags.learner_device),
                 alive_before_act.unsqueeze(0).to(device=flags.learner_device)]
            )

    def train(self) -> NoReturn:
        initial_start_time = time.time()
        last_checkpoint_time = initial_start_time
        os.environ["OMP_NUM_THREADS"] = "1"
        mp.set_start_method("spawn")

        train_buffers, log_buffers = create_buffers(self.flags)
        actor_proc = mp.Process(
            target=actor,
            kwargs=dict(
                flags=self.flags,
                model=self.actor_model,
                train_buffers=train_buffers,
                log_buffers=log_buffers
            )
        )
        actor_proc.start()

        try:
            batches = []
            for _ in tqdm.trange(self.flags.n_batches):
                need_new_batches = len(batches) == 0
                if need_new_batches:
                    batches.extend(get_batches(self.flags, train_buffers))
                with amp.autocast(enabled=self.flags.use_mixed_precision):
                    self._train_on_batch(batches.pop())
                # noinspection PyTypeChecker
                self.actor_model.load_state_dict(self.learner_model.state_dict())
                if time.time() - last_checkpoint_time > self.checkpoint_freq * 60.:
                    self.checkpoint()
                    last_checkpoint_time = time.time()
                buffered_batches = (train_buffers['write_index'] - train_buffers['read_index']) * \
                                   self.flags.n_envs / self.flags.batch_size
                self.log_misc(
                    log_buffers,
                    train_buffers['read_index'].item() % self.flags.num_buffers,
                    buffered_batches + len(batches),
                    initial_start_time,
                    need_new_batches
                )
                self.batch_counter += 1
            self.save(self.exp_folder, finished=True)
        finally:
            actor_proc.terminate()
            actor_proc.join()
            actor_proc.close()
            time.sleep(1.)

    def _train_on_batch(self, batch: Batch) -> NoReturn:
        batch_start_time = time.time()

        batch = Batch(**{
            key: val.to(self.flags.learner_device, non_blocking=True) for key, val in batch._asdict().items()
        })
        obs = {
            key: torch.flatten(val, 0, 1) for key, val in extract_obs(batch._asdict()).items()
        }
        learner_outputs = agent_out_to_dict(self.learner_model, obs)
        learner_outputs = {
            key: val.view(self.flags.batch_len + 1, self.flags.batch_size, *val.shape[1:])
            for key, val in learner_outputs.items()
        }

        # Take final value function slice for bootstrapping
        bootstrap_value = learner_outputs['baseline'][-1]

        # Move from obs[t] -> action[t] to action[t] -> obs[t]
        learner_alive_before_act = batch.alive_before_act[:-1]
        batch = Batch(**{key: tensor[1:] for key, tensor in batch._asdict().items()})
        learner_outputs = {key: tensor[:-1] for key, tensor in learner_outputs.items()}

        discounts = batch.alive_after_act.float() * self.flags.gamma

        vtrace_returns = vtrace.from_log_probs(
            behavior_policy_log_probs=batch.log_probs,
            target_policy_log_probs=learner_outputs['log_probs'],
            actions=batch.actions,
            discounts=discounts,
            rewards=batch.reward,
            values=learner_outputs['baseline'],
            bootstrap_value=bootstrap_value
        )

        log_probs_masked = learner_outputs['log_probs'][learner_alive_before_act]
        pg_loss = compute_policy_gradient_loss(
            log_probs_masked,
            batch.actions[learner_alive_before_act],
            vtrace_returns.pg_advantages[learner_alive_before_act],
            reduction=self.flags.reduction
        )
        baseline_loss = self.flags.baseline_cost * compute_baseline_loss(
            vtrace_returns.vs[learner_alive_before_act] - learner_outputs['baseline'][learner_alive_before_act],
            reduction=self.flags.reduction
        )
        pct_complete = self.batch_counter / self.flags.n_batches
        entropy_decay = 1. - (1. - self.flags.linear_entropy_decay_target) * pct_complete
        entropy_cost = self.flags.entropy_cost * entropy_decay
        entropy_loss = entropy_cost * compute_entropy_loss(
            log_probs_masked,
            reduction=self.flags.reduction
        )
        total_loss = pg_loss + baseline_loss + entropy_loss

        assert not torch.isnan(pg_loss)
        assert not torch.isnan(baseline_loss)
        assert not torch.isnan(entropy_loss)

        self.optimizer.zero_grad()
        if self.flags.use_mixed_precision:
            self.grad_scaler.scale(total_loss).backward()
            if self.flags.clip_grads is not None:
                self.grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.learner_model.parameters(), self.flags.clip_grads)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            total_loss.backward()
            if self.flags.clip_grads is not None:
                torch.nn.utils.clip_grad_norm_(self.learner_model.parameters(), self.flags.clip_grads)
            self.optimizer.step()
        if self.lr_scheduler is not None:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning)
                self.lr_scheduler.step()

        # Logging
        self.summary_writer.add_scalar('Misc/entropy_cost',
                                       entropy_cost,
                                       self.batch_counter)
        self.log_losses({
            'pg_loss': pg_loss.detach().cpu().item(),
            'baseline_loss': baseline_loss.detach().cpu().item(),
            'entropy_loss': entropy_loss.detach().cpu().item(),
            'total_loss': total_loss.detach().cpu().item(),
        })
        self.summary_writer.add_scalar('Time/batch', time.time() - batch_start_time, self.batch_counter)

    def log_losses(self, log_dict: Dict[str, float]) -> NoReturn:
        for key, val in log_dict.items():
            self.summary_writer.add_scalar(f'Loss/{key}', val, self.batch_counter)

    def log_misc(
            self,
            log_buffers: Buffers,
            read_idx: int,
            n_batches_queued: int,
            initial_start_time: float,
            new_batches: bool
    ) -> NoReturn:
        # Game stats
        if new_batches:
            game_counter = log_buffers['game_counter'][read_idx].item()
            self.summary_writer.add_scalar('Misc/total_games_played',
                                           game_counter,
                                           self.batch_counter)
            self.summary_writer.add_scalar('Misc/games_per_second',
                                           game_counter / (time.time() - initial_start_time),
                                           self.batch_counter)
            if game_counter > 0:
                self.summary_writer.add_scalar('Time/game',
                                               (time.time() - initial_start_time) / game_counter,
                                               self.batch_counter)

        # Learning rate schedule
        if self.lr_scheduler is not None:
            last_lr = self.lr_scheduler.get_last_lr()
            assert len(last_lr) == 1, 'Logging per-parameter LR still needs support'
            last_lr = last_lr[0]
        else:
            last_lr = self.optimizer.param_groups[0]['lr']
        self.summary_writer.add_scalar('Misc/learning_rate',
                                       last_lr,
                                       self.batch_counter)

        # Number of batches in queue
        self.summary_writer.add_scalar('Misc/queue_size', n_batches_queued, self.batch_counter)

        # Other logging
        if new_batches:
            for key, val in log_buffers.items():
                if key == 'game_counter':
                    continue
                if not torch.isnan(val[read_idx]).item():
                    self.summary_writer.add_scalar(key, val[read_idx].item(), self.batch_counter)

    def checkpoint(self) -> NoReturn:
        checkpoint_start_time = time.time()
        for name, param in self.learner_model.named_parameters():
            if param.requires_grad:
                self.summary_writer.add_histogram(
                    f'params/{name}',
                    param.detach().cpu().clone().numpy(),
                    self.batch_counter
                )
        checkpoint_dir = self.exp_folder / f'{self.batch_counter:06}'
        checkpoint_dir.mkdir()
        self.render_n_games(checkpoint_dir)
        self.save(checkpoint_dir)
        self.summary_writer.add_scalar('Time/checkpoint_time_s',
                                       (time.time() - checkpoint_start_time),
                                       int(self.batch_counter / self.checkpoint_freq))

    def render_n_games(self, checkpoint_dir: Path) -> NoReturn:
        self.learner_model.eval()
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
                ).to(device=self.flags.learner_device).view(-1, 4)
                still_alive = torch.from_numpy(
                    ~agent_dones
                ).to(device=self.flags.learner_device).view(-1, 4)
                available_actions_mask = torch.from_numpy(
                    info_dict['available_actions_mask']
                ).to(device=self.flags.learner_device).view(-1, rendering_env.n_players, 4)
                a = self.learner_model.choose_best_action(
                    s.to(device=self.flags.learner_device).view(-1, *s_shape[-3:]),
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
        self.learner_model.train()

    def save(self, save_dir: Union[str, Path], finished: bool = False) -> NoReturn:
        save_dir = Path(save_dir)
        if finished:
            save_dir = save_dir / f'final_{self.batch_counter}'
            save_dir.mkdir()
        # Save model params as cp.pt and model + optimizer params as full_cp.pt
        self.learner_model.cpu()
        torch.save(self.learner_model.state_dict(), save_dir / 'cp.pt')
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            torch.save({
                'batch_counter': self.batch_counter,
                'model': self.learner_model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None
            }, save_dir / 'full_cp.pt')
        self.learner_model.to(device=self.flags.learner_device)
