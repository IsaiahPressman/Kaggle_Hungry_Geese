from contextlib import redirect_stdout
import io
# Silence "Loading environment football failed: No module named 'gfootball'" message
with redirect_stdout(io.StringIO()):
    import kaggle_environments
import numpy as np
import os
from pathlib import Path
import shutil
import torch
from torch import nn

from hungry_geese.config import N_ROWS, N_COLS
from hungry_geese.nns import conv_blocks
import hungry_geese.env.goose_env as ge
from hungry_geese.training.impala import Flags, Impala
from hungry_geese.utils import format_experiment_name

os.environ["OMP_NUM_THREADS"] = "1"

if __name__ == '__main__':
    obs_type = ge.ObsType.COMBINED_GRADIENT_OBS_FULL
    #n_heads = 4
    #n_channels = n_heads * 32
    n_channels = 64
    activation = nn.ReLU
    normalize = False
    use_preprocessing = False
    model_kwargs = dict(
        preprocessing_layer=nn.Sequential(
            nn.Conv2d(
                obs_type.get_obs_spec()[-3],
                n_channels,
                (1, 1)
            ),
            activation(),
            nn.Conv2d(
                n_channels,
                n_channels,
                (1, 1)
            )
        ) if use_preprocessing else None,
        block_class=conv_blocks.BasicConvolutionalBlock,
        block_kwargs=[
            dict(
                in_channels=n_channels if use_preprocessing else obs_type.get_obs_spec()[-3],
                out_channels=n_channels,
                kernel_size=3,
                activation=activation,
                normalize=normalize,
            ),
            *[dict(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=3,
                activation=activation,
                normalize=normalize,
            ) for _ in range(5)]
        ],
        base_out_channels=n_channels,
        actor_critic_activation=activation,
        n_action_value_layers=1,
        cross_normalize_value=True,
        use_separate_action_value_heads=True,
        # **ge.RewardType.RANK_ON_DEATH.get_recommended_value_activation_scale_shift_dict()
    )

    flags = Flags(
        # env params
        n_envs=512,
        obs_type=obs_type,

        # actor params
        batch_len=20,
        use_action_masking=True,
        actor_device=torch.device('cuda:1'),
        num_buffers=15,
        max_queue_len=2,

        # learner params
        n_batches=200_000,
        batch_size=128,
        gamma=0.9995,
        baseline_cost=1.,
        entropy_cost=2e-3,
        linear_entropy_decay_target=0.1,
        use_mixed_precision=True,
        reduction='mean',
        clip_grads=10.,
        learner_device=torch.device('cuda:0'),
    )

    experiment_name = 'impala_' + format_experiment_name(
        obs_type,
        ge.RewardType.RANK_ON_DEATH,
        ge.ActionMasking.OPPOSITE,
        [n_channels],
        model_kwargs.get('block_kwargs', model_kwargs.get('base_model', [None])[:-1])
    ) + '_v0'
    exp_folder = Path(f'runs/impala/active/{experiment_name}')

    lr = 0.001
    min_lr = lr * 0.01
    min_lr_mod = np.linspace(min_lr / lr, 1., flags.n_batches)[::-1]

    def lr_lambda(epoch):
        try:
            return min_lr_mod[epoch]
        except IndexError:
            print(f'Failed to index min_lr_mod with length {len(min_lr_mod)} at epoch #{epoch}')
            return min_lr

    train_alg = Impala(
        flags=flags,
        model_kwargs=model_kwargs,
        optimizer_class=torch.optim.RMSprop,
        optimizer_kwargs=dict(
            lr=lr,
            alpha=0.9,
            # momentum=0.,
            # eps=0.01,
            # weight_decay=1e-5,
        ),
        lr_scheduler_class=torch.optim.lr_scheduler.LambdaLR,
        lr_scheduler_kwargs=dict(
            lr_lambda=lr_lambda
        ),
        exp_folder=exp_folder,
        checkpoint_freq=20.,
        checkpoint_render_n_games=5,
    )
    this_script = Path(__file__).absolute()
    shutil.copy(this_script, train_alg.exp_folder / f'_{this_script.name}')

    try:
        train_alg.train()
    except KeyboardInterrupt:
        print('KeyboardInterrupt: saving model')
        train_alg.save(train_alg.exp_folder, finished=True)
