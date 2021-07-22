from contextlib import redirect_stdout
import io
# Silence "Loading environment football failed: No module named 'gfootball'" message
with redirect_stdout(io.StringIO()):
    import kaggle_environments
from pathlib import Path
import torch
from torch import nn

from hungry_geese.nns import conv_blocks
import hungry_geese.env.goose_env as ge
from hungry_geese.training.impala import Flags, Impala
from hungry_geese.utils import format_experiment_name

if __name__ == '__main__':
    obs_type = ge.ObsType.COMBINED_GRADIENT_OBS_FULL
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
            ),
            activation(),
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
            dict(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=3,
                activation=activation,
                normalize=normalize,
            ),
            dict(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=3,
                activation=activation,
                normalize=normalize,
            ),
            dict(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=3,
                activation=activation,
                normalize=normalize,
            ),
        ],
        n_action_value_layers=2,
        squeeze_excitation=True,
        cross_normalize_value=True,
        use_separate_action_value_heads=True,
    )

    flags = Flags(
        # env params
        n_envs=2048,
        obs_type=obs_type,

        # actor params
        batch_len=5,
        use_action_masking=True,
        actor_device=torch.device('cuda:1'),
        num_buffers=10,
        max_queue_len=5,

        # learner params
        n_batches=100_000,
        batch_size=512,
        gamma=0.9995,
        baseline_cost=1.,
        entropy_cost=2e-3,
        linear_entropy_decay_target=0.3,
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

    def lr_lambda(epoch):
        return 1 - min(epoch, flags.n_batches) / flags.n_batches

    train_alg = Impala(
        flags=flags,
        model_kwargs=model_kwargs,
        optimizer_class=torch.optim.RMSprop,
        optimizer_kwargs=dict(
            lr=0.001,
            alpha=0.9,
            momentum=0.,
            # eps=0.01,
            # weight_decay=1e-5,
        ),
        lr_scheduler_class=torch.optim.lr_scheduler.LambdaLR,
        lr_scheduler_kwargs=dict(
            lr_lambda=lr_lambda
        ),
        #exp_folder=exp_folder,
        checkpoint_freq=20.,
        checkpoint_render_n_games=5,
    )

    try:
        train_alg.train()
    except KeyboardInterrupt:
        print('KeyboardInterrupt: saving model')
        train_alg.save(train_alg.exp_folder, finished=True)
