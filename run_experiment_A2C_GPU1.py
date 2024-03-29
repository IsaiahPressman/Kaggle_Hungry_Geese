import math
from pathlib import Path
import shutil
import torch
from torch import nn

from hungry_geese.config import N_ROWS, N_COLS
from hungry_geese.nns import models, conv_blocks
import hungry_geese.env.goose_env as ge
from hungry_geese.env.torch_env import TorchEnv
from hungry_geese.training.a2c import A2C
from hungry_geese.utils import format_experiment_name

if __name__ == '__main__':
    DEVICE = torch.device('cuda:1')

    obs_type = ge.ObsType.COMBINED_GRADIENT_OBS_FULL
    n_channels = 64
    activation = nn.GELU
    normalize = False
    use_mhsa = False
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
                use_mhsa=False
            ),
            dict(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=3,
                activation=activation,
                normalize=normalize,
                use_mhsa=False
            ),
            dict(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=3,
                activation=activation,
                normalize=normalize,
                use_mhsa=False
            ),
            dict(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=3,
                activation=activation,
                normalize=normalize,
                use_mhsa=False
            ),
            dict(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=3,
                activation=activation,
                normalize=normalize,
                use_mhsa=False
            ),
            dict(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=3,
                activation=activation,
                normalize=normalize,
                use_mhsa=False
            ),
        ],
        n_action_value_layers=2,
        squeeze_excitation=True,
        cross_normalize_value=True,
        use_separate_action_value_heads=True,
        # **ge.RewardType.RANK_ON_DEATH.get_recommended_value_activation_scale_shift_dict()
    )
    model = models.FullConvActorCriticNetwork(**model_kwargs)
    #model.load_state_dict(torch.load('PRETRAINED_LOAD_PATH/cp.pt'))
    model.to(device=DEVICE)
    starting_lr = 0.001
    optimizer = torch.optim.RMSprop(
        model.parameters(),
        lr=starting_lr,
        alpha=0.9,
        momentum=0.,
        #eps=0.01,
        #weight_decay=1e-5,
    )
    # NB: lr_scheduler counts steps in batches, not epochs
    n_batches = int(2e5)
    min_lr = starting_lr * 0.1
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        math.exp(math.log(min_lr / starting_lr) / n_batches)
    )
    #lr_scheduler = None
    env = TorchEnv(
        n_envs=256,
        obs_type=obs_type,
        device=DEVICE,
    )

    experiment_name = 'A2C_' + format_experiment_name(
        obs_type,
        ge.RewardType.RANK_ON_DEATH,
        ge.ActionMasking.OPPOSITE,
        [n_channels],
        model_kwargs.get('block_kwargs', model_kwargs.get('base_model', [None])[:-1])
    ) + '_v4'
    exp_folder = Path(f'runs/A2C/active/GPU1_{experiment_name}')
    train_alg = A2C(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        env=env,
        policy_weight=1.,
        value_weight=1.,
        entropy_weight=1e-4,
        use_action_masking=True,
        use_mixed_precision=True,
        clip_grads=10.,
        exp_folder=exp_folder,
        checkpoint_freq=20.,
        checkpoint_render_n_games=5
    )
    this_script = Path(__file__).absolute()
    shutil.copy(this_script, train_alg.exp_folder / f'_{this_script.name}')

    """
    # Load a previous checkpoint
    train_alg.load_checkpoint(
        'runs/A2C/active/GPU0_A2C_combined_gradient_obs_large_rank_on_death_opposite_4_blocks_92_dims_v0_c0/final_70711'
    )
    """
    try:
        train_alg.train(
            n_batches=n_batches,
            batch_len=5,
            gamma=0.9995
        )
    except KeyboardInterrupt:
        print('KeyboardInterrupt: saving model')
        train_alg.save(train_alg.exp_folder, finished=True)
