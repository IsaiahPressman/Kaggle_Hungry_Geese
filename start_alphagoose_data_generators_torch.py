from kaggle_environments.envs.hungry_geese.hungry_geese import Configuration
from kaggle_environments import make as kaggle_make
from pathlib import Path
import torch
from torch import nn
from torch import multiprocessing as mp

from hungry_geese.training.alphagoose.alphagoose_data_generator import get_latest_weights_file
from hungry_geese.training.alphagoose.alphagoose_data_generator_torch import start_alphagoose_data_generator
from hungry_geese.env import goose_env as ge
from hungry_geese.nns import conv_blocks
from hungry_geese.utils import format_experiment_name


def main():
    device = torch.device('cuda:0')

    obs_type = ge.ObsType.COMBINED_GRADIENT_OBS_LARGE
    n_channels = 64
    activation = nn.ReLU
    normalize = False
    use_mhsa = False
    model_kwargs = dict(
        block_class=conv_blocks.BasicConvolutionalBlock,
        conv_block_kwargs=[
            dict(
                in_channels=obs_type.get_obs_spec()[-3],
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
                use_mhsa=use_mhsa,
                mhsa_heads=4,
            ),
        ],
        squeeze_excitation=True,
        cross_normalize_value=True,
        use_separate_action_value_heads=True,
        # **ge.RewardType.RANK_ON_DEATH.get_recommended_value_activation_scale_shift_dict()
    )

    experiment_name = 'alphagoose_' + format_experiment_name(obs_type,
                                                             ge.RewardType.RANK_ON_DEATH,
                                                             ge.ActionMasking.OPPOSITE,
                                                             [n_channels],
                                                             model_kwargs['conv_block_kwargs']) + '_v0'
    weights_dir = Path(
        f'runs/alphagoose/active/{experiment_name}/all_checkpoints_pt'
    )
    print(f'Loading initial model weights from: {get_latest_weights_file(weights_dir)}')

    dataset_dir = Path('/home/isaiah/data/alphagoose_data')
    dataset_dir.mkdir(exist_ok=True)
    print(f'Saving self-play data to: {dataset_dir}')

    env_kwargs = dict(
        config=Configuration(kaggle_make('hungry_geese', debug=False).configuration),
        n_envs=5000,
        obs_type=obs_type,
        device=device,
    )

    mcts_kwargs = dict(
        n_iter=200,
        c_puct=1.,
        add_noise=True,
        noise_val=2.,
        noise_weight=0.25,
        device=device,
    )
    # TorchMCTSTree kwargs
    mcts_kwargs.update(dict(
        n_envs=env_kwargs['n_envs'],
        max_size=mcts_kwargs['n_iter'] + 10,
        policy_temp=1.,
    ))

    start_alphagoose_data_generator(
        dataset_dir=dataset_dir,
        weights_dir=weights_dir,
        allow_resume=True,
        max_saved_batches=10000,
        update_model_freq=10,
        env_kwargs=env_kwargs,
        model_kwargs=model_kwargs,
        mcts_kwargs=mcts_kwargs,
    )


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
