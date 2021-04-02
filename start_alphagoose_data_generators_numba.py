from pathlib import Path
import torch
from torch import nn

from hungry_geese.training.alphagoose.alphagoose_data_generator import get_latest_weights_file
from hungry_geese.training.alphagoose.alphagoose_data_generator_numba import start_selfplay_loop
from hungry_geese.env import goose_env as ge
from hungry_geese.nns import models, conv_blocks

if __name__ == '__main__':
    device = torch.device('cuda')

    obs_type = ge.ObsType.COMBINED_GRADIENT_OBS
    n_channels = 64
    activation = nn.ReLU
    normalize = False
    use_mhsa = True
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
                use_mhsa=True,
                mhsa_heads=4,
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
    model = models.FullConvActorCriticNetwork(**model_kwargs)

    weights_dir = Path(
        'runs/alphagoose/alphagoose_combined_gradient_obs_rank_on_death_lethal_5_blocks_64_dims_v0/all_checkpoints_pt'
    )
    print(f'Loading initial model weights from: {get_latest_weights_file(weights_dir)}')

    dataset_dir = Path('/home/isaiah/data/alphagoose_data')
    dataset_dir.mkdir(exist_ok=True)
    print(f'Saving self-play data to: {dataset_dir}')

    start_selfplay_loop(
        model=model,
        device=device,
        dataset_dir=dataset_dir,
        weights_dir=weights_dir,
        max_saved_batches=10000,
        obs_type=obs_type,
        allow_resume=True
    )
