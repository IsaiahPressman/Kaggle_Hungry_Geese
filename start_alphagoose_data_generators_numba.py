from pathlib import Path
import torch
from torch import nn

from hungry_geese.training.alphagoose.alphagoose_data_generator import get_latest_weights_file
from hungry_geese.training.alphagoose.alphagoose_data_generator_numba import start_selfplay_loop
from hungry_geese.env import goose_env as ge
from hungry_geese import models

if __name__ == '__main__':
    device = torch.device('cuda')

    obs_type = ge.ObsType.COMBINED_GRADIENT_OBS
    n_channels = 128
    activation = nn.ReLU
    model_kwargs = dict(
        block_class=models.BasicConvolutionalBlock,
        conv_block_kwargs=[
            dict(
                in_channels=obs_type.get_obs_spec()[-3],
                out_channels=n_channels,
                kernel_size=3,
                activation=activation,
                normalize=False
            ),
            dict(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=3,
                activation=activation,
                normalize=False
            ),
            dict(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=3,
                activation=activation,
                normalize=False
            ),
        ],
        squeeze_excitation=True,
        cross_normalize_value=True,
        # **ge.RewardType.RANK_ON_DEATH.get_recommended_value_activation_scale_shift_dict()
    )
    model = models.FullConvActorCriticNetwork(**model_kwargs)

    weights_dir = Path(
        'runs/alphagoose/alphagoose_combined_gradient_obs_rank_on_death_lethal_3_blocks_128_dims_v4/all_checkpoints_pt'
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
