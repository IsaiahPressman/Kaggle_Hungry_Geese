import base64
from pathlib import Path
import pickle
import torch
from torch import nn

from hungry_geese.training.alphagoose.alphagoose_data_generator import multiprocess_alphagoose_data_generator
from hungry_geese.env import goose_env as ge
from hungry_geese import models

if __name__ == '__main__':
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

    weights_dir = Path('runs/alphagoose/alphagoose_combined_gradient_obs_rank_on_death_none_3_blocks_128_dims_v1')

    multiprocess_alphagoose_data_generator(
        n_workers=5,
        device=torch.device('cuda'),
        data_dir=Path('/home/isaiah/data/alphagoose_data'),
        max_saved_steps=int(1e6),
        model_kwargs=model_kwargs,
        n_envs_per_worker=20,
        weights_dir=weights_dir,
        obs_type=obs_type,
        model_reload_freq=100,
        n_iter=100,
    )