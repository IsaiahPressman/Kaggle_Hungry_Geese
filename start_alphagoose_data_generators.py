from pathlib import Path
import torch
from torch import nn

from hungry_geese.training.alphagoose import alphagoose_data_generator as adg
from hungry_geese.env import goose_env as ge
from hungry_geese.nns import conv_blocks

if __name__ == '__main__':
    DEVICE = torch.device('cuda')

    obs_type = ge.ObsType.COMBINED_GRADIENT_OBS
    n_channels = 128
    activation = nn.ReLU
    model_kwargs = dict(
        block_class=conv_blocks.BasicConvolutionalBlock,
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

    weights_dir = Path(
        'runs/alphagoose/alphagoose_combined_gradient_obs_rank_on_death_lethal_3_blocks_128_dims_v3/all_checkpoints_pt'
    )
    print(f'Loading initial model weights from: {adg.get_latest_weights_file(weights_dir)}')

    adg.multiprocess_alphagoose_data_generator(
        n_workers=5,
        device=DEVICE,
        dataset_dir=Path('/home/isaiah/data/alphagoose_data'),
        max_saved_episodes=10000,
        model_kwargs=model_kwargs,
        n_envs_per_worker=20,
        weights_dir=weights_dir,
        obs_type=obs_type,
        model_reload_freq=10,
        n_iter=50,
        include_food=False,
        add_noise=False,
        noise_val=1.,
    )
