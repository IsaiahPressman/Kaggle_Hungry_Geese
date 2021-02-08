import base64
import copy
import numpy as np
from pathlib import Path
import pickle
import shutil
import torch
from torch import nn

from rl.awac import AWAC, BasicReplayBuffer
from rl import goose_env as ge
from rl import models
from rl.utils import format_experiment_name

if __name__ == '__main__':
    DEVICE = torch.device('cuda')

    initial_out_channels = 64
    initial_downsample = nn.Conv2d(
        in_channels=initial_out_channels,
        out_channels=initial_out_channels,
        kernel_size=3,
        stride=2,
        padding=1,
        padding_mode='circular'
    )
    model_kwargs = dict(conv_block_kwargs=[
        dict(
            in_channels=13,
            out_channels=initial_out_channels,
            kernel_size=5,
            downsample=initial_downsample
        ),
        dict(
            in_channels=initial_out_channels,
            out_channels=128,
            kernel_size=5,
            downsample=nn.AvgPool2d(2),
        )
    ],
        fc_in_channels=128 * 2 * 3
    )
    model = models.BasicActorCriticNetwork(**model_kwargs)
    """
    with open(f'runs/awac/attention_4_32_0_norm_v1/final_81_cp.txt', 'r') as f:
        serialized_string = f.readline()[2:-1].encode()
    state_dict_bytes = base64.b64decode(serialized_string)
    loaded_state_dicts = pickle.loads(state_dict_bytes)
    model.load_state_dict(loaded_state_dicts['model_state_dict'])
    """
    model.to(device=DEVICE)
    optimizer = torch.optim.Adam(model.parameters(),
                                 weight_decay=1e-5
                                 )

    env_kwargs = dict(
        n_envs=32,
        obs_type=ge.HEAD_CENTERED_OBS,
        reward_type=ge.RANK_ON_DEATH
    )
    rl_train_kwargs = dict(
        batch_size=1024,
        #n_pretrain_batches=10000,
        n_pretrain_batches=0,
        n_steps_per_epoch=200,
        n_train_batches_per_epoch=None,
        gamma=0.995,
        lagrange_multiplier=1.
    )

    """
    replay_s_a_r_d_s = load_s_a_r_d_s(
        '/home/isaiah/GitHub/Kaggle/Santa_2020/episode_scraping/latest_250_replays_database_SUMMED_OBS_WITH_TIMESTEP/'
    )"""
    replay_buffer = BasicReplayBuffer(
        s_shape=ge.GooseEnvVectorized.get_obs_spec(env_kwargs['obs_type'])[-3:],
        max_len=5e5,
        starting_s_a_r_d_s=None,
        #starting_s_a_r_d_s=replay_s_a_r_d_s
    )
    # Conserve memory
    # del replay_s_a_r_d_s

    validation_kwargs_base = dict(
        n_envs=10,
        obs_type=env_kwargs['obs_type']
    )
    validation_opponent_kwargs = [
        # dict(
        #     opponent=va.BasicThompsonSampling(),
        #     opp_obs_type=ve.SUMMED_OBS
        # ),
    ]
    validation_kwargs_dicts = []
    for opponent_kwargs in validation_opponent_kwargs:
        validation_kwargs_dicts.append(copy.copy(validation_kwargs_base))
        validation_kwargs_dicts[-1].update(opponent_kwargs)

    experiment_name = format_experiment_name()
    awac_alg = AWAC(model, optimizer, ge.GooseEnvVectorized(**env_kwargs), replay_buffer,
                    validation_kwargs_dicts=validation_kwargs_dicts,
                    deterministic_validation_policy=True,
                    device=DEVICE,
                    q_val_clamp=(-1., 1.),
                    exp_folder=Path(f'runs/awac/{experiment_name}'),
                    clip_grads=10.,
                    checkpoint_freq=5,
                    checkpoint_render_n_games=10)
    this_script = Path(__file__).absolute()
    shutil.copy(this_script, awac_alg.exp_folder / f'_{this_script.name}')

    try:
        awac_alg.train(
            n_epochs=10000,
            **rl_train_kwargs
        )
    except KeyboardInterrupt:
        if awac_alg.epoch_counter > awac_alg.checkpoint_freq:
            print('KeyboardInterrupt: saving model')
            awac_alg.save(awac_alg.exp_folder, finished=True)
