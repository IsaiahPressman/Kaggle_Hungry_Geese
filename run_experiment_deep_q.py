import base64
import copy
from pathlib import Path
import pickle
import shutil
import torch
from torch import nn

from rl.deep_q import DeepQ
from rl import goose_env as ge
from rl import models
from rl.replay_buffers import *
from rl.utils import format_experiment_name

if __name__ == '__main__':
    DEVICE = torch.device('cuda')

    obs_type = ge.ObsType.HEAD_CENTERED_OBS
    reward_type = ge.RewardType.RANK_ON_DEATH
    """
    initial_out_channels = 64
    initial_downsample = nn.Conv2d(
        in_channels=initial_out_channels,
        out_channels=initial_out_channels,
        kernel_size=3,
        stride=2,
        padding=1,
        padding_mode='circular'
    )"""
    n_channels = [64, 128]
    first_downsample = nn.AvgPool2d(2, stride=1)
    model_kwargs = dict(conv_block_kwargs=[
        dict(
            in_channels=obs_type.get_obs_spec()[2],
            out_channels=n_channels[0],
            kernel_size=3,
            downsample=nn.Identity()
        ),
        #dict(
        #    in_channels=n_channels[0],
        #    out_channels=n_channels[0],
        #    kernel_size=3,
        #    downsample=nn.Identity(),
        #),
        dict(
            in_channels=n_channels[0],
            out_channels=n_channels[1],
            kernel_size=3,
            downsample=first_downsample
        ),

        dict(
            in_channels=n_channels[1],
            out_channels=n_channels[1],
            kernel_size=3,
            downsample=nn.Identity(),
        ),
        #dict(
        #    in_channels=n_channels[1],
        #    out_channels=n_channels[1],
        #    kernel_size=3,
        #    downsample=nn.Identity(),
        #),
        dict(
            in_channels=n_channels[1],
            out_channels=n_channels[1],
            kernel_size=3,
            downsample=nn.AvgPool2d(2),
        )
    ],
        use_adaptive_avg_pool=True,
        # fc_in_channels=n_channels[1] * 3 * 5,
        fc_in_channels=n_channels[1]
        # **reward_type.get_recommended_value_activation_scale_shift_dict()
    )
    model = models.DeepQNetwork(
        optimizer_constructor=lambda params: torch.optim.Adam(params,
                                                              weight_decay=1e-5,
                                                              ),
        epsilon=0.1,
        delayed_updates=True,
        double_q=True,
        dueling_q=True,
        tau=5e-3,
        **model_kwargs
    )
    """
    with open(f'runs/awac/attention_4_32_0_norm_v1/final_81_cp.txt', 'r') as f:
        serialized_string = f.readline()[2:-1].encode()
    state_dict_bytes = base64.b64decode(serialized_string)
    loaded_state_dicts = pickle.loads(state_dict_bytes)
    model.load_state_dict(loaded_state_dicts['model_state_dict'])
    """
    model.to(device=DEVICE)

    env_kwargs = dict(
        n_envs=32,
        obs_type=obs_type,
        reward_type=reward_type
    )
    rl_train_kwargs = dict(
        batch_size=512,
        n_steps_per_epoch=200,
        n_train_batches_per_epoch=None,
        gamma=0.995,
    )

    """
    replay_s_a_r_d_s = load_s_a_r_d_s(
        '/home/isaiah/GitHub/Kaggle/Santa_2020/episode_scraping/latest_250_replays_database_SUMMED_OBS_WITH_TIMESTEP/'
    )"""
    replay_buffer = DataAugmentationReplayBuffer(
        obs_type=obs_type,
        use_channel_shuffle=True,
        s_shape=obs_type.get_obs_spec()[-3:],
        max_len=2e5,
        starting_s_a_r_d_s=None,
        #starting_s_a_r_d_s=replay_s_a_r_d_s
    )
    # Conserve memory
    # del replay_s_a_r_d_s

    validation_kwargs_base = dict(
        n_envs=10,
        obs_type=obs_type
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
    deep_q_alg = DeepQ(model, ge.GooseEnvVectorized(**env_kwargs), replay_buffer,
                       validation_kwargs_dicts=validation_kwargs_dicts,
                       device=DEVICE,
                       exp_folder=Path(f'runs/deep_q/{experiment_name}'),
                       use_action_masking=True,
                       clip_grads=10.,
                       checkpoint_freq=5,
                       checkpoint_render_n_games=10)
    this_script = Path(__file__).absolute()
    shutil.copy(this_script, deep_q_alg.exp_folder / f'_{this_script.name}')

    try:
        deep_q_alg.train(
            n_epochs=10000,
            **rl_train_kwargs
        )
    except KeyboardInterrupt:
        if deep_q_alg.epoch_counter > deep_q_alg.checkpoint_freq:
            print('KeyboardInterrupt: saving model')
            deep_q_alg.save(deep_q_alg.exp_folder, finished=True)
