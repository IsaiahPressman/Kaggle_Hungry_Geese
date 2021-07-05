import copy
from pathlib import Path
import shutil
from torch import nn

from awac import AWAC
import models
from replay_buffers import *
from utils import format_experiment_name

if __name__ == '__main__':
    DEVICE = torch.device('cuda')

    obs_type = ge.ObsType.HEAD_CENTERED_OBS_SMALL
    reward_type = ge.RewardType.EVERY_STEP_LENGTH
    action_masking = ge.ActionMasking.LETHAL
    channel_dims = [32, 64, 128]
    first_downsample = nn.AvgPool2d(2, stride=1)
    model_kwargs = dict(conv_block_kwargs=[
        dict(
            in_channels=obs_type.get_obs_spec()[2],
            out_channels=channel_dims[0],
            kernel_size=3,
            downsample=nn.Identity()
        ),
        # dict(
        #    in_channels=channel_dims[0],
        #    out_channels=channel_dims[0],
        #    kernel_size=3,
        #    downsample=nn.Identity(),
        # ),
        dict(
            in_channels=channel_dims[0],
            out_channels=channel_dims[1],
            kernel_size=3,
            downsample=first_downsample
        ),

        dict(
            in_channels=channel_dims[1],
            out_channels=channel_dims[1],
            kernel_size=3,
            downsample=nn.Identity(),
        ),
        # dict(
        #    in_channels=channel_dims[1],
        #    out_channels=channel_dims[1],
        #    kernel_size=3,
        #    downsample=nn.Identity(),
        # ),
        dict(
            in_channels=channel_dims[1],
            out_channels=channel_dims[2],
            kernel_size=3,
            downsample=nn.AvgPool2d(2),
        ),
        dict(
            in_channels=channel_dims[2],
            out_channels=channel_dims[2],
            kernel_size=3,
            downsample=nn.Identity(),
        ),
        # dict(
        #    in_channels=channel_dims[1],
        #    out_channels=channel_dims[1],
        #    kernel_size=3,
        #    downsample=nn.Identity(),
        # ),
        dict(
            in_channels=channel_dims[2],
            out_channels=channel_dims[2],
            kernel_size=3,
            downsample=nn.AvgPool2d(2, stride=1),
        )
    ],
        use_adaptive_avg_pool=False,
        fc_in_channels=channel_dims[2] * 2 * 4,
        # fc_in_channels=channel_dims[1],
        **reward_type.get_recommended_value_activation_scale_shift_dict(),
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
                                 # weight_decay=1e-5
                                 )

    env_kwargs = dict(
        n_envs=16,
        obs_type=obs_type,
        reward_type=reward_type,
        action_masking=action_masking
    )
    rl_train_kwargs = dict(
        batch_size=1024,
        #n_pretrain_batches=10000,
        n_pretrain_batches=0,
        n_steps_per_epoch=200,
        n_train_batches_per_epoch=None,
        gamma=0.99,
        lagrange_multiplier=1.
    )

    """
    replay_s_a_r_d_s = load_s_a_r_d_s(
        '/home/isaiah/GitHub/Kaggle/Santa_2020/episode_scraping/latest_250_replays_database_SUMMED_OBS_WITH_TIMESTEP/'
    )"""
    replay_buffer = DataAugmentationReplayBuffer(
        obs_type=obs_type,
        use_channel_shuffle=True,
        s_shape=obs_type.get_obs_spec()[-3:],
        max_len=8e5,
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

    experiment_name = 'awac_' + format_experiment_name(obs_type,
                                                       reward_type,
                                                       action_masking,
                                                       channel_dims,
                                                       model_kwargs['conv_block_kwargs']) + '_v1'
    awac_alg = AWAC(model, optimizer, ge.VectorizedEnv(**env_kwargs), replay_buffer,
                    validation_kwargs_dicts=validation_kwargs_dicts,
                    deterministic_validation_policy=True,
                    device=DEVICE,
                    q_val_clamp=reward_type.get_cumulative_reward_spec(),
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
