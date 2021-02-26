import copy
from pathlib import Path
import shutil
from torch import nn

from rl.deep_q import DeepQ
from rl import models
from rl.replay_buffers import *
from rl.utils import format_experiment_name, EpsilonScheduler

if __name__ == '__main__':
    DEVICE = torch.device('cuda')

    obs_type = ge.ObsType.HEAD_CENTERED_OBS_SMALL
    reward_type = ge.RewardType.ON_EAT_AND_ON_DEATH
    action_masking = ge.ActionMasking.FULL
    channel_dims = [32, 64, 128]
    first_downsample = nn.AvgPool2d(2, stride=1)
    model_kwargs = dict(conv_block_kwargs=[
        dict(
            in_channels=obs_type.get_obs_spec()[2],
            out_channels=channel_dims[0],
            kernel_size=3,
            downsample=first_downsample
        ),
        dict(
            in_channels=channel_dims[0],
            out_channels=channel_dims[1],
            kernel_size=3,
            downsample=nn.AvgPool2d(2),
        ),
        dict(
            in_channels=channel_dims[1],
            out_channels=channel_dims[2],
            kernel_size=3,
            downsample=nn.AvgPool2d(2, stride=1),
        )
    ],
        use_adaptive_avg_pool=False,
        fc_in_channels=channel_dims[2] * 2 * 4,
        # fc_in_channels=channel_dims[1],
        **reward_type.get_recommended_value_activation_scale_shift_dict(),
        epsilon=None,
        tau=5e-3,
        softmax_exploration=True,
        delayed_updates=True,
        double_q=True,
        dueling_q=True,
        optimizer_constructor=lambda params: torch.optim.Adam(params,
                                                              # weight_decay=1e-5,
                                                              ),
    )
    model = models.DeepQNetwork(**model_kwargs)
    """
    run_dir = 'runs/deep_q/deep_q_head_centered_obs_small_every_step_length_opposite_6_blocks_32_64_128_dims_v1/'
    with open(f'{run_dir}/0620/cp.txt', 'r') as f:
        serialized_string = f.readline()[2:-1].encode()
    state_dict_bytes = base64.b64decode(serialized_string)
    loaded_state_dicts = pickle.loads(state_dict_bytes)
    model.load_state_dict(loaded_state_dicts)
    """
    model.to(device=DEVICE)

    env_kwargs = dict(
        n_envs=16,
        obs_type=obs_type,
        reward_type=reward_type,
        action_masking=action_masking
    )
    rl_train_kwargs = dict(
        batch_size=1024,
        n_steps_per_epoch=int(1e3),
        train_frequency=4,
        n_train_batches=1,
        gamma=0.99,
        epsilon_scheduler=EpsilonScheduler(
            start_vals=torch.tensor([1., 1., 1., 1.]),
            #min_vals=torch.tensor([0.025, 0.05, 0.1, 0.2]),
            min_vals=0.15,
            train_steps_to_reach_min_vals=5e5,
            #start_vals=1.,
            #min_vals=0.02,
        )
    )

    replay_buffer = DataAugmentationReplayBuffer(
        obs_type=obs_type,
        use_channel_shuffle=True,
        s_shape=obs_type.get_obs_spec()[-3:],
        max_len=8e5,
        starting_s_a_r_d_s=None,
    )

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

    experiment_name = 'deep_q_' + format_experiment_name(obs_type,
                                                         reward_type,
                                                         action_masking,
                                                         channel_dims,
                                                         model_kwargs['conv_block_kwargs']) + '_v1'
    deep_q_alg = DeepQ(model, ge.VectorizedEnv(**env_kwargs), replay_buffer,
                       validation_kwargs_dicts=validation_kwargs_dicts,
                       device=DEVICE,
                       exp_folder=Path(f'runs/deep_q/{experiment_name}'),
                       use_action_masking=True,
                       clip_grads=10.,
                       checkpoint_freq=10,
                       checkpoint_render_n_games=10)
    this_script = Path(__file__).absolute()
    shutil.copy(this_script, deep_q_alg.exp_folder / f'_{this_script.name}')

    try:
        deep_q_alg.train(
            n_epochs=int(1e6),
            **rl_train_kwargs
        )
    except KeyboardInterrupt:
        if deep_q_alg.epoch_counter > deep_q_alg.checkpoint_freq:
            print('KeyboardInterrupt: saving model')
            deep_q_alg.save(deep_q_alg.exp_folder, finished=True)
