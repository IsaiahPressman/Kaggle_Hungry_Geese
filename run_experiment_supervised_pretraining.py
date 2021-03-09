import numpy as np
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from hungry_geese import models
import hungry_geese.env.goose_env as ge
from hungry_geese.training.alphagoose.alphagoose_data import AlphaGoosePretrainDataset, PretrainRandomReflect, ToTensor
from hungry_geese.training.alphagoose.supervised_pretraining import SupervisedPretraining
from hungry_geese.utils import format_experiment_name

if __name__ == '__main__':
    DEVICE = torch.device('cuda')

    obs_type = ge.ObsType.COMBINED_GRADIENT_OBS
    n_channels = 128
    activation = nn.ReLU
    normalize = True
    model_kwargs = dict(
        block_class=models.BasicConvolutionalBlock,
        conv_block_kwargs=[
            dict(
                in_channels=obs_type.get_obs_spec()[-3],
                out_channels=n_channels,
                kernel_size=3,
                activation=activation,
                normalize=normalize
            ),
            dict(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=3,
                activation=activation,
                normalize=normalize
            ),
            dict(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=3,
                activation=activation,
                normalize=normalize
            ),
        ],
        squeeze_excitation=True,
        cross_normalize_value=True,
        # **ge.RewardType.RANK_ON_DEATH.get_recommended_value_activation_scale_shift_dict()
    )
    model = models.FullConvActorCriticNetwork(**model_kwargs)
    """
    run_dir = 'runs/deep_q/deep_q_head_centered_obs_small_every_step_length_opposite_6_blocks_32_64_128_dims_v1/'
    with open(f'{run_dir}/0620/cp.txt', 'r') as f:
        serialized_string = f.readline()[2:-1].encode()
    state_dict_bytes = base64.b64decode(serialized_string)
    loaded_state_dicts = pickle.loads(state_dict_bytes)
    model.load_state_dict(loaded_state_dicts)
    """
    model.to(device=DEVICE)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=1e-4
    )
    # NB: lr_scheduler counts steps in batches, not epochs
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        # Stop reducing LR beyond 1e-5
        milestones=[150000*i for i in range(2, 5)],
        gamma=0.1
    )

    dataset_loc = Path('/home/isaiah/data/alphagoose_pretrain_data_1000/')
    with open(dataset_loc / 'all_saved_episodes.txt', 'r') as f:
        all_episodes = [replay_name.rstrip() for replay_name in f.readlines()]
    train_episodes, test_episodes = train_test_split(np.array(all_episodes), test_size=0.05)
    train_episodes = set(train_episodes)
    test_episodes = set(test_episodes)
    train_dataset = AlphaGoosePretrainDataset(
        dataset_loc,
        ge.ObsType.COMBINED_GRADIENT_OBS,
        transform=transforms.Compose([
            PretrainRandomReflect(obs_type),
            ToTensor()
        ]),
        include_episode=lambda x: x.stem in train_episodes
    )
    test_dataset = AlphaGoosePretrainDataset(
        dataset_loc,
        ge.ObsType.COMBINED_GRADIENT_OBS,
        transform=ToTensor(),
        include_episode=lambda x: x.stem in test_episodes
    )
    print(f'Split {len(train_episodes) + len(test_episodes)} episodes into '
          f'{len(train_dataset)} samples from {len(train_episodes)} train episodes and '
          f'{len(test_dataset)} samples from {len(test_episodes)} test episodes.')
    dataloader_kwargs = dict(
        batch_size=512,
        shuffle=True,
        pin_memory=True
    )
    train_dataloader = DataLoader(train_dataset, num_workers=6, **dataloader_kwargs)
    test_dataloader = DataLoader(test_dataset, num_workers=10, **dataloader_kwargs)

    experiment_name = 'supervised_pretraining_' + format_experiment_name(obs_type,
                                                                         ge.RewardType.RANK_ON_DEATH,
                                                                         ge.ActionMasking.NONE,
                                                                         [n_channels],
                                                                         model_kwargs['conv_block_kwargs']) + '_v5'
    exp_folder = Path(f'runs/supervised_pretraining/active/{experiment_name}')
    train_alg = SupervisedPretraining(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        policy_weight=1.,
        value_weight=1.,
        entropy_weight=0.05,
        device=DEVICE,
        use_mixed_precision=True,
        exp_folder=exp_folder,
        checkpoint_freq=10,
        checkpoint_render_n_games=2
    )
    this_script = Path(__file__).absolute()
    shutil.copy(this_script, train_alg.exp_folder / f'_{this_script.name}')
    with open(train_alg.exp_folder / 'train_episodes.txt', 'w') as f:
        f.writelines([f'{rn}\n' for rn in sorted(list(train_episodes), key=lambda x: int(x))])
    with open(train_alg.exp_folder / 'test_episodes.txt', 'w') as f:
        f.writelines([f'{rn}\n' for rn in sorted(list(test_episodes), key=lambda x: int(x))])

    try:
        train_alg.train(n_epochs=int(1e6))
    except KeyboardInterrupt:
        if train_alg.epoch_counter > train_alg.checkpoint_freq:
            print('KeyboardInterrupt: saving model')
            train_alg.save(train_alg.exp_folder, finished=True)
