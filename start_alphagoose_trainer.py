from pathlib import Path
import torch
from torch import nn
from torchvision import transforms

from hungry_geese.training.alphagoose.alphagoose_trainer import AlphaGooseTrainer
from hungry_geese.training.alphagoose.alphagoose_data import AlphaGooseRandomReflect, ToTensor
from hungry_geese.env import goose_env as ge
from hungry_geese.nns import models, conv_blocks
from hungry_geese.utils import format_experiment_name

if __name__ == '__main__':
    DEVICE = torch.device('cuda:1')

    obs_type = ge.ObsType.COMBINED_GRADIENT_OBS_SMALL
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
    model.to(device=DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(),
        #lr=0.05,
        #momentum=0.9,
        #weight_decay=1e-4
    )
    batch_size = 2048
    # NB: lr_scheduler counts steps in batches, not epochs
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        #milestones=[int(100000 * 512 * i / batch_size) for i in [3]],
        milestones=[],
        gamma=0.1
    )
    dataset_kwargs = dict(
        dataset_dir='/home/isaiah/data/alphagoose_data',
        obs_type=obs_type,
        transform=transforms.Compose([
            AlphaGooseRandomReflect(obs_type),
            ToTensor()
        ]),
    )
    dataloader_kwargs = dict(
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=False
    )

    experiment_name = 'alphagoose_' + format_experiment_name(obs_type,
                                                             ge.RewardType.RANK_ON_DEATH,
                                                             ge.ActionMasking.OPPOSITE,
                                                             [n_channels],
                                                             model_kwargs['conv_block_kwargs']) + '_v1'
    exp_folder = Path(f'runs/alphagoose/active/{experiment_name}')
    train_alg = AlphaGooseTrainer(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        dataset_kwargs=dataset_kwargs,
        dataloader_kwargs=dataloader_kwargs,
        device=DEVICE,
        use_mixed_precision=True,
        exp_folder=exp_folder,
        checkpoint_freq=1,
        checkpoint_render_n_games=0,

        # min_saved_steps=10,
        # min_saved_steps=int(5e5),
        start_from_scratch=False,
    )

    try:
        train_alg.train(n_epochs=int(1e7))
    except KeyboardInterrupt:
        if train_alg.epoch_counter > train_alg.checkpoint_freq:
            print('KeyboardInterrupt: saving model')
            train_alg.save(train_alg.exp_folder, finished=True)
