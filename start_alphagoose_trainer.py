from pathlib import Path
import torch
from torch import nn
from torchvision import transforms

from hungry_geese.training.alphagoose.alphagoose_trainer import AlphaGooseTrainer
from hungry_geese.training.alphagoose.alphagoose_data import AlphaGooseRandomReflect, ToTensor
from hungry_geese.env import goose_env as ge
from hungry_geese import models
from hungry_geese.utils import format_experiment_name

if __name__ == '__main__':
    DEVICE = torch.device('cuda')

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
    )
    model = models.FullConvActorCriticNetwork(**model_kwargs)
    model.to(device=DEVICE)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.02,
        momentum=0.9,
        weight_decay=1e-4
    )
    # NB: lr_scheduler counts steps in batches, not epochs
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        # Stop reducing LR beyond 2e-4
        milestones=[200000 * i for i in range(2, 4)],
        gamma=0.1
    )
    dataset_kwargs = dict(
        root='/home/isaiah/data/alphagoose_data/',
        obs_type=obs_type,
        transform=transforms.Compose([
            AlphaGooseRandomReflect(obs_type),
            ToTensor()
        ]),
    )
    dataloader_kwargs = dict(
        batch_size=512,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    experiment_name = 'alphagoose_' + format_experiment_name(obs_type,
                                                             ge.RewardType.RANK_ON_DEATH,
                                                             ge.ActionMasking.LETHAL,
                                                             [n_channels],
                                                             model_kwargs['conv_block_kwargs']) + '_v1'
    exp_folder = Path(f'runs/alphagoose/{experiment_name}')
    train_alg = AlphaGooseTrainer(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        dataset_kwargs=dataset_kwargs,
        dataloader_kwargs=dataloader_kwargs,
        device=DEVICE,
        use_mixed_precision=False,
        exp_folder=exp_folder,
        min_saved_steps=1000,
        checkpoint_freq=5,
        checkpoint_render_n_games=5,
    )

    try:
        train_alg.train(n_epochs=int(1e7))
    except KeyboardInterrupt:
        if train_alg.epoch_counter > train_alg.checkpoint_freq:
            print('KeyboardInterrupt: saving model')
            train_alg.save(train_alg.exp_folder, finished=True)
