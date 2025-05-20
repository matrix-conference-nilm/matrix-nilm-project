import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .unet.dataloader import InsNilm
from .unet.model import UNetNiLM
from .unet.train import train, valiation
from .handler import TrainingHandler, device_statistics, make_training_data_split, get_str_now


def main(config):

    train_data, Val_data, test_data = make_training_data_split(config)
    training_handler = TrainingHandler(config)

    if config['logging']:
        training_handler.logging_init(config)
        training_handler.logging_msg(f'Script called with: {" ".join(sys.argv)}')
        training_handler.logging_msg(f'Script started with configuration: {config}',)
        training_handler.logging_msg(f'Training start from {get_str_now(to_sec=True)}')
        training_handler.logging_msg(f'\nTraining data count: {train_data.shape[0]}\nValidation data count: {Val_data.shape[0]}\nTest data count: {test_data.shape[0]}\n')

    stats = device_statistics(config)
    channel_device = config['channel_device']
    for epoch in range(0, config['epoch']):
        
        for channel, device_name in channel_device.items():
            # channel = 02,03 ...
            channel_int = int(channel)
            channel_str = channel
            channel = 'ch'+channel
            _df_train = train_data[train_data['channel'] == channel]
            _df_val = Val_data[Val_data['channel'] == channel]

            if config['debug']:
                print("=== RUNNING ON DEBUG MODE ===")
                _df_train = _df_train.sample(n=10)
                _df_val = _df_val.sample(n=3)

            train_dataset = InsNilm(channel_int, 'train', config, _df_train, stats)
            val_dataset = InsNilm(channel_int, 'val', config, _df_val, stats)

            train_loader = DataLoader(train_dataset, shuffle=False, batch_size=config['batch_size'], num_workers=config['num_workers'])
            val_loader = DataLoader(val_dataset, shuffle=False, batch_size=config['batch_size'])

            model = UNetNiLM(
                num_layers=config['num_layers'],
                features_start=8,
                n_channels=config['n_channels'],
                num_classes=config['num_classes'],
                pooling_size=config['pooling_size'],
                window_size=config['window_size'],
                num_quantiles=len(config['taus']),
                dropout=0.1,
                d_model=128,
            )

            model = model.to(config['device'])
            if config['parallel']:
                model = nn.DataParallel(model)
            else :
                model = model.to(config['device'])

            training_handler.load_state(channel_str, epoch)
            if training_handler.get_state_value('last_ckp') is not None:
                CKP_PATH = training_handler.get_state_value('last_ckp')
                if Path(CKP_PATH).exists():
                    print(f'TRY LOAD CKP ... {CKP_PATH}')
                    model.load_state_dict(torch.load(CKP_PATH))

            optim = torch.optim.Adam(model.parameters(), **config['optim_kwargs'])
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, **config['scheduler_kwargs'])
            
            training_handler.update_state_value('epoch', epoch)
            config['current_device_name'] = device_name
            config['current_epoch'] = epoch

            train_loss, train_f1 = train(model, train_loader, optim, config, epoch, training_handler)
            if config['logging']:
                training_handler.logging_msg(f"Training {device_name} Epoch {epoch} End {get_str_now(to_sec=True)}\n")
                training_handler.logging_msg(f"Validation {device_name} Epoch {epoch} Start {get_str_now(to_sec=True)}")

            val_loss, val_f1 = valiation(model, val_loader, config, epoch, training_handler)
            if config['logging']:
                training_handler.logging_msg(f"VAlidation {device_name} Epoch {epoch} End {get_str_now(to_sec=True)}\n")

            if config['wandb']:
                channel = training_handler.get_state_value('channel')
                training_handler.wandb_logging(
                        {
                            f"{channel}_{config['current_device_name']}/train_lass" : train_loss,
                            f"{channel}_{config['current_device_name']}/train_f1" : train_f1,
                            f"{channel}_{config['current_device_name']}/val_lass" : val_loss,
                            f"{channel}_{config['current_device_name']}/val_f1" : val_f1,
                        }
                    )
            scheduler.step(val_loss)
            
    return

