import sys
from pathlib import Path

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from .seq2point.model import Seq2Point
from .seq2point.dataloader import ParquetCategoryDataset
from .seq2point.train import train, validation
from .handler import TrainingHandler, device_statistics, get_str_now, make_training_data_split


def main(config):

    train_data, Val_data, test_data = make_training_data_split(config)

    training_handler = TrainingHandler(config)
    if config['logging']:
        training_handler.logging_init(config)
        training_handler.logging_msg(f'Script called with: {" ".join(sys.argv)}')
        training_handler.logging_msg(f'Script started with configuration: {config}',)
        training_handler.logging_msg(f'Training start from {get_str_now(to_sec=True)}')
        training_handler.logging_msg(f'\nTraining data count: {train_data.shape[0]}\nValidation data count: {Val_data.shape[0]}\nTest data count: {test_data.shape[0]}\n')

    stats = device_statistics(config) # datafrome, df.loc[int channel 번호로 호출]
    channel_device = config['channel_device']
    for epoch in range(0, config['epochs']):
        # first epoch in all device first
        for channel, device_name in channel_device.items():
            channel_int = int(channel)
            channel_str = channel
            channel = 'ch'+channel
            _device_stat = stats.loc[[1, channel_int]]

            if config['logging']:
                training_handler.logging_msg(f"Training {device_name} Epoch {epoch} Start {get_str_now(to_sec=True)}")

            config['device_mean'] = stats.loc[channel_int]['active_interval_mean']
            config['device_std'] = stats.loc[channel_int]['active_interval_std']

            _df_train = train_data[train_data['channel'] == channel]
            _df_val = Val_data[Val_data['channel'] ==  channel]

            train_dataset = ParquetCategoryDataset(
                num_cached_parquet=config['num_cached_parquet'], 
                channel=channel_int,
                _df=_df_train,
                _stat=_device_stat,
                config=config,
                phase='train'
                )
            val_dataset = ParquetCategoryDataset(
                num_cached_parquet=config['num_cached_parquet'], 
                channel=channel_int,
                _df=_df_val,
                _stat=_device_stat,
                config=config,
                phase='val'
                )
            train_loader = DataLoader(train_dataset, shuffle=False, 
                                    batch_size=config['batch_size'], 
                                    num_workers=config['num_workers'])
            val_loader = DataLoader(val_dataset, shuffle=False,
                                    batch_size=config['batch_size'], 
                                    num_workers=config['num_workers'])

            if config['parallel']:
                model = Seq2Point(config['sequence_length'], True)
                model = nn.DataParallel(model)
            else :
                model = Seq2Point(config['sequence_length'], True).to(config['device'])

            training_handler.load_state(channel_str, epoch)
            if training_handler.get_state_value('last_ckp'):
                # try to load previous ckp model
                CKP_PATH = training_handler.get_state_value('last_ckp')
                if Path(CKP_PATH).exists():
                    print(f'try load ckp {CKP_PATH}')
                    model.load_state_dict(torch.load(CKP_PATH))

            criterion = nn.MSELoss()
            mae_criterion = nn.L1Loss()
            optimizer = optim.Adam(model.parameters(), lr=config['lr'])
            
            config['current_device_name'] = device_name
            config['current_epoch'] = epoch

            training_handler.update_state_value('epoch', epoch)

            mse, mae = train(model, train_loader, criterion, mae_criterion, optimizer, config, epoch, training_handler)

            if config['logging']:
                training_handler.logging_msg(f"Training {device_name} Epoch {epoch} End {get_str_now(to_sec=True)}\n")

                training_handler.logging_msg(f"Validation {device_name} Epoch {epoch} Start {get_str_now(to_sec=True)}")
            mae_loss = validation(model, val_loader, mae_criterion, config, epoch, training_handler)

            if config['logging']:
                training_handler.logging_msg(f"VAlidation {device_name} Epoch {epoch} End {get_str_now(to_sec=True)}\n")

