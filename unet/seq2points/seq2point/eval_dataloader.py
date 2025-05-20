from pathlib import Path
import pandas as pd
import numpy as np
import fastparquet
from numpy.lib.stride_tricks import sliding_window_view

import torch
from torch.utils.data import Dataset


class EvalDataset(Dataset):  
    def __init__(self, num_cached_parquet, channel, device_path, main_path, _stat, config):

        self.raw_cols = 'active_power'  # parquet file에서 사용할 column
        self.channel = channel # int channel
        self.sequence_length = config['sequence_length']
        self.units_to_pad = self.sequence_length // 2
        self.data_root = config['data_dir']
        self.stat_df = _stat 
        self.main_mean, self.main_std = _stat['mean_val'][1], _stat['std_dev'][1]
        self.device_mean, self.device_std = _stat['mean_val'][self.channel], _stat['std_dev'][self.channel]

        if isinstance(device_path, list):
            self.parquet_list = [self.data_root+Path(x).stem+'.parquet' for x in device_path]
            self.parquet_main = [self.data_root+Path(x).stem+'.parquet' for x in main_path]
        elif isinstance(device_path, str):
            self.parquet_list = [self.data_root+Path(str(device_path)).stem+'.parquet']
            self.parquet_main = [self.data_root+Path(str(main_path)).stem+'.parquet']

            self.fparquet = fastparquet.ParquetFile(self.parquet_list[0])
            self.fparquet_main = fastparquet.ParquetFile(self.parquet_main[0])

        self._cache_setting()
        self.total_len = self.get_total_length()

    def _cache_setting(self):
        self.new_main, self.device_active_power = self._cache_parquet()

    def get_total_length(self):
        fdf = fastparquet.ParquetFile(self.parquet_list)
        total_len = 0
        for _df in fdf.iter_row_groups(columns=['active_power']):
            total_len += len(_df)
        return total_len -1

    def _cache_parquet(self):
        list_df = []
        for _df in zip( self.fparquet.iter_row_groups(columns=[self.raw_cols]),\
                        self.fparquet_main.iter_row_groups(columns=[self.raw_cols]) ):
            _device = _df[0]
            _device['main_power'] = _df[1]['active_power']
            
            list_df.append(_device)
        df_data = pd.concat(list_df)

        # normalization
        df_data['active_power'] = self.normalization(df_data['active_power'], self.device_mean, self.device_std)
        df_data['main_power'] = self.normalization(df_data['main_power'], self.main_mean, self.main_std)
        active_power = df_data['active_power'].values
        # Padding
        new_main = np.pad(df_data['main_power'].values, (self.units_to_pad, self.units_to_pad), 'constant', constant_values=(0, 0))
        # Efficient sequence creation using sliding_window_view
        new_main = sliding_window_view(new_main, window_shape=self.sequence_length)
        new_main = new_main.reshape(-1, new_main.shape[-1])
        new_main = new_main[0:-1, :]
        new_main = new_main.reshape(-1, 1, self.sequence_length)

        return new_main, active_power

    def normalization(self, data_array, mean, std):
        return (data_array - mean) / std

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        
        main_active_power = self.new_main[idx]
        device_active_power = self.device_active_power[idx]

        # Make a copy of the numpy arrays to ensure they are writable
        main_active_power = np.copy(main_active_power)
        device_active_power = np.copy(device_active_power)

        # Convert to PyTorch tensors
        main_active_power = torch.from_numpy(main_active_power)
        device_active_power = torch.from_numpy(device_active_power)

        return main_active_power, device_active_power