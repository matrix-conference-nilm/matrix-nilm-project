import json
import random
from pathlib import Path
import pandas as pd
import numpy as np
import fastparquet

from torch.utils.data import Dataset


class InsNilm(Dataset):
    
    def __init__(self, appliances, fold, config, _df, stats):

        self.fold = fold
        self.appliances = appliances
        self.window_size = config['window_size']
        self.slide_size = int(config['window_size']/10)
        self.data_root = config['data_dir']
        if isinstance(_df, str):
            self.files = [_df]
            self.labels = [_df['label_path']]
        else :
            self.files = _df['f_path'].to_list()
            self.labels = _df['label_path'].to_list()

        self.stat = stats

        self.device_std = self.stat.loc[self.stat.index == self.appliances, 'active_interval_std'].values[0]
        self.device_mean = self.stat.loc[self.stat.index == self.appliances, 'active_interval_mean'].values[0]
        
        self.create_index_map()
        self.cached_data = None
        self.cached_file = None

    def create_index_map(self):
        # length_mapper 에서 매핑한 길이정보로 index_map

        self.index_map = []
        for file, labels in zip(self.files, self.labels):
            # Fixed, number of datapoints
            file_length = 2592000
            num_windows_in_file = (file_length - self.window_size) // self.slide_size + 1
            file_indices = [i * self.slide_size for i in range(num_windows_in_file)]
            self.index_map.append((file, file_indices, labels))

    def reshuffle(self):
        for _, file_indices, _ in self.index_map:
            random.shuffle(file_indices)

    def min_max_scaling(self, data_array, max, min):
        return (data_array - min) / (max - min)

    def normalization(self, data_array, mean, std):
        return (data_array - mean) / std

    def load_data_from_files(self, file_name, label_name):
        
        if file_name != self.cached_file:
            list_df = []
            file_path = Path(self.data_root) / f'{Path(file_name).stem}.parquet'

            try :
                fdf = fastparquet.ParquetFile(str(file_path))
                for _df in fdf.iter_row_groups(columns=['time_stamps','active_power']):
                    list_df.append(_df)

                df_data = pd.concat(list_df)
                df_data = df_data.reset_index(drop=True)
                df_data['active_power'] = self.normalization(df_data['active_power'], self.device_mean, self.device_std)
            except Exception as e:
                print(str(file_path), e)

            active_power = df_data['active_power'].to_numpy()
            if Path(label_name).exists():
                with open(label_name, 'r') as j:
                    _label = json.load(j)['labels']

            # Initialize the label column to 0
            df_data['label'] = 0
            for start, end in _label:
                start_dt = pd.to_datetime(start)
                end_dt = pd.to_datetime(end)
                df_data.loc[df_data['time_stamps'].between(start_dt, end_dt), 'label'] = 1

            active_power = df_data['active_power'].to_numpy()
            labels_broadcasting = df_data['label'].to_numpy()
            
            self.cached_file = file_name
            self.cached_data = (active_power, labels_broadcasting)
        
        else :
            active_power, labels_broadcasting = self.cached_data

        return active_power, labels_broadcasting
    
    def __len__(self):
        return sum(len(file_indices) for _, file_indices, _ in self.index_map)
    
    def __getitem__(self, idx):
        for file, file_indices, labels in self.index_map:

            if idx < len(file_indices):
                start_idx = file_indices[idx]
                active_power, labels_broadcasting = self.load_data_from_files(file, labels)

                if start_idx + self.window_size > len(active_power):
                    # raise IndexError("Window exceeds available data.")
                    assert start_idx + self.window_size > len(active_power), f"Window exceeds available data. {self.window_size} :: {len(active_power)} :: {file}"
                
                window_active_power = active_power[start_idx:start_idx+self.window_size]
                window_labels_broadcasting = labels_broadcasting[start_idx:start_idx+self.window_size]

                # return torch.tensor(window_active_power), torch.tensor(window_labels_broadcasting)
                return window_active_power, window_labels_broadcasting
            idx -= len(file_indices)


