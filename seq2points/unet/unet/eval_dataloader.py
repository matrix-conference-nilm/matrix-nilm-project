import json
import random
from pathlib import Path
import pandas as pd
import fastparquet

import torch
from torch.utils.data import Dataset


class InsNilm(Dataset):
    
    def __init__(self, appliances, fold, config, _df, stats):

        self.fold = fold
        self.appliances = appliances
        self.window_size = config['window_size']
        self.slide_size = int(config['window_size']/10)
        self.data_root = config['data_dir']
        self.json_name_mapper = config['json_name_mapper']

        if isinstance(_df, str):
            self.files = [_df]
        else :
            self.files = _df['f_path'].to_list()

        self.stat = stats
        self.device_std = self.stat.loc[self.stat.index == self.appliances, 'active_interval_std'].values[0]
        self.device_mean = self.stat.loc[self.stat.index == self.appliances, 'active_interval_mean'].values[0]
        
        self.create_index_map()
        self.cached_data = None
        self.cached_file = None

    def create_index_map(self):
        # length_mapper 에서 매핑한 길이정보로 index_map

        self.index_map = []
        for file in self.files:
            # TODO
            if self.fold == 'test':
                file_length = 2592000
            else :
                file_length = 2592000
            num_windows_in_file = (file_length - self.window_size) // self.slide_size + 1
            file_indices = [i * self.slide_size for i in range(num_windows_in_file)]
            self.index_map.append((file, file_indices))

    def reshuffle(self):
        for _, file_indices in self.index_map:
            random.shuffle(file_indices)

    def min_max_scaling(self, data_array, max, min):
        return (data_array - min) / (max - min)

    def normalization(self, data_array, mean, std):
        return (data_array - mean) / std

    def load_data_from_files(self, file_name):
        
        if file_name != self.cached_file:
            list_df = []
            
            # TODO
            # remove
            if self.fold == 'test':
                file_path = Path(self.data_root) / f'{file_name.replace("_donwsampling","")}.parquet'
            else :
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
            # add up label 
            labeling_json_path = self.json_name_mapper.get(Path(file_name).stem, None)
            if labeling_json_path:
                with open(labeling_json_path, 'r') as j:
                    labels = json.load(j)['labels']
            else :
                labels = []

            # Initialize the label column to 0
            df_data['label'] = 0
            for start, end in labels:
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
        return sum(len(file_indices) for _, file_indices in self.index_map)
    
    def __getitem__(self, idx):
        for file, file_indices in self.index_map:

            if idx < len(file_indices):
                start_idx = file_indices[idx]
                active_power, labels_broadcasting = self.load_data_from_files(file)

                if start_idx + self.window_size > len(active_power):
                    # raise IndexError("Window exceeds available data.")
                    assert start_idx + self.window_size > len(active_power), f"Window exceeds available data. {self.window_size} :: {len(active_power)} :: {file}"
                
                window_active_power = active_power[start_idx:start_idx+self.window_size]
                window_labels_broadcasting = labels_broadcasting[start_idx:start_idx+self.window_size]

                return window_active_power, window_labels_broadcasting
            idx -= len(file_indices)

