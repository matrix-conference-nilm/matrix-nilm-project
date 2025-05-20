from pathlib import Path
import numpy as np
import pandas as pd
import fastparquet

from torch.utils.data import Dataset


class ParquetCategoryDataset(Dataset):  
    def __init__(self, num_cached_parquet, channel, _df, _stat, config, phase):

        self.raw_cols = 'active_power'  # parquet file에서 사용할 column
        self.num_cached_parquet = num_cached_parquet  # 캐시할 파일 개수
        self.channel = channel # int channel
        self.sequence_length = config['sequence_length']
        self.units_to_pad = self.sequence_length // 2
        self.phase = phase
        self.shuffle = config['shuffle']
        self.num_workers = config['num_workers']
        # self.refresh_freq = config['refresh_freq']
        self.data_root = config['data_dir']
        self.stat_df = _stat 
        self.main_mean, self.main_std = _stat['mean_val'][1], _stat['std_dev'][1]
        self.device_mean, self.device_std = _stat['active_interval_mean'][self.channel], _stat['active_interval_std'][self.channel]

        _df.loc[:,'f_path'] = _df['f_path'].apply(lambda x: self.data_root+Path(x).stem+'.parquet')
        _df.loc[:,'main_path'] = _df['main_path'].apply(lambda x: self.data_root+Path(x).stem+'.parquet')

        self._df = _df.loc[:,['f_path', 'main_path']].values

        self.debug = config['debug']
        if self.debug:
            self.n_limit_data = 0.1
        else :
            self.n_limit_data = 1

        self.reshuffle_file_list()

        # 캐시 기반 데이터로더 설계
        self.steps_cache = int(np.ceil(
            len(self.parquet_list) / self.num_cached_parquet))  # cache step
        self.current_parquet_idx = 0
        self.current_pd = None  # cached parquets
        # cache할 segment 길이
        self.segment_length = int(config['batch_size'] * 100)
        self.current_segment = None # cached sequence
        self.current_segment_label = None
        self.current_segment_idx = []
        self.current_segment_start_idx = 0

        self.current_pd_indices_in_cache = []  # data index in cached parquet
        self.steps_per_epoch = 0
        self.total_len = self.get_total_length()

        self._cache_setting()

    def reshuffle_file_list(self):
        limit_idx = int(self._df.shape[0] * self.n_limit_data)
        self._df = np.random.permutation(self._df)

        self.parquet_list = list(self._df[:limit_idx,0])
        self.parquet_main = list(self._df[:limit_idx,1])

    def _cache_setting(self):
        cur_pd, cur_indices = self._cache_parquet(self.current_parquet_idx)
        self.current_pd = cur_pd 
        self.current_pd_indices_in_cache = cur_indices

    def get_total_length(self):
        if self.debug:
            length_of_day = 30*60*60*24 # 2592000 1일 데이터 row
            return int(length_of_day * len(self.parquet_list))
        # else :
        #     fdf = fastparquet.ParquetFile(self.parquet_list)
        #     total_len = 0
        #     for _df in fdf.iter_row_groups(columns=['active_power']):
        #         total_len += len(_df)
        #     return total_len

    def _cache_parquet(self, idx):
        # current_parquet_idx -> 다음 버킷 만큼 데이터를 불러오기
        next_idx = (idx+1)*self.num_cached_parquet
        next_idx = None if next_idx > len(self.parquet_list) else next_idx

        list_part_parquet = self.parquet_list[
            idx*self.num_cached_parquet:next_idx
            ]
        list_part_parquet_main = self.parquet_main[
            idx*self.num_cached_parquet:next_idx
        ]

        list_parquet_path = [[x[0], x[1]] for x in zip(list_part_parquet, list_part_parquet_main)]
        list_parquet_path = np.array(list_parquet_path)
        # 일단위 parquet 로드
        fparquet = fastparquet.ParquetFile(list(list_parquet_path[:,0]))
        fparquet_main = fastparquet.ParquetFile(list(list_parquet_path[:,1]))

        list_df = []
        for _df in zip( fparquet.iter_row_groups(columns=[self.raw_cols]),\
                        fparquet_main.iter_row_groups(columns=[self.raw_cols]) ):
            _device = _df[0]
            _device['main_power'] = _df[1]['active_power']
            list_df.append(_device)
        df_data = pd.concat(list_df)
        
        # normalization
        df_data = df_data.reset_index(drop=True)
        df_data['active_power'] = self.normalization(df_data['active_power'], self.device_mean, self.device_std)
        df_data['main_power'] = self.normalization(df_data['main_power'], self.main_mean, self.main_std)
        np_indices = np.arange(len(df_data))
        list_indices = np_indices.tolist()

        return df_data, list_indices

    def normalization(self, data_array, mean, std):
        return (data_array - mean) / std

    def transform_seq_window(self, array):

        main_active_power =np.pad(array, (self.units_to_pad, self.units_to_pad), 'constant', constant_values=(0,0))
        new_main = [main_active_power[i:i+self.sequence_length] for i in range(len(main_active_power) - self.sequence_length)]
        new_main = np.array(new_main).reshape(-1, 1, self.sequence_length)

        return new_main

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        
        refresh_idx = 1
        if len(self.current_pd_indices_in_cache) <= refresh_idx:
            self.current_parquet_idx += 1
            if self.current_parquet_idx >= self.steps_cache:
                self.current_parquet_idx = 0

            self._cache_setting()
            self.current_segment_start_idx = 0

        # Check if current sequence cache is empty or needs to be updated or is_refreshed
        if len(self.current_segment_idx) < 1 or self.current_segment_start_idx == 0:
            start_idx = self.current_segment_start_idx
            end_idx = start_idx + self.segment_length
            self.current_segment = self.transform_seq_window(self.current_pd.iloc[start_idx:end_idx]['main_power'].to_numpy())
            self.current_segment_label = self.current_pd.iloc[start_idx:end_idx]['active_power'].to_numpy().reshape(-1, 1)

            # Update indices cache after segment is processed
            self.current_segment_idx = list(range(0, len(self.current_segment)))[::-1]

        if self.shuffle:
            rng = np.random.RandomState(seed=42)
            self.current_segment_idx = list(rng.permutation(self.current_segment_idx))

        self.current_pd_indices_in_cache.pop()
        relative_idx = self.current_segment_idx.pop()

        main_active_power = self.current_segment[relative_idx]
        device_active_power = self.current_segment_label[relative_idx]

        self.current_segment_start_idx += 1

        return main_active_power, device_active_power
