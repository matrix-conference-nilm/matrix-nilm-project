from pathlib import Path
import numpy as np
import pandas as pd
import fastparquet

from torch.utils.data import Dataset


class ParquetCategoryDataset(Dataset):  
    """
    Parquet 파일로 저장된 시계열 데이터를 효율적으로 로드하는 커스텀 데이터셋
    - 대용량 데이터를 메모리에 한 번에 올리지 않고 캐싱하여 사용
    - 시계열 데이터를 시퀀스 윈도우로 변환하여 반환
    - 정규화 및 셔플 기능 지원
    """
    def __init__(self, num_cached_parquet, channel, _df, _stat, config, phase):
        """
        Args:
            num_cached_parquet: 한 번에 캐시할 parquet 파일 개수
            channel: 사용할 디바이스 채널 인덱스
            _df: parquet 파일 경로 정보가 담긴 DataFrame
            _stat: 정규화에 사용할 통계값이 담긴 DataFrame
            config: 설정 정보 (시퀀스 길이, 배치 크기 등)
            phase: 학습/검증/테스트 구분
        """
        # 기본 설정
        self.raw_cols = 'active_power'  # parquet 파일에서 사용할 column
        self.num_cached_parquet = num_cached_parquet  # 캐시할 파일 개수
        self.channel = channel # int channel
        self.sequence_length = config['sequence_length']  # 시퀀스 윈도우 길이
        self.units_to_pad = self.sequence_length // 2  # 시퀀스 앞뒤로 패딩할 길이
        self.phase = phase
        self.shuffle = config['shuffle']  # 셔플 여부
        self.num_workers = config['num_workers']  # DataLoader에서 사용할 워커 수
        self.data_root = config['data_dir']  # 데이터 파일 경로

        # 정규화에 사용할 통계값
        self.stat_df = _stat 
        self.main_mean, self.main_std = _stat['mean_val'][1], _stat['std_dev'][1]  # 메인 파워 통계
        self.device_mean, self.device_std = _stat['active_interval_mean'][self.channel], _stat['active_interval_std'][self.channel]  # 디바이스 파워 통계

        # parquet 파일 경로 설정
        _df.loc[:,'f_path'] = _df['f_path'].apply(lambda x: self.data_root+Path(x).stem+'.parquet')
        _df.loc[:,'main_path'] = _df['main_path'].apply(lambda x: self.data_root+Path(x).stem+'.parquet')
        self._df = _df.loc[:,['f_path', 'main_path']].values

        # 디버그 모드 설정
        self.debug = config['debug']
        if self.debug:
            self.n_limit_data = 0.1  # 디버그 모드일 때 데이터 10%만 사용
        else:
            self.n_limit_data = 1  # 전체 데이터 사용

        # 파일 리스트 셔플 및 초기화
        self.reshuffle_file_list()

        # 캐시 기반 데이터로더 설계
        self.steps_cache = int(np.ceil(
            len(self.parquet_list) / self.num_cached_parquet))  # cache step
        self.current_parquet_idx = 0  # 현재 캐시된 parquet 인덱스
        self.current_pd = None  # cached parquets
        self.segment_length = int(config['batch_size'] * 100)  # cache할 segment 길이
        self.current_segment = None  # cached sequence
        self.current_segment_label = None
        self.current_segment_idx = []
        self.current_segment_start_idx = 0

        self.current_pd_indices_in_cache = []  # data index in cached parquet
        self.steps_per_epoch = 0
        self.total_len = self.get_total_length()

        # 첫 번째 parquet 파일 캐싱
        self._cache_setting()

    def reshuffle_file_list(self):
        """
        데이터 파일 리스트를 셔플하고, 일부만 사용(디버그 모드 시)
        """
        limit_idx = int(self._df.shape[0] * self.n_limit_data)
        self._df = np.random.permutation(self._df)

        self.parquet_list = list(self._df[:limit_idx,0])  # 디바이스 파워 parquet 경로
        self.parquet_main = list(self._df[:limit_idx,1])  # 메인 파워 parquet 경로

    def _cache_setting(self):
        """
        현재 인덱스에 해당하는 parquet 파일들을 읽어와 DataFrame으로 캐싱
        """
        cur_pd, cur_indices = self._cache_parquet(self.current_parquet_idx)
        self.current_pd = cur_pd 
        self.current_pd_indices_in_cache = cur_indices

    def get_total_length(self):
        """
        전체 데이터 길이 반환
        - 디버그 모드일 때만 구현됨
        """
        if self.debug:
            length_of_day = 30*60*60*24  # 2592000 1일 데이터 row
            return int(length_of_day * len(self.parquet_list))
        # else:
        #     fdf = fastparquet.ParquetFile(self.parquet_list)
        #     total_len = 0
        #     for _df in fdf.iter_row_groups(columns=['active_power']):
        #         total_len += len(_df)
        #     return total_len

    def _cache_parquet(self, idx):
        """
        지정된 인덱스 범위의 parquet 파일을 읽어와 DataFrame으로 합침
        Args:
            idx: 캐시할 parquet 파일의 시작 인덱스
        Returns:
            df_data: 캐시된 DataFrame
            list_indices: DataFrame의 인덱스 리스트
        """
        # current_parquet_idx -> 다음 버킷 만큼 데이터를 불러오기
        next_idx = (idx+1)*self.num_cached_parquet
        next_idx = None if next_idx > len(self.parquet_list) else next_idx

        # 캐시할 parquet 파일 경로 리스트
        list_part_parquet = self.parquet_list[
            idx*self.num_cached_parquet:next_idx
            ]
        list_part_parquet_main = self.parquet_main[
            idx*self.num_cached_parquet:next_idx
        ]

        # 디바이스/메인 파워 parquet 경로 쌍으로 묶기
        list_parquet_path = [[x[0], x[1]] for x in zip(list_part_parquet, list_part_parquet_main)]
        list_parquet_path = np.array(list_parquet_path)
        
        # parquet 파일 로드
        fparquet = fastparquet.ParquetFile(list(list_parquet_path[:,0]))
        fparquet_main = fastparquet.ParquetFile(list(list_parquet_path[:,1]))

        # 디바이스/메인 파워 데이터 합치기
        list_df = []
        for _df in zip( fparquet.iter_row_groups(columns=[self.raw_cols]),\
                        fparquet_main.iter_row_groups(columns=[self.raw_cols]) ):
            _device = _df[0]
            _device['main_power'] = _df[1]['active_power']
            list_df.append(_device)
        df_data = pd.concat(list_df)
        
        # 정규화 수행
        df_data = df_data.reset_index(drop=True)
        df_data['active_power'] = self.normalization(df_data['active_power'], self.device_mean, self.device_std)
        df_data['main_power'] = self.normalization(df_data['main_power'], self.main_mean, self.main_std)
        
        # 인덱스 리스트 생성
        np_indices = np.arange(len(df_data))
        list_indices = np_indices.tolist()

        return df_data, list_indices

    def normalization(self, data_array, mean, std):
        """
        데이터 정규화: (값 - 평균) / 표준편차
        """
        return (data_array - mean) / std

    def transform_seq_window(self, array):
        """
        1차원 시계열 데이터를 시퀀스 윈도우(슬라이딩 윈도우)로 변환
        Args:
            array: 1차원 시계열 데이터
        Returns:
            new_main: 3차원 배열로 변환된 시퀀스 윈도우
        """
        # 앞뒤로 패딩
        main_active_power = np.pad(array, (self.units_to_pad, self.units_to_pad), 'constant', constant_values=(0,0))
        # 시퀀스 길이만큼 잘라서 윈도우 생성
        new_main = [main_active_power[i:i+self.sequence_length] for i in range(len(main_active_power) - self.sequence_length)]
        new_main = np.array(new_main).reshape(-1, 1, self.sequence_length)

        return new_main

    def __len__(self):
        """
        전체 데이터 길이 반환
        """
        return self.total_len

    def __getitem__(self, idx):
        """
        DataLoader에서 인덱스로 호출될 때 실행
        Args:
            idx: 요청된 데이터 인덱스
        Returns:
            main_active_power: 메인 파워 시퀀스
            device_active_power: 디바이스 파워 라벨
        """
        # 캐시 부족 시 다음 parquet 파일 로드
        refresh_idx = 1
        if len(self.current_pd_indices_in_cache) <= refresh_idx:
            self.current_parquet_idx += 1
            if self.current_parquet_idx >= self.steps_cache:
                self.current_parquet_idx = 0

            self._cache_setting()
            self.current_segment_start_idx = 0

        # segment가 비었거나 새로 시작해야 하면 새로 생성
        if len(self.current_segment_idx) < 1 or self.current_segment_start_idx == 0:
            start_idx = self.current_segment_start_idx
            end_idx = start_idx + self.segment_length
            self.current_segment = self.transform_seq_window(self.current_pd.iloc[start_idx:end_idx]['main_power'].to_numpy())
            self.current_segment_label = self.current_pd.iloc[start_idx:end_idx]['active_power'].to_numpy().reshape(-1, 1)

            # segment 인덱스 리스트 생성 (역순으로 저장)
            self.current_segment_idx = list(range(0, len(self.current_segment)))[::-1]

        # 셔플 옵션이 있으면 인덱스 셔플
        if self.shuffle:
            rng = np.random.RandomState(seed=42)
            self.current_segment_idx = list(rng.permutation(self.current_segment_idx))

        # 다음 데이터 준비
        self.current_pd_indices_in_cache.pop()
        relative_idx = self.current_segment_idx.pop()

        # 시퀀스와 라벨 반환
        main_active_power = self.current_segment[relative_idx]
        device_active_power = self.current_segment_label[relative_idx]

        self.current_segment_start_idx += 1

        return main_active_power, device_active_power
