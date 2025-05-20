from pathlib import Path
import pandas as pd
import numpy as np
import fastparquet
from numpy.lib.stride_tricks import sliding_window_view

import torch
from torch.utils.data import Dataset


class EvalDataset(Dataset):  
    """
    평가용 데이터셋 클래스
    - 학습 데이터셋과 달리 전체 데이터를 한 번에 메모리에 로드
    - 시계열 데이터를 시퀀스 윈도우로 변환하여 반환
    - 정규화 수행
    """
    def __init__(self, num_cached_parquet, channel, device_path, main_path, _stat, config):
        """
        Args:
            num_cached_parquet: 캐시할 parquet 파일 개수 (사용되지 않음)
            channel: 사용할 디바이스 채널 인덱스
            device_path: 디바이스 파워 parquet 파일 경로 (리스트 또는 문자열)
            main_path: 메인 파워 parquet 파일 경로 (리스트 또는 문자열)
            _stat: 정규화에 사용할 통계값이 담긴 DataFrame
            config: 설정 정보 (시퀀스 길이, 데이터 디렉토리 등)
        """
        # 기본 설정
        self.raw_cols = 'active_power'  # parquet 파일에서 사용할 column
        self.channel = channel  # int channel
        self.sequence_length = config['sequence_length']  # 시퀀스 윈도우 길이
        self.units_to_pad = self.sequence_length // 2  # 시퀀스 앞뒤로 패딩할 길이
        self.data_root = config['data_dir']  # 데이터 파일 경로

        # 정규화에 사용할 통계값
        self.stat_df = _stat 
        self.main_mean, self.main_std = _stat['mean_val'][1], _stat['std_dev'][1]  # 메인 파워 통계
        self.device_mean, self.device_std = _stat['mean_val'][self.channel], _stat['std_dev'][self.channel]  # 디바이스 파워 통계

        # parquet 파일 경로 설정
        if isinstance(device_path, list):
            # 여러 파일을 평가할 경우
            self.parquet_list = [self.data_root+Path(x).stem+'.parquet' for x in device_path]
            self.parquet_main = [self.data_root+Path(x).stem+'.parquet' for x in main_path]
        elif isinstance(device_path, str):
            # 단일 파일을 평가할 경우
            self.parquet_list = [self.data_root+Path(str(device_path)).stem+'.parquet']
            self.parquet_main = [self.data_root+Path(str(main_path)).stem+'.parquet']

            # parquet 파일 로드
            self.fparquet = fastparquet.ParquetFile(self.parquet_list[0])
            self.fparquet_main = fastparquet.ParquetFile(self.parquet_main[0])

        # 데이터 캐싱 및 초기화
        self._cache_setting()
        self.total_len = self.get_total_length()

    def _cache_setting(self):
        """
        전체 데이터를 메모리에 로드하고 시퀀스 윈도우로 변환
        """
        self.new_main, self.device_active_power = self._cache_parquet()

    def get_total_length(self):
        """
        전체 데이터 길이 반환
        Returns:
            total_len: 전체 데이터 길이 (마지막 시퀀스 제외)
        """
        fdf = fastparquet.ParquetFile(self.parquet_list)
        total_len = 0
        for _df in fdf.iter_row_groups(columns=['active_power']):
            total_len += len(_df)
        return total_len - 1  # 마지막 시퀀스 제외

    def _cache_parquet(self):
        """
        parquet 파일을 읽어와 DataFrame으로 합치고 시퀀스 윈도우로 변환
        Returns:
            new_main: 시퀀스 윈도우로 변환된 메인 파워 데이터
            active_power: 디바이스 파워 데이터
        """
        # 디바이스/메인 파워 데이터 합치기
        list_df = []
        for _df in zip( self.fparquet.iter_row_groups(columns=[self.raw_cols]),\
                        self.fparquet_main.iter_row_groups(columns=[self.raw_cols]) ):
            _device = _df[0]
            _device['main_power'] = _df[1]['active_power']
            list_df.append(_device)
        df_data = pd.concat(list_df)

        # 정규화 수행
        df_data['active_power'] = self.normalization(df_data['active_power'], self.device_mean, self.device_std)
        df_data['main_power'] = self.normalization(df_data['main_power'], self.main_mean, self.main_std)
        active_power = df_data['active_power'].values

        # 시퀀스 윈도우 생성
        # 1. 앞뒤로 패딩
        new_main = np.pad(df_data['main_power'].values, (self.units_to_pad, self.units_to_pad), 'constant', constant_values=(0, 0))
        # 2. sliding_window_view를 사용하여 효율적으로 시퀀스 생성
        new_main = sliding_window_view(new_main, window_shape=self.sequence_length)
        new_main = new_main.reshape(-1, new_main.shape[-1])
        new_main = new_main[0:-1, :]  # 마지막 시퀀스 제외
        new_main = new_main.reshape(-1, 1, self.sequence_length)

        return new_main, active_power

    def normalization(self, data_array, mean, std):
        """
        데이터 정규화: (값 - 평균) / 표준편차
        """
        return (data_array - mean) / std

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
            main_active_power: 메인 파워 시퀀스 (PyTorch tensor)
            device_active_power: 디바이스 파워 라벨 (PyTorch tensor)
        """
        # 메모리에서 데이터 가져오기
        main_active_power = self.new_main[idx]
        device_active_power = self.device_active_power[idx]

        # numpy 배열을 쓰기 가능한 복사본으로 변환
        main_active_power = np.copy(main_active_power)
        device_active_power = np.copy(device_active_power)

        # PyTorch tensor로 변환
        main_active_power = torch.from_numpy(main_active_power)
        device_active_power = torch.from_numpy(device_active_power)

        return main_active_power, device_active_power