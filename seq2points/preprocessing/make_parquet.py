import time
from tqdm import tqdm
from pathlib import Path
import concurrent.futures
import numpy as np
import pandas as pd


def create_partition_column(df, timestamp_col, freq):
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    # Creating the partition column - round down to the nearest 2-hour block
    df['partition'] = df[timestamp_col].dt.floor(freq)
    df['partition'] = df['partition'].dt.strftime('%Y-%m-%dT%H-%M-%S')
    return df

"""
주어진 데이터프레임에서 특정 시간 열을 기준으로 파티션 열을 생성합니다.
시간 데이터를 주어진 주기(freq)에 맞춰 내림하여 파티션을 생성하고, 이를 문자열 형식으로 변환합니다.
"""

def save_parquet_partition(save_path, data_df, interval):
    data_df = create_partition_column(data_df, 'time_stamps', interval)
    partition_column = 'partition'
    data_df.to_parquet(save_path, index=False, partition_cols=[partition_column], engine='fastparquet', compression='snappy')

"""
데이터프레임을 Parquet 파일로 저장합니다.
create_partition_column 함수를 사용하여 파티션 열을 추가한 후, fastparquet 엔진을 사용하여 압축된 Parquet 파일로 저장합니다.
"""


def read_csv(f_path):
    data_df = pd.read_csv(f_path, engine='pyarrow')[['date_time','active_power']].rename(columns={'date_time':'time_stamps'})
    return data_df

"""
read_csv 함수:
CSV 파일을 읽어와서 date_time 열을 time_stamps로 이름을 변경하고, active_power 열과 함께 데이터프레임으로 반환합니다.
"""

def data_preprocessing(data_df):
    target_length = 2592000

    if len(data_df) != target_length:
        raise Exception(f"{len(data_df)} is not matching {target_length}")
    data_df.loc[data_df['active_power'] < 0, 'active_power'] = np.nan

    if sum(data_df['active_power'].isna()) > 0:
        print(f'DATA HAS NAN VALUE {sum(data_df["active_power"].isna())}')
    # Interpolate
    data_df['active_power'] = data_df['active_power'].interpolate(method='linear')
    # if first or last value is nan
    data_df['active_power'] = data_df['active_power'].fillna(0)
    # active - inactive labeling
    return data_df

"""
데이터프레임의 길이가 목표 길이(target_length)와 일치하는지 확인합니다.
active_power 값이 음수인 경우 NaN으로 설정하고, NaN 값을 선형 보간법으로 채웁니다.
처음이나 마지막 값이 NaN인 경우 0으로 채웁니다.
"""

def convert_to_parquet(csv_path, config):
    PREP_PARQUET_PATH = Path(config['output_dir'])
    
    if not PREP_PARQUET_PATH.exists():
        PREP_PARQUET_PATH.mkdir()

    house_id, channel_id, collected_date = Path(csv_path).stem.split("_")
    
    file_stem = f"{house_id}_{channel_id}_{collected_date}"
    save_file_name = PREP_PARQUET_PATH / f"{file_stem}.parquet"
    if save_file_name.exists(): return
    
    df = read_csv(csv_path)
    processed_data_df = data_preprocessing(df)
    save_parquet_partition(save_file_name, processed_data_df, '12H')

"""
CSV 파일 경로와 설정을 받아 Parquet 파일로 변환합니다.
출력 디렉토리가 존재하지 않으면 생성하고, 이미 존재하는 파일은 건너뜁니다.
CSV 파일을 읽고 전처리한 후, Parquet 파일로 저장합니다.
"""


def parallel(file_list, config):

    with concurrent.futures.ProcessPoolExecutor(max_workers=config['max_workers']) as executor:
        # Create a dictionary to map futures to file paths
        future_to_file = {executor.submit(convert_to_parquet, file_path, config): file_path for file_path in file_list}
        # Display progress bar with tqdm
        for future in tqdm(concurrent.futures.as_completed(future_to_file), total=len(file_list), desc='Processing files'):
            file_path = future_to_file[future]
            try:
                # Attempt to get the result of the future
                future.result()
            except Exception as exc:
                # Print the error along with the file path that caused it
                print(f'An error occurred with file {file_path}: {exc}')

"""
parallel 함수:
여러 CSV 파일을 병렬로 처리하여 Parquet 파일로 변환합니다.
concurrent.futures.ProcessPoolExecutor를 사용하여 병렬 처리를 수행하고, tqdm을 사용하여 진행 상황을 표시합니다.
"""


if __name__ == "__main__":
    import random

    f_list = [f for f in Path("/workspace/external_5").glob('**/*.csv')] + [f for f in Path("/workspace/external_6").glob('**/*.csv')] + [f for f in Path("/workspace/external_3").glob('**/*.csv')] + [f for f in Path("/workspace/external_4").glob('**/*.csv')]
    # label_list = [f for f in Path("/workspace/Data/raw_data/labeling_json").glob('**/*.json')]
    # json_csv_mapper = name_mapper(label_list)
    print(len(f_list))
    # for i in range(4):
    #     f = random.choice(f_list)
    #     start = time.time()
    #     csv_path = f
    #     print(csv_path)
    #     try :
    #         convert_to_parquet(f)
    #     except Exception as e:
    #         print(e)
    #         continue
    #     print(time.time() - start)

    start = time.time()
    main(f_list)
    print(f"preprocessing is {(time.time() - start):.1f}s")
    
"""
여러 디렉토리에서 CSV 파일 목록을 가져와 f_list에 저장합니다.
main 함수를 호출하여 파일 목록을 처리하고, 처리 시간을 출력합니다.
"""