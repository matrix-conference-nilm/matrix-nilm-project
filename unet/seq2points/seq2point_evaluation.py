from pathlib import Path
import pandas as pd

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import OrderedDict

from .seq2point.eval_dataloader import EvalDataset
from .seq2point.model import Seq2Point


def device_statistics(config):
    return pd.read_csv(config['stats_path'], index_col='channel')

def listup_test_dataset(data_dir):

    device_parquet_file_list = []
    main_parquet_file = []
    for f in Path(data_dir).iterdir():
        if f.is_dir():
            if 'ch01' in f.stem:
                continue
            else :
                device_parquet_file_list.append(f.stem)
    
    for device in device_parquet_file_list:
        # H001_ch01_20230922.parquet
        user_id, _, collected_date = device.split("_")
        main_parquet = f"{user_id}_ch01_{collected_date}"
        main_parquet_file.append(main_parquet)

    return main_parquet_file, device_parquet_file_list


def main(config):

    channel_device = config['channel_device']
    main_parquet_list, device_parquet_file_list = listup_test_dataset(config['data_dir'])
    stats = device_statistics(config)

    result_grabber = {
        'test_file_name' : [],
        'mae_score': [],
    }

    total_file_counts = len(device_parquet_file_list)
    file_count = 0
    current_channel_id = None
    for device_parquet_file, main_parquet_file in zip(device_parquet_file_list, main_parquet_list):
        file_count +=1
        user_id, channel_id, collected_date = Path(device_parquet_file).stem.split("_")

        collected_date = collected_date.replace(".parquet", "")
        channel_id = channel_id.replace('ch','')
        
        device_channel = int(channel_id)
        device_name = channel_device[f"{device_channel:02}"]
        
        if current_channel_id != channel_id:
            current_channel_id = channel_id
            print(f"\n\nPrepare Model Evaluation on {device_name}\nSeq2point Model Checkpoint loading ...\n")

            CKP_PATH = Path(config['ckp_path_root']) / channel_id / f"{channel_id}-epoch-000.pth"
            model = Seq2Point(config['sequence_length'], config['device']).to(config['device'])
            checkpoint = torch.load(CKP_PATH, map_location=config['device'])
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                name = k[7:] if k.startswith('module.') else k  # remove `module.` prefix if it exists
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        else :
            print(f"\n\nPrepare Model Evaluation on {device_name}\nSeq2point Model From Previous Loaded model ...\n")

        dataset = EvalDataset(
            num_cached_parquet=10,
            channel=device_channel,
            device_path=device_parquet_file,
            main_path=main_parquet_file,
            _stat = stats,
            config=config
        )

        eval_loader = DataLoader(dataset, shuffle=False, 
                                batch_size=config['batch_size'], 
                                num_workers=config['num_workers'])

        device_mean = stats.loc[int(device_channel)]['mean_val']
        device_std = stats.loc[int(device_channel)]['std_dev']
        user_id, _, _ = Path(main_parquet_file).stem.split("_")

        print(f"now on {file_count}/{total_file_counts}...")
        print(f"Input Data : {user_id} 메인 분전반 Collected On {collected_date}")
        print(f"Input Data length: {len(dataset)}time-points")
        config['current_device_name'] = device_name

        i = 1
        model.eval()
        batch_mae_score = []
        with torch.no_grad():
            pbar = tqdm(eval_loader, desc=f"{config['current_device_name']} Evaluation", dynamic_ncols=True)
            for b in pbar:
                main, device = b
                main = main.to(config['device']).float()
                device = device.to(config['device']).float()

                y_pred = model(main)
                y_pred_denormalized = (y_pred * device_std) + device_mean
                device_denormalized = (device * device_std) + device_mean

                batch_mae = torch.mean(torch.abs(device_denormalized - y_pred_denormalized)).item()
                batch_mae_score.append(batch_mae)
                pbar.set_description(f"Evaluation On {config['current_device_name']} Active Power At {i}-th batch")
                i+=1


        mae_score = sum(batch_mae_score)/len(batch_mae_score)

        print(f"\nModel evaluation results")
        print(f"Evaluation Device: {config['current_device_name']}")
        print("MAE Score:", mae_score)

        result_grabber['test_file_name'].append(device_parquet_file)
        result_grabber['mae_score'].append(mae_score)

        if not Path(config['output_dir']).exists():
            Path(config['output_dir']).mkdir(parents=True)
        
        pd.DataFrame(result_grabber).to_csv(f"{config['output_dir']}/seq2points_evaluation_results.csv")