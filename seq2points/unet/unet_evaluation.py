import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader

# Set seeds
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)

from .unet.eval_dataloader import InsNilm
from .unet.model import UNetNiLM
from .unet.metrics import _compute_f1


def json_name_mapper(json_path):
    mapper = {}
    json_path_list = [f for f in Path(json_path).glob('**/*.json')]
    for f in json_path_list:
        if f.stem[0] == 'H':
            house_id, channel_id, collected_date = f.stem.split("_")
        else :
            user_id, channel, _, collected_date, _ = f.stem.split("_")
            house_id = f'H{int(user_id.replace("user","")):03}'
            channel_id = 'ch'+channel
        key = f'{house_id}_{channel_id}_{collected_date}'
        mapper[key] = f
    return mapper

def device_statistics(config):
    return pd.read_csv(config['stats_path'], index_col='channel')

def listup_test_dataset(data_dir):

    device_parquet_file_list = []
    for f in Path(data_dir).iterdir():
        if f.is_dir():
            if 'ch01' in f.stem:
                continue
            else :
                device_parquet_file_list.append(f.stem)

    return device_parquet_file_list

def main(config):

    channel_device = config['channel_device']
    file_list = listup_test_dataset(config['data_dir'])
    stats = device_statistics(config)
    config['json_name_mapper'] = json_name_mapper('labeling_dir')

    results = {
        'file_name' : [],
        'channel_id' : [],
        'f1': [],
        'tp': [],
        'fp': [],
        'fn': [],
    }
    file_count = 0
    total_file_counts = len(file_list)
    current_channel = None
    for file in file_list:
        file_count +=1

        user_id, channel, collected_date = Path(file).stem.split("_")
        channel_id = channel.replace("ch","")
        config['current_device_name'] = channel_device[channel_id]


        test_dataset = InsNilm(int(channel_id), 'test', config, str(file), stats)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=config['batch_size'])

        print(f"\nnow on {file_count}/{total_file_counts}...")
        print(f"Input Data : {user_id} {config['current_device_name']} Collected On {collected_date}")
        print(f"Input Data length: {len(test_dataset)} window")
        if current_channel != channel_id:
            current_channel = channel_id
            model = UNetNiLM(
                num_layers=config['num_layers'],
                features_start=8,
                n_channels=config['n_channels'],
                num_classes=config['num_classes'],
                pooling_size=config['pooling_size'],
                window_size=config['window_size'],
                num_quantiles=len(torch.tensor(config['taus'])),
                dropout=0.1,
                d_model=128,
            )

            CKP_PATH = Path(config['ckp_path_root']) / channel_id / f"{channel_id}-epoch-best.pth"
            if Path(CKP_PATH).exists():
                print(f'TRY LOAD CKP ... {CKP_PATH}')
                checkpoint = torch.load(CKP_PATH, map_location=config['device'])
                model.load_state_dict(checkpoint)
        else :
            print(f'USE Previous model LOADED ... {current_channel}, {channel}, {CKP_PATH}')

        # new_state_dict = OrderedDict()
        # if config['parallel']:
        #     model = nn.DataParallel(model)
        #     model.load_state_dict(checkpoint)
        # else :
        #     for k, v in checkpoint.items():
        #         name = k[7:] if k.startswith('module.') else k  # remove `module.` prefix if it exists
        #         new_state_dict[name] = v
        #     model.load_state_dict(new_state_dict)
        model = model.to(config['device'])
        
        pbar = tqdm(test_loader, desc=f"{config['current_device_name']} Evaluation", dynamic_ncols=True)
        current_batch_count = 0
        model.eval()

        current_batch_count = 0
        total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
        with torch.no_grad():
            for batch in pbar:
                current_batch_count += 1

                X, y_true = batch[0].float().to(config['device']), batch[1].to(config['device'])
                _, y_pred = model(X)
                y_pred_sigmoid = torch.sigmoid(y_pred).squeeze(-1) > 0.5
                _, tp, fp, fn, _, _ = _compute_f1(y_pred_sigmoid, y_true)
                total_tp += tp
                total_fp += fp
                total_fn += fn

                pbar.set_description(f"Evaluation On {config['current_device_name']} Active Power At {current_batch_count}-th batch")

        precision = total_tp / (total_tp + total_fp + 1e-8)
        recall = total_tp / (total_tp + total_fn + 1e-8)
        overall_f1 = 2 * precision * recall / (precision + recall + 1e-8)

        results['file_name'].append(Path(file).stem)
        results['channel_id'].append(channel_id)
        results['f1'].append(overall_f1.item())
        results['tp'].append(total_tp.item())
        results['fp'].append(total_fp.item())
        results['fn'].append(total_fn.item())

        print(f"\nModel evaluation results")
        print(f"Evaluation Device: {config['current_device_name']}")
        print("Avg F1 Score:", overall_f1.item())

        if not Path(config['output_dir']).exists():
            Path(config['output_dir']).mkdir(parents=True)

        pd.DataFrame(results).to_csv(f"{config['output_dir']}/unet_evaluation_results.csv")