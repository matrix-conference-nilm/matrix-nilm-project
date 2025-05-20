import logging
from datetime import datetime
import json
from pathlib import Path
import pandas as pd
import numpy as np

import torch
import wandb
from sklearn.model_selection import train_test_split


def get_str_now(to_sec=True):
    if to_sec:
        return datetime.strftime(datetime.now(), "%Y%m%d-%H:%M:%S")
    else :
        return datetime.strftime(datetime.now(), "%Y%m%d")


class TrainingHandler():
    def __init__(self, config):

        if config['wandb']:
            self.project = 'nilm-autolabel'
            self.wandb = wandb.init(
                project=self.project,
            )
            self.wandb.name = f"{get_str_now()}, window_size:{config['window_size']}, lr:{config['optim_kwargs']['lr']}"

        self.current_channel = None
        self.current_state = None
        self.ckp_path = Path(config['ckp_path_root'])
        self.current_state_json = None

        self.current_running_mse = float(np.inf)
        self.no_improve = 0
        self.patience = config.get('patience', 10)

    def logging_init(self, config):
        logger_path = Path(config["output_dir"])
        if not logger_path.exists():
            logger_path.mkdir(parents=True)

        logging.basicConfig(filename=f'{config["output_dir"]}/{config["fold"]}_log_{config["logger_name"]}.log', level=logging.INFO)
        self.logger = logging.getLogger('Training Logger')

    def wandb_logging(self, metrics):
        try :
            self.wandb.log({**metrics})
        except Exception as e:
            print(f"wandb logger error. {e}")
            return
    
    def logging_msg(self, msg):
        self.logger.info(msg)
    
    def save_state(self):
        if not (self.current_state_json.parent).exists():
            (self.current_state_json.parent).mkdir(parents=True, exist_ok=True)
        with open(self.current_state_json, 'w') as j:
            json.dump(self.current_state, j)

    def update_state(self, state_json):
        self.current_state = state_json

    def load_state(self, channel, epoch):
        if not Path(self.ckp_path).exists():
            Path(self.ckp_path).mkdir()
        if not Path(self.ckp_path / f"{channel}").exists():
            Path(self.ckp_path / f"{channel}").mkdir()

        self.current_state_json = self.ckp_path / f"{channel}" /f"state-{channel}.json"
        print(self.current_state_json)
        if not self.current_state_json.exists():
            _temp_json = {
                'batch_count': 0,
                'epoch': 0,
                'channel': channel,
                'train_best_loss' : None,
                'train_best_f1_score' : None,
                'val_best_loss' : None,
                'val_best_f1': None,
                'last_ckp': None
            }
            self.current_state = _temp_json
            self.save_state()

        else :
            with open(self.current_state_json, 'r') as j:
                _temp_json = json.load(j)
            self.update_state(_temp_json)

        self.no_improve = 0

    def get_state_value(self, key):
        """
        Get the value from the current state by key.
        """
        return self.current_state.get(key, None)

    def update_state_value(self, key, value):
        """
        Update the value in the current state for a given key.
        If the existing value is an array, append to it; otherwise, set the new value.
        """
        if key in self.current_state:
            if isinstance(self.current_state[key], list):
                self.current_state[key].append(value)
            else:
                self.current_state[key] = value
        else:
            # If the key doesn't exist, set the new value
            self.current_state[key] = value

        # Optionally, save the state after updating
        self.save_state()

    def check_early_stopping(self):
        """
        Check if early stopping criteria are met.
        Returns True if training should be stopped, otherwise False.
        """
        if self.no_improve >= self.patience:
            print("Early stopping triggered.")
            return True
        return False
    
    def save_model_ckp(self, model):
        channel = int(self.get_state_value('channel'))
        epoch = self.get_state_value('epoch')
        ckp_subpath = self.ckp_path / f"{channel:02d}"
        if not ckp_subpath.exists():
            ckp_subpath.mkdir(exist_ok=False, parents=False)

        # ckp file is TOO heavy save only BEST
        ckp_file = str((ckp_subpath / f"{channel:02d}-epoch-best.pth"))

        torch.save(model.state_dict(), ckp_file)
        self.update_state_value('last_ckp', ckp_file)
    

def make_training_data_split(config):

    DATA_PATH = Path(config['data_dir'])
    f_list = [f for f in DATA_PATH.iterdir()]

    data_dict = {
        "user_id": [],
        "channel" : [],
        "collected_date" : [],
        "f_path" : [],
    }
    for f in f_list:
        f = Path(f)
        user_id, channel, collected_date = f.stem.split("_")

        if channel != 'ch01':
            data_dict['user_id'].append(user_id)
            data_dict['channel'].append(channel)
            data_dict['collected_date'].append(collected_date)
            data_dict['f_path'].append(str(f))

            # 라벨링 데이터 pair
            json_file_name = Path(config['labeling_dir']) / f"{user_id}_{channel}_{collected_date}.json"
            data_dict['label_path'] = str(json_file_name)

    df_all = pd.DataFrame.from_dict(data_dict)
    X, test_sets = train_test_split(df_all, test_size=0.2, stratify=df_all['channel'])
    Val, test = train_test_split(test_sets, test_size=0.5, stratify=test_sets['channel'])
    return X, Val, test

def device_statistics(config):
    return pd.read_csv(config['stats_path'], index_col='channel')
