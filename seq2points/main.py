import argparse
import constant


from preprocessing.make_parquet import convert_to_parquet, parallel
from seq2points import seq2point_evaluation, seq2point_training
from unet import unet_evaluation, unet_training
from utils.file_handler import listup_files


def main(args):

    # Example usage of arguments and constants
    print(f"RUNNING : {args.f}")
    print(f"DATA DIR: {args.data_dir}")

    # You can modify hyperparameters based on arguments
    if args.f == 'prep':
        # Example modification
        config = constant.configs['prep']
        config['data_dir'] = args.data_dir if args.data_dir else config['data_dir']
        config['output_dir'] = args.output_dir if args.output_dir else config['output_dir']
        file_list = listup_files(args.data_dir, config['suffix'])
        
        if args.parallel == 0:
            for file in file_list:
                convert_to_parquet(file, config)
        else :
            config['max_workers'] = args.parallel
            parallel(file_list, config)

    elif args.f == 'seq2points':
        config = constant.configs['sep2points']
        config['channel_device'] = constant.configs['channel_device']
        config['data_dir'] = args.data_dir if args.data_dir else config['data_dir']
        config['output_dir'] = args.output_dir if args.output_dir else config['output_dir']
        config['batch_size'] = args.batch_size if args.batch_size != 128 else config['batch_size']
        config['ckp_path_root'] = args.ckp_path_root if args.ckp_path_root != None else config['ckp_path_root']
        config['fold'] = args.fold

        print(f"OUTPUT DIR : {config['output_dir']}")
        print(f"CKP DIR: {config['ckp_path_root']}\n")
        if args.fold == 'eval':
            seq2point_evaluation.main(config)
        elif args.fold == 'train':
            seq2point_training.main(config)

    elif args.f == 'unet':
        config = constant.configs['unet']
        config['channel_device'] = constant.configs['channel_device']
        config['data_dir'] = args.data_dir if args.data_dir else config['data_dir']
        config['output_dir'] = args.output_dir if args.output_dir else config['output_dir']
        config['labeling_dir'] = args.labeling_dir if args.labeling_dir else config['labeling_dir']
        config['batch_size'] = args.batch_size if args.batch_size != 128 else config['batch_size']
        config['ckp_path_root'] = args.ckp_path_root if args.ckp_path_root != None else config['ckp_path_root']
        config['fold'] = args.fold

        if args.labeling_dir is None:
            raise Exception(f"LABEL DIR IS None")
        
        print(f"OUTPUT DIR : {config['output_dir']}")
        print(f"LABEL DIR: {config['labeling_dir']}")
        print(f"CKP DIR: {config['ckp_path_root']}\n")

        if args.fold == 'eval':
            unet_evaluation.main(config)
        elif args.fold == 'train':
            unet_training.main(config)

    else :
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="It requires specifying directories for data, output, and checkpoint paths. ")
    
    parser.add_argument('--data_dir', type=str, required=True, help='')
    parser.add_argument('--output_dir', type=str, required=True, help='')
    parser.add_argument('--ckp_path_root', type=str, required=False)
    parser.add_argument('--labeling_dir', type=str, default=None, help='')
    parser.add_argument('--f', type=str, help='')
    parser.add_argument('--fold', type=str, required=False, help='')

    parser.add_argument('--seed', type=int, default=42, help='')
    parser.add_argument('--parallel', type=int, default=0, help='')
    parser.add_argument('--batch_size', type=int, default=128, help='')

    args = parser.parse_args()

    # Conditional check
    if args.f != "pred" and args.ckp_path_root is None:
        parser.error("--ckp_path_root is required when --f is not 'pred'")
    if args.f != "pred" and args.labeling_dir is None:
        parser.error("--labeling_dir is required when --f is not 'pred'")

    if args.f != "pred" and args.fold is None:
        parser.error("--fold is required when --f is not 'pred' 'train' OR 'eval'")
    main(args)