import eval
import train
import generate
import datetime
import time
from modules.config import read_config
import argparse
import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Generate datasets',
        description='Generates datasets for training, testing and validation based on a given function and noise level. Configurations are read from config.ini. The generated datasets are saved in a named data directory.',
        epilog='Example: python generate.py -c DEFAULT')
    parser.add_argument('-dir', '--directory',
                        help='Name of base directory where data will be stored', required=True)
    parser.add_argument('-p', '--profiles', nargs="+",
                        help='Specified profile to run', default="DEFAULT")
    parser.add_argument('-dp', '--data-profiles', nargs="+",
                        help='Specified data profile to run', default='DEFAULT')
    parser.add_argument('-g', '--generate',
                        help='Generate datasets', action='store_true')
    parser.add_argument('-t', '--train', help='Train model',
                        action='store_true')
    parser.add_argument(
        '-e', '--eval', help='Evaluate model', action='store_true')
    parser.add_argument('-dev', '--device',
                        help='Device to use for training', default='')
    parser.add_argument('-print', '--print',
                        help='Print training progress', action='store_true')
    args = parser.parse_args()

    if not args.generate and not args.train and not args.eval:
        parser.error(
            "at least one of --generate or --train or --eval is required")

    start = time.time()
    print(f"Start time: {datetime.datetime.now()}")

    # Set base directory
    BASE_DIR = args.directory
    if not os.path.exists(f"{BASE_DIR}"):
        raise ValueError(f"Base directory {BASE_DIR} not found")

    GENERATE = args.generate
    TRAIN = args.train
    EVAL = args.eval

    # Load configs
    if not os.path.exists(f"{BASE_DIR}/config.ini"):
        raise ValueError(f"Config file {BASE_DIR}/config.ini not found")

    configs = read_config(f"{BASE_DIR}/config.ini")
    dataset_configs = read_config(f"{BASE_DIR}/dataset_config.ini")

    # Validate profile arguments
    if args.profiles != "DEFAULT":
        for p in args.profiles:
            if p not in configs:
                raise ValueError(f"Profile {p} not found in config.ini")

    if args.data_profiles != "DEFAULT":
        for p in args.data_profiles:
            if p not in dataset_configs:
                raise ValueError(
                    f"Profile {p} not found in dataset_config.ini")

    # Set profiles to run
    profiles = args.profiles if args.profiles != "DEFAULT" else [
        p for p in configs if p != "DEFAULT"]
    dataset_profiles = args.data_profiles if args.data_profiles != "DEFAULT" else [
        p for p in dataset_configs if p != "DEFAULT"]

    # Generate datasets
    for p in dataset_profiles:
        config = dataset_configs[p]

        if GENERATE:
            if not os.path.exists(f"{BASE_DIR}/datasets"):
                os.mkdir(f"{BASE_DIR}/datasets")

            generate.gen(config, f"{BASE_DIR}/datasets")

    # Run pipeline for each config except DEFAULT
    for p in profiles:
        config = configs[p]
        NAME = config["NAME"]
        dataset_config = dataset_configs[config["DATASET"]]

        RERUNS = int(config["RERUNS"])

        # Train model
        if TRAIN:
            if not os.path.exists(f"{BASE_DIR}/models"):
                os.mkdir(f"{BASE_DIR}/models")

            if not os.path.exists(f"{BASE_DIR}/results"):
                os.mkdir(f"{BASE_DIR}/results")

            train.train(
                config, dataset_config, f"{BASE_DIR}", device=args.device, print_train=args.print, reruns=RERUNS)

        # Evaluate model
        if EVAL:
            if not os.path.exists(f"{BASE_DIR}/results"):
                os.mkdir(f"{BASE_DIR}/results")

            eval.eval(config, dataset_config,
                      f"{BASE_DIR}", device=args.device, reruns=RERUNS)

    end = time.time()
    print(f"End time: {datetime.datetime.now()}")
    td = datetime.timedelta(seconds=end-start)
    print(f"Total time: {td}")
