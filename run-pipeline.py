import argparse
from modules.config import read_config
import generate
import train
import eval
import os



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog = 'Generate datasets',
                    description = 'Generates datasets for training, testing and validation based on a given function and noise level. Configurations are read from config.ini. The generated datasets are saved in a named data directory.',
                    epilog = 'Example: python generate.py -c DEFAULT')
    parser.add_argument('-dir', '--directory', help='Name of base directory where data will be stored', required=True)
    parser.add_argument('-p', '--profiles', nargs="+", help='Specified profile to run', default="DEFAULT")
    parser.add_argument('-dp', '--data-profiles', nargs="+", help='Specified data profile to run', default='DEFAULT')
    parser.add_argument('-g', '--generate', help='Generate datasets', action='store_true')
    parser.add_argument('-t', '--train', help='Train model', action='store_true')
    parser.add_argument('-e', '--eval', help='Evaluate model', action='store_true')
    args = parser.parse_args()

    # Set base directory
    BASE_DIR = args.directory
    if not os.path.exists(f"{BASE_DIR}"):
        raise ValueError(f"Base directory {BASE_DIR} not found")

    GENERATE = args.generate
    TRAIN = args.train
    EVAL = args.eval
    ALL = not GENERATE and not TRAIN and not EVAL

    # Load configs
    if not os.path.exists(f"{BASE_DIR}/config.ini"):
        raise ValueError(f"Config file {BASE_DIR}/config.ini not found")

    configs = read_config(f"{BASE_DIR}/config.ini")
    dataset_configs = read_config(f"{BASE_DIR}/dataset_config.ini")

    config_filter = [p for p in configs.keys() if p in args.profiles]
    dataset_filter = [p for p in dataset_configs.keys() if p in args.data_profiles]

    # Generate datasets
    for dc in [p for p in dataset_configs if p != "DEFAULT" and p in dataset_filter]:
        config = dataset_configs[dc]

        if GENERATE or ALL:
            if not os.path.exists(f"{BASE_DIR}/datasets"):
                os.mkdir(f"{BASE_DIR}/datasets")
            generate.gen(config, f"{BASE_DIR}/datasets")

    # Run pipeline for each config except DEFAULT
    for c in [p for p in configs if p != "DEFAULT" and p in config_filter]:
        config = configs[c]
        NAME = config["NAME"]
        dataset_config = dataset_configs[config["DATASET"]]

        # Train model
        if TRAIN or ALL:
            if not os.path.exists(f"{BASE_DIR}/{NAME}"):
                os.mkdir(f"{BASE_DIR}/{NAME}")
            inference_model = train.train(config, f"{BASE_DIR}")

        # Evaluate model
        if EVAL or ALL:
            if not os.path.exists(f"{BASE_DIR}/{NAME}"):
                os.mkdir(f"{BASE_DIR}/{NAME}")
            eval.eval(config, dataset_config, f"{BASE_DIR}")