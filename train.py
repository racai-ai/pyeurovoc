import argparse
import yaml
from load import load_data


def train():
    with open(args.config, "r") as config_fp:
        config = yaml.safe_load(config_fp)

    print("Models config:\n{}\n".format(config))

    dict_datasets = load_data(args.data_path, args.batch_size)

    for lang, (train_dataset, dev_dataset, test_dataset) in dict_datasets:
        print("Training for language: {} using: {}".format(lang, config[lang]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/models.yml", help="Tokenizer used for each language.")
    parser.add_argument("--data_path", type=str, default="data/eurovoc", help="Path to the EuroVoc data.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size of the dataset.")
    parser.add_argument("--verbose", type=int, default=0)

    args = parser.parse_args()

    train()
