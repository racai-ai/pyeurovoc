import argparse
import yaml
import os
import re
from transformers import AutoTokenizer
from skmultilearn.model_selection import IterativeStratification
from sklearn.preprocessing import MultiLabelBinarizer
import torch
import numpy as np
import pickle
from utils import SEEDS
from sklearn.utils import shuffle


def process_document(path, tokenizer, max_len=512):
    if args.verbose == 1:
        print("\n" + "*" * 50 + "\n")

    document_ct = 0
    big_document_ct = 0
    unk_ct = 0
    tokens_ct = 0

    list_inputs = []
    list_labels = []
    list_masks = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            labels = [int(x) for x in line.split()[:-2]]
            text = file.readline()
            text = re.sub(r"<.*?>", "", text)
            text = re.sub(r"\s+", " ", text)
            text = text.strip()

            if args.verbose == 1:
                print("Document nr: {}".format(document_ct + 1))
                print("EuroVoc labels: {}".format(labels))
                print("Text: {}".format(text))

            inputs_ids = torch.tensor(tokenizer.encode(text))

            if args.verbose == 1:
                print("Input ids: {}\n".format(inputs_ids))

            document_ct += 1

            for token in inputs_ids[1: -1]:
                if token == tokenizer.unk_token_id:
                    unk_ct += 1

                tokens_ct += 1

            if len(inputs_ids) > max_len:
                big_document_ct += 1
                inputs_ids = inputs_ids[:max_len]

            list_inputs.append(inputs_ids)
            list_labels.append(labels)
            list_masks.append(torch.ones_like(inputs_ids))

    print("Dataset stats - total documents: {}, big documents: {}, ratio: {:.4f}%".format(document_ct,
                                                                                          big_document_ct,
                                                                                          big_document_ct / document_ct * 100))
    print("              - total tokens: {}, unk tokens: {}, ratio: {:.4f}%".format(tokens_ct,
                                                                                    unk_ct,
                                                                                    unk_ct / tokens_ct * 100))

    return list_inputs, list_masks, list_labels


def save_splits(X, masks, y, directory):
    for i, seed in enumerate(SEEDS):
        stratifier = IterativeStratification(n_splits=2, order=2, sample_distribution_per_fold=[0.2, 0.8])
        train_idx, aux_idx = next(stratifier.split(X, y))
        train_X, train_mask, train_y = X[train_idx, :], masks[train_idx, :], y[train_idx, :]

        assert train_X.shape[0] == train_mask.shape[0] == train_y.shape[0]

        stratifier = IterativeStratification(n_splits=2, order=2, sample_distribution_per_fold=[0.5, 0.5])
        dev_idx, test_idx = next(stratifier.split(X[aux_idx, :], y[aux_idx, :]))
        dev_X, dev_mask, dev_y = X[dev_idx, :], masks[dev_idx, :], y[dev_idx, :]
        test_X, test_mask, test_y = X[test_idx, :], masks[test_idx, :], y[test_idx, :]

        assert dev_X.shape[0] == dev_mask.shape[0] == dev_y.shape[0]
        assert test_X.shape[0] == test_mask.shape[0] == test_y.shape[0]

        print("{} - Splitted the documents in - train: {}, dev: {}, test: {}".format(i, train_X.shape[0],
                                                                                     dev_X.shape[0],
                                                                                     test_X.shape[0]))

        if not os.path.exists(os.path.join(args.data_path, directory, "split_{}".format(i))):
            os.makedirs(os.path.join(args.data_path, directory, "split_{}".format(i)))

        np.save(os.path.join(args.data_path, directory, "split_{}".format(i), "train_X.npy"), train_X)
        np.save(os.path.join(args.data_path, directory, "split_{}".format(i), "train_mask.npy"), train_mask)
        np.save(os.path.join(args.data_path, directory, "split_{}".format(i), "train_y.npy"), train_y)

        np.save(os.path.join(args.data_path, directory, "split_{}".format(i), "dev_X.npy"), dev_X)
        np.save(os.path.join(args.data_path, directory, "split_{}".format(i), "dev_mask.npy"), dev_mask)
        np.save(os.path.join(args.data_path, directory, "split_{}".format(i), "dev_y.npy"), dev_y)

        np.save(os.path.join(args.data_path, directory, "split_{}".format(i), "test_X.npy"), test_X)
        np.save(os.path.join(args.data_path, directory, "split_{}".format(i), "test_mask.npy"), test_mask)
        np.save(os.path.join(args.data_path, directory, "split_{}".format(i), "test_y.npy"), test_y)

        X, masks, y = shuffle(X, masks, y, random_state=seed)
        

def process_datasets(data_path, directory, tokenizer_name):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    print("Processing Acquis...")
    list_ac_inputs, list_ac_masks, list_ac_labels = process_document(os.path.join(data_path, directory, "acquis.cf"),
                                                                     tokenizer)

    print("Processing OPOCE...")
    list_op_inputs, list_op_masks, list_op_labels = process_document(os.path.join(data_path, directory, "opoce.cf"),
                                                                     tokenizer)

    list_inputs = list_ac_inputs + list_op_inputs
    list_masks = list_ac_masks + list_op_masks
    list_labels = list_ac_labels + list_op_labels

    assert len(list_inputs) == len(list_masks) == len(list_labels)

    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(list_labels)

    X = torch.nn.utils.rnn.pad_sequence(list_inputs, batch_first=True, padding_value=tokenizer.pad_token_id).numpy()
    masks = torch.nn.utils.rnn.pad_sequence(list_masks, batch_first=True, padding_value=0).numpy()

    with open(os.path.join(args.data_path, directory, "mlb_encoder.pickle"), "wb") as pickle_fp:
        pickle.dump(mlb, pickle_fp, protocol=pickle.HIGHEST_PROTOCOL)

    save_splits(X, masks, y, directory)


def preprocess_data():
    with open(args.config, "r") as config_fp:
        config = yaml.safe_load(config_fp)

    print("Tokenizers config:\n{}\n".format(config))

    for directory in os.listdir(args.data_path):
        print("\nWorking on directory: {}...".format(directory))
        lang = directory[:2]
        print("Lang: '{}', Tokenizer: '{}'".format(lang, config[lang]))

        process_datasets(args.data_path, directory, config[lang])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="pyeurovoc/configs/models.yml", help="Tokenizer used for each language.")
    parser.add_argument("--data_path", type=str, default="data/eurovoc", help="Path to the EuroVoc data.")
    parser.add_argument("--verbose", type=int, default=0)

    args = parser.parse_args()

    preprocess_data()
