import os
import argparse
import yaml
import torch
from load import load_data
from utils import Meter
import numpy as np
import json
import pickle


def evaluate_model(model, test_loader, mlb_encoder, mt_labels, device):
    meter = Meter(mt_labels=mt_labels, mlb_encoder=mlb_encoder)
    f1 = 0

    for i, (dev_x, dev_mask, dev_y) in enumerate(test_loader):
        if i > 2:
            break

        if i % args.logging_step == 0:
            print("Testing - It: {}, F1: {:.4f}".format(i, f1))
        logits = model.forward(dev_x.to(device), dev_mask.to(device))

        f1 = meter.update_params_eval(logits.cpu(), dev_y.cpu())

    return meter


def evaluate():
    with open(args.config, "r") as config_fp:
        config = yaml.safe_load(config_fp)

    with open(args.mt_labels, "r", encoding="utf-8") as file:
        mt_labels = json.load(file)

    print("Models config:\n{}\n".format(config))

    device = torch.device(args.device)
    print("Working on device: {}\n".format(args.device))

    for lang in config.keys():
        if not lang == "ro":
            continue

        with open(os.path.join(args.data_path, "{}-full-eurovoc-1.0".format(lang), "mlb_encoder.pickle"), "rb") as file:
            mlb_encoder = pickle.load(file)

        datasets = load_data(args.data_path, lang, args.batch_size)

        if not os.path.exists(os.path.join(args.output_path, lang)):
            os.makedirs(os.path.join(args.output_path, lang))

        f1k_scores = []
        f1k_mt_scores = []
        f1k_domain_scores = []

        for split_idx, (_, _, test_loader, _) in enumerate(datasets[:2]):
            print("\nEvaluating model: '{}'...".format("model_{}.pt".format(split_idx)))

            model = torch.load(os.path.join(args.models_path, lang, "model_{}.pt".format(split_idx)), map_location=device)
            model.eval()

            meter = evaluate_model(model, test_loader, mlb_encoder, mt_labels, device)

            print("Test results -  F1@6: {:.4f}, F1@6_MT: {:.4f}, F1@6_Domain: {:.4f}".format(meter.f1k,
                                                                                              meter.f1k_mt,
                                                                                              meter.f1k_domain))

            f1k_scores.append(meter.f1k * 100)
            f1k_mt_scores.append(meter.f1k_mt * 100)
            f1k_domain_scores.append(meter.f1k_domain * 100)

        print("\nOverall results for language '{}' - F1@6: {:.2f} ± ({:.2f}), F1@6_MT: {:.2f} ± "
              "({:.2f}), F1@6_Domain: {:.2f} ± ({:.2f})".format(lang,
                                                                np.mean(f1k_scores), np.std(f1k_scores),
                                                                np.mean(f1k_mt_scores), np.std(f1k_mt_scores),
                                                                np.mean(f1k_domain_scores), np.std(f1k_domain_scores)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="pyeurovoc/configs/models.yml", help="Tokenizer used for each language.")
    parser.add_argument("--mt_labels", type=str, default="pyeurovoc/resources/mt_labels.json")
    parser.add_argument("--data_path", type=str, default="data/eurovoc", help="Path to the EuroVoc data.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to train on.")
    parser.add_argument("--models_path", type=str, default="models", help="Path of the saved models.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size of the dataset.")
    parser.add_argument("--output_path", type=str, default="output", help="Models evaluation output path.")
    parser.add_argument("--logging_step", type=int, default=100)
    parser.add_argument("--verbose", type=int, default=0)

    args = parser.parse_args()

    evaluate()