import argparse
import yaml
from load import load_data
from model import LangModelWithDense
from transformers import *
from tqdm import tqdm
import torch
from utils import Meter
import os


def train_model(model, train_loader, dev_loader, optimizer, scheduler, criterion, lang, device):
    meter = Meter()

    best_f1 = -1

    for epoch in range(args.epochs):
        train_tqdm = tqdm(train_loader, leave=False)
        model.train()
        loss, f1 = 0, 0

        for i, (train_x, train_mask, train_y) in enumerate(train_tqdm):
            train_tqdm.set_description("Training - Epoch: {}/{}, Loss: {:.4f}, F1: {:.4f}".format(epoch + 1, args.epochs, loss, f1))
            train_tqdm.refresh()

            optimizer.zero_grad()

            logits = model.forward(train_x, train_mask)

            loss = criterion(logits.to(device), train_y.to(device))
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss, f1 = meter.update_params(loss, logits, train_y)

        meter.reset()

        dev_tqdm = tqdm(dev_loader, leave=False)
        loss, f1 = 0, 0
        model.eval()

        for i, (dev_x, dev_mask, dev_y) in enumerate(dev_tqdm):
            dev_tqdm.set_description("Evaluating - Epoch: {}/{}, Loss: {:.4f}, F1: {:.4f}".format(epoch + 1, args.epochs, loss, f1))
            dev_tqdm.refresh()

            optimizer.zero_grad()

            logits = model.forward(dev_x, dev_mask)

            loss = criterion(logits.to(device), dev_y.to(device))
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss, f1 = meter.update_params(loss, logits, dev_y)

        meter.reset()

        if f1 > best_f1:
            print("\nNew best model found: {:.4f} -> {:.4f}".format(best_f1, f1))
            torch.save(model, os.path.join(args.save_path, "{}_model.pt".format(lang)))


def train():
    with open(args.config, "r") as config_fp:
        config = yaml.safe_load(config_fp)

    print("Models config:\n{}\n".format(config))

    device = torch.device(args.device)
    print("Working on device: {}\n".format(args.device))

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    print("Loading preprocessed datasets...")
    dict_datasets = load_data(args.data_path, args.batch_size)

    for lang, (train_loader, dev_loader, test_loader, num_classes) in dict_datasets.items():
        print("\nTraining for language: '{}' using: '{}'...".format(lang, config[lang]))

        lang_model = AutoModel.from_pretrained(config[lang])
        model = LangModelWithDense(lang_model, num_classes)

        optimizer = AdamW(model.parameters(), lr=3e-5)

        total_steps = len(train_loader) * args.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=len(train_loader),
                                                    num_training_steps=total_steps)

        criterion = torch.nn.BCEWithLogitsLoss()

        train_model(model, train_loader, dev_loader, optimizer, scheduler, criterion, lang, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/models.yml", help="Tokenizer used for each language.")
    parser.add_argument("--data_path", type=str, default="data/eurovoc", help="Path to the EuroVoc data.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train the model.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size of the dataset.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to train on.")
    parser.add_argument("--save_path", type=str, default="models", help="Save path of the models")
    parser.add_argument("--verbose", type=int, default=0)

    args = parser.parse_args()

    train()
