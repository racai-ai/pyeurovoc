import argparse
import yaml
from load import load_data
from model import LangModelWithDense
from transformers import *
import torch
from utils import Meter
import os


def train_model(model, train_loader, dev_loader, optimizer, scheduler, criterion, lang, split_idx, device):
    meter = Meter()

    best_f1 = -1

    for epoch in range(args.epochs):
        model.train()
        loss, f1 = 0, 0
        print("Epoch: {}/{}".format(epoch + 1, args.epochs))

        for i, (train_x, train_mask, train_y) in enumerate(train_loader):
            if i % args.logging_step == 0:
                print("\tTraining - It: {}, Loss: {:.4f}, F1: {:.4f}".format(i , loss, f1))

            optimizer.zero_grad()

            logits = model.forward(train_x.to(device), train_mask.to(device))

            loss = criterion(logits.to(device), train_y.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            loss, f1 = meter.update_params(loss, logits.cpu(), train_y.cpu())

        optimizer.zero_grad()
        meter.reset()
        loss, f1 = 0, 0
        model.eval()

        for i, (dev_x, dev_mask, dev_y) in enumerate(dev_loader):
            if i % args.logging_step == 0:
                print("\tEvaluating - It: {}, Loss: {:.4f}, F1: {:.4f}".format(i, loss, f1))

            with torch.no_grad():
                logits = model.forward(dev_x.to(device), dev_mask.to(device))

                loss = criterion(logits.cpu(), dev_y.cpu())
                loss, f1 = meter.update_params(loss, logits.cpu(), dev_y.cpu())

        meter.reset()

        if f1 > best_f1:
            print("\n\tNew best model found: {:.4f} -> {:.4f}\n".format(best_f1, f1))
            torch.save(model, os.path.join(args.save_path, lang, "model_{}.pt".format(split_idx)))
            best_f1 = f1


def train():
    with open(args.config, "r") as config_fp:
        config = yaml.safe_load(config_fp)

    print("Models config:\n{}\n".format(config))

    device = torch.device(args.device)
    print("Working on device: {}\n".format(args.device))

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    print("Loading preprocessed datasets...")

    for lang in config.keys():
        if not lang == "ro":
            continue

        datasets = load_data(args.data_path, lang, args.batch_size)

        if not os.path.exists(os.path.join(args.save_path, lang)):
            os.makedirs(os.path.join(args.save_path, lang))

        for split_idx, (train_loader, dev_loader, _, num_classes) in enumerate(datasets):
            print("\nTraining for language: '{}' using: '{}'...".format(lang, config[lang]))

            lang_model = AutoModel.from_pretrained(config[lang])
            model = LangModelWithDense(lang_model, num_classes).to(device)

            optimizer = AdamW(model.parameters(), lr=5e-4)

            total_steps = len(train_loader) * args.epochs
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=len(train_loader),
                                                        num_training_steps=total_steps)

            criterion = torch.nn.BCEWithLogitsLoss()

            train_model(model, train_loader, dev_loader, optimizer, scheduler, criterion, lang, split_idx, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/models.yml", help="Tokenizer used for each language.")
    parser.add_argument("--data_path", type=str, default="data/eurovoc", help="Path to the EuroVoc data.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train the model.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size of the dataset.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to train on.")
    parser.add_argument("--save_path", type=str, default="models", help="Save path of the models")
    parser.add_argument("--max_grad_norm", type=int, default=5, help="Gradient clipping norm.")
    parser.add_argument("--logging_step", type=int, default=100)
    parser.add_argument("--verbose", type=int, default=0)

    args = parser.parse_args()

    train()
