import os
import torch
import yaml
from transformers import AutoTokenizer


PYEUROVOC_PATH = os.path.join(os.path.expanduser("~"), ".cache", "pyeurovoc")
REPOSITORY_URL = ""


class EuroVocBERT:
    def __init__(self, lang="en"):
        if not os.path.exists(PYEUROVOC_PATH):
            os.makedirs(PYEUROVOC_PATH)

        # model must be downloaded from the repostiory
        if not os.path.exists(os.path.join(PYEUROVOC_PATH, f"model_{lang}.pt")):
            print(f"Model 'model_{lang}.pt' not found in the .cache directory at '{PYEUROVOC_PATH}'")
            print(f"Downloading 'model_{lang}.pt from {REPOSITORY_URL}...")
        # model already exists, loading from .cache directory
        else:
            print(f"Model 'model_{lang}.pt' found in the .cache directory at '{PYEUROVOC_PATH}'")
            print("Loading model...")

        # load the model
        self.model = torch.load(os.path.join(PYEUROVOC_PATH, f"model_{lang}.pt"))

        # load the model dictionary (e.g. language -> bert_model)
        with open(os.path.join("configs", "models.yml"), "r") as yml_file:
            dict_models = yaml.load(yml_file)

        # load the tokenizer according to the model dictionary
        self.tokenizer = AutoTokenizer.from_pretrained(dict_models[lang])

    def __call__(self, document_text, num_id_labels=6, num_mt_labels=5, num_do_labels=4):
        encoding_ids = self.tokenizer.encode(
            document_text,
            return_attention_mask=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
