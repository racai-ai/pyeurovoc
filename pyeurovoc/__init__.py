import os
import torch
import pickle
import json
from transformers import AutoTokenizer


PYEUROVOC_PATH = os.path.join(os.path.expanduser("~"), ".cache", "pyeurovoc")
REPOSITORY_URL = ""

DICT_MODELS = {
    "bg": "TurkuNLP/wikibert-base-bg-cased",
    "cs": "TurkuNLP/wikibert-base-cs-cased",
    "da": "Maltehb/danish-bert-botxo",
    "de": "bert-base-german-cased",
    "el": "nlpaueb/bert-base-greek-uncased-v1",
    "en": "nlpaueb/legal-bert-base-uncased",
    "es": "dccuchile/bert-base-spanish-wwm-cased",
    "et": "tartuNLP/EstBERT",
    "fi": "TurkuNLP/bert-base-finnish-cased-v1",
    "fr": "camembert-base",
    "hu": "SZTAKI-HLT/hubert-base-cc",
    "it": "dbmdz/bert-base-italian-cased",
    "lt": "TurkuNLP/wikibert-base-lt-cased",
    "lv": "TurkuNLP/wikibert-base-lv-cased",
    "mt": "bert-base-multilingual-cased",
    "nl": "wietsedv/bert-base-dutch-cased",
    "pl": "dkleczek/bert-base-polish-cased-v1",
    "pt": "neuralmind/bert-base-portuguese-cased",
    "ro": "dumitrescustefan/bert-base-romanian-cased-v1",
    "sk": "TurkuNLP/wikibert-base-sk-cased",
    "sl": "TurkuNLP/wikibert-base-sl-cased",
    "sv": "KB/bert-base-swedish-cased"
}


class EuroVocBERT:
    def __init__(self, lang="en"):
        if lang not in DICT_MODELS.keys():
            raise ValueError("Language parameter must be one of the following languages: {}".format(DICT_MODELS.keys()))

        if not os.path.exists(PYEUROVOC_PATH):
            os.makedirs(PYEUROVOC_PATH)

        # model must be downloaded from the repostiory
        if not os.path.exists(os.path.join(PYEUROVOC_PATH, f"model_{lang}.pt")):
            print(f"Model 'model_{lang}.pt' not found in the .cache directory at '{PYEUROVOC_PATH}'. "
                  f"Downloading from '{REPOSITORY_URL}'...")
        # model already exists, loading from .cache directory
        else:
            print(f"Model 'model_{lang}.pt' found in the .cache directory at '{PYEUROVOC_PATH}'. "
                  f"Loading...")

        # load the model
        self.model = torch.load(os.path.join(PYEUROVOC_PATH, f"model_{lang}.pt"))

        # load the multi-label encoder for eurovoc, y, download from repository if not found in .cache directory
        if not os.path.exists(os.path.join(PYEUROVOC_PATH, f"mlb_encoder_{lang}.pickle")):
            print(f"Label encoder 'mlb_encoder_{lang}.pickle' not found in the .cache directory at '{PYEUROVOC_PATH}'."
                  f" Downloading from '{REPOSITORY_URL}'...")
        else:
            print(f"Label encoder 'mlb_encoder_{lang}.pickle' found in the .cache directory at '{PYEUROVOC_PATH}'."
                  f" Loading...")

        with open(os.path.join(PYEUROVOC_PATH, f"mlb_encoder_{lang}.pickle"), "rb") as pck_file:
            self.mlb_encoder = pickle.load(pck_file)

        # load MT descriptors dictionary, download from repository if not found in .cache directory
        if not os.path.exists(os.path.join(PYEUROVOC_PATH, "mt_labels.json")):
            print(f"MT descriptors dictionary 'mt_labels.json' not found in the .cache directory at '{PYEUROVOC_PATH}'."
                  f" Downloading from '{REPOSITORY_URL}'...")
        else:
            print(f"MT descriptors dictionary 'mt_labels.json' found in the .cache directory at '{PYEUROVOC_PATH}'. "
                  f"Loading...")

        with open(os.path.join(PYEUROVOC_PATH, "mt_labels.json"), "r") as json_file:
            self.dict_mt_labels = json.load(json_file)

        # load the tokenizer according to the model dictionary
        self.tokenizer = AutoTokenizer.from_pretrained(DICT_MODELS[lang])

    def __call__(self, document_text, num_id_labels=6):
        input_ids = self.tokenizer.encode(
            document_text,
            return_attention_mask=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).reshape(1, -1)

        with torch.no_grad():
            logits = self.model(
                input_ids,
                torch.ones_like(input_ids)
            )[0]

        probs = torch.sigmoid(logits).detach().cpu()

        probs_sorted, idx = torch.sort(probs, descending=True)

        outputs = torch.zeros_like(logits)
        outputs[idx[:num_id_labels]] = 1

        id_labels = self.mlb_encoder.inverse_transform(outputs.reshape(1, -1))[0]
        id_probs = probs[idx[:num_id_labels]]

        result = {}

        for id_label, id_prob in zip(id_labels, id_probs):
            result[str(id_label)] = float(id_prob)

        return result