import os
import torch
import pickle
from transformers import AutoTokenizer, BertTokenizer
from .util import download_file
import re


PYEUROVOC_PATH = os.path.join(os.path.expanduser("~"), ".cache", "pyeurovoc")
REPOSITORY_URL = "https://relate.racai.ro/resources/eurovocbert/models/"

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
    def __init__(self, lang: str = "en", device: str = "cuda"):
        """
            Initialize the EuroVocBERT model.

            Args:
                lang (str): Language used as input for the model. Selects the corresponding BERT model.
                    Defaults to "en".
                device (str): Device to run the computation on. Defaults to "cuda".
        """
        self.device = torch.device(device)

        if lang not in DICT_MODELS.keys():
            raise ValueError("Language parameter must be one of the following languages: {}".format(DICT_MODELS.keys()))

        if not os.path.exists(PYEUROVOC_PATH):
            os.makedirs(PYEUROVOC_PATH)

        # model must be downloaded from the repostiory
        if not os.path.exists(os.path.join(PYEUROVOC_PATH, f"model_{lang}.pt")):
            print(f"Model 'model_{lang}.pt' not found in the .cache directory at '{PYEUROVOC_PATH}'\n"
                  f"Downloading from '{REPOSITORY_URL}'...")

            download_file(
                REPOSITORY_URL + f"model_{lang}.pt",
                os.path.join(PYEUROVOC_PATH, f"model_{lang}.pt")
            )
        # model already exists, loading from .cache directory
        else:
            print(f"Model 'model_{lang}.pt' found in the .cache directory at '{PYEUROVOC_PATH}'. "
                  f"Loading...")

        # load the model
        self.model = torch.load(os.path.join(PYEUROVOC_PATH, f"model_{lang}.pt"), map_location=self.device)
        self.model.eval()

        # load the multi-label encoder for eurovoc, y, download from repository if not found in .cache directory
        if not os.path.exists(os.path.join(PYEUROVOC_PATH, f"mlb_encoder_{lang}.pickle")):
            print(f"Label encoder 'mlb_encoder_{lang}.pickle' not found in the .cache directory at '{PYEUROVOC_PATH}'\n"
                  f" Downloading from '{REPOSITORY_URL}'...")

            download_file(
                REPOSITORY_URL + f"mlb_encoder_{lang}.pickle",
                os.path.join(PYEUROVOC_PATH, f"mlb_encoder_{lang}.pickle")
            )
        else:
            print(f"Label encoder 'mlb_encoder_{lang}.pickle' found in the .cache directory at '{PYEUROVOC_PATH}'."
                  f" Loading...")

        with open(os.path.join(PYEUROVOC_PATH, f"mlb_encoder_{lang}.pickle"), "rb") as pck_file:
            self.mlb_encoder = pickle.load(pck_file)

        # load the tokenizer according to the model dictionary
        if "wikibert" in DICT_MODELS[lang]:
            self.tokenizer = BertTokenizer.from_pretrained(DICT_MODELS[lang])
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(DICT_MODELS[lang])

    def __call__(self, document_text, num_labels=6):
        """
            EuroVoc descriptors prediction method.

            Args:
                document_text (str): The input document text.
                num_labels (int): Number of EuroVoc descriptors to return. Defaults to 6.
            Returns:
                A dictionary containing the first most probable "num_labels" together with their probabilities.
        """
        # clean the document text
        document_text = re.sub(r"<.*?>", "", document_text)
        document_text = re.sub(r"\s+", " ", document_text)
        document_text = document_text.strip()

        # tokenize the document text
        input_ids = self.tokenizer.encode(
            document_text,
            return_attention_mask=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).reshape(1, -1)

        # run the input tokens trough the model to obtain the logits
        with torch.no_grad():
            logits = self.model(
                input_ids.to(self.device),
                torch.ones_like(input_ids).to(self.device)
            )[0].detach().cpu()

        # obtain the probabilities and sort them descendingly
        probs = torch.sigmoid(logits)
        probs_sorted, idx_sort = torch.sort(probs, descending=True)

        # save the first 'num_labels' eurovoc ids and their probabilities
        result = {}
        for idx in idx_sort[:num_labels]:
            outputs = torch.zeros_like(logits)
            outputs[idx] = 1
            id_label = self.mlb_encoder.inverse_transform(outputs.reshape(1, -1))[0][0]

            result[str(id_label)] = float(probs[idx])

        return result