import os
import argparse
import yaml
from transformers import *
import re


def tokenizer_statistics():
    with open(args.config, "r") as config_fp:
        config = yaml.safe_load(config_fp)

    for directory in os.listdir("eurovoc"):
        lang = directory[:2]
        tokenizer = AutoTokenizer.from_pretrained(config[lang])

        list_unk_tokens = []
        unk_tokens = 0
        total_white_tokens = 0
        total_tokens = 0

        with open(os.path.join("eurovoc", directory, "acquis.cf"), "r", encoding="utf-8") as file:
            for line in file:
                line = re.sub(r"<.*?>", "", line)
                line = re.sub(r"\s+", " ", line)
                line = line.strip()
                tokens = tokenizer.encode(line[:-1], add_special_tokens=False)

                for token in tokens:
                    if token == tokenizer.unk_token_id:
                        # print(line)
                        # print(tokenizer.decode(tokens))
                        unk_tokens += 1

                total_white_tokens += len(line.split())
                total_tokens += len(tokens)

        print("Lang {}: Tok/Word: {}, UNK: {}".format(lang,
                                                      total_tokens / total_white_tokens,
                                                      unk_tokens))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="../pyeurovoc/configs/models.yml",
                        help="Tokenizer used for each language.")
    parser.add_argument("--data_path", type=str, default="eurovoc", help="Path to the EuroVoc data.")

    args = parser.parse_args()

    tokenizer_statistics()
