import argparse
import logging
from waitress import serve
from flask import Flask, Response, request
import yaml
from transformers import AutoTokenizer
import torch
import traceback
import json

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        content = request.json

        if content is None:
            return Response(response='{"message": "The POST request does not contain a JSON payload."}',
                            status=400, mimetype='application/json')

        if "data" not in content:
            return Response(response='{"message": "The field \"data\" was not found in the sent JSON payload."}',
                            status=400, mimetype='application/json')

        if not type(content["data"]) == str:
            return Response(response='{"message": "The field \"data\" must be a byte array encoded in Base64."}',
                            status=400, mimetype='application/json')

        input_ids = tokenizer.encode(content["data"], return_tensors="pt")
        mask = torch.ones_like(input_ids)

        output = model(input_ids, mask)[0]

        id_labels = torch.sort(output, descending=True)[1].tolist()
        mt_labels = [dict_mt_labels[str(label)]for label in id_labels  if str(label) in dict_mt_labels]
        do_labels = [dict_mt_labels[str(label)][:2] for label in id_labels if str(label) in dict_mt_labels]

        print(id_labels)
        print(mt_labels)
        print(do_labels)

        return {
            "id_labels": [str(label) for label in id_labels[:config["num_id_labels"]]],
            "mt_labels": mt_labels[:config["num_mt_labels"]],
            "do_labels": do_labels[:config["num_do_labels"]]
        }
    except:
        logging.error(traceback.format_exc())

        return Response(response='{"message": "An unexpected error occurred during transcription."}',
                        status=400, mimetype='application/json')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_config", type=str, default="pyeurovoc/configs/server.yml")
    args = parser.parse_args()

    with open(args.server_config, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    with open(config["mt_labels_path"], "r") as file:
        dict_mt_labels = json.load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.getLogger().setLevel(logging.INFO)

    logging.info('Setting up server...')

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["language_model"])
    model = torch.load(config["model"]["model_path"], map_location=device)
    model.eval()

    logging.info('Server initialised')

    serve(app, host=config["server"]["host"], port=config["server"]["port"])