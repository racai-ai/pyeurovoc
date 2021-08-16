# EuroVoc-BERT

PyEuroVoc is a tool for legal document classification with [EuroVoc](https://eur-lex.europa.eu/browse/eurovoc.html) descriptors. It supports 22 languages: Bulgarian (bg), Czech (cs), Danish (da), German (de), Greek (el), English (en), Spanish (es), Estonian (et), Finnish (fi), French (fr), Hungarian (hu), Italian (it), Lithuanian (lt), Latvian (lv), Maltese (mt), Dutch (nl), Polish (pl), Portuguese (pt), Romanian (ro), Slovak (sk), Slovenian (sl), Sweedish (sv). 

The tool uses BERT at its core. The list of BERT variant for each language can be found [here](https://github.com/racai-ai/pyeurovoc/blob/master/configs/models.yml). The performance of each model is outlined in our [paper](https://arxiv.org/abs/2108.01139).

## Installation

Make sure you have Python3 installed and a suitable package for PyTorch. Then, install `pyeurovoc` with pip:

```
pip install pyeurovoc
```

## Usage

Import the `EuroVocBERT` class from `pyeurovoc`. Instantiate the class with the desired langauge (default is "en") and then simply pass a document text to the model.

``` python
from pyeurovoc import EuroVocBERT

model = EuroVocBERT(lang="en")
prediction = model("Commission Decision on a modification of the system of aid applied in Italy in respect of shipbuilding")
```

The prediction of the model is a dictionary that contains the predicted ID descriptors as keys together with their confidence score as values.

``` python
{'155': 0.9990228414535522, '365': 0.9199643731117249, '431': 0.8993396759033203, '889': 0.6949650645256042, '1519': 0.03358537331223488, '5541': 0.03317505866289139}
```

The number of most probable labels returned by the model is controlled by the `num_labels` parameter (default is 6).


``` python
prediction = model("Commission Decision on a modification of the system of aid applied in Italy in respect of shipbuilding", num_labels=4)
```

Which outputs:
``` python
{'155': 0.9990228414535522, '365': 0.9199643731117249, '431': 0.8993396759033203, '889': 0.6949650645256042}
```

## Training your own models

### Download Dataset

Firstly, you need to download the datasets. Use the `download_datasets.sh` script in data to do that.

``` sh
./download_datasets.sh
```

### Preprocess

Once the datasets has finished downloading, you need to preprocess them using the `preprocess.py` script. It takes as input the model per language configuration file and the path to the dataset.

```
python preprocess.py --config [model_config] --data_path [dataset_path]
```

### Train

Training is done using the `train.py` script. It will automatically load the preprocessed files created by the previous step, and will save the best model for each split at the path given by the `-save_path` argument. To view the full list of available arguments, run `python train.py --help`.

```
python train.py --config [model_config] --data_path [dataset_path] 
                --epochs [n_epochs] --batch_size [batch_size] 
                --max_grad_norm [max_grad_norm]
                --device [device]
                --save_path [model_save_path]
                --logging_step [logging_step]
                --verbose [verbose]
```

### Evaluate

To evaluate the performance of each model on a split, run the `evaluate.py` script. As in the case of training, it provides several arguments that can be visualized with `python evaluate.py --help`.

```
python evaluate.py --config [model_config] --mt_labels [mt_labels_path] --data_path [dataset_path]
                   --models_path [models_ckpt_path] 
                   --batch_size [batch_size]
                   --device [device]
                   --output_path [results_output_path]
                   --loggin_step [logging_step]
                   --verbose [verbose]
```

## Acknowledgments

This research was supported by the EC grant no. INEA/CEF/ICT/A2017/1565710 for the Action no. 2017-EU-IA-0136 entitled “Multilingual Resources for CEF.AT in the legal domain” (MARCELL).

## Credits

Please consider citing the following [paper](https://arxiv.org/abs/2108.01139) as a thank you to the authors of the PyEuroVoc: 
```
Avram, Andrei-Marius, Vasile Pais, and Dan Tufis. "PyEuroVoc: A Tool for Multilingual Legal Document Classification with EuroVoc Descriptors." arXiv preprint arXiv:2108.01139 (2021).
```
or in .bibtex format:
```
@article{avram2021pyeurovoc,
  title={PyEuroVoc: A Tool for Multilingual Legal Document Classification with EuroVoc Descriptors},
  author={Avram, Andrei-Marius and Pais, Vasile and Tufis, Dan},
  journal={arXiv preprint arXiv:2108.01139},
  year={2021}
}
```
