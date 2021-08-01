# EuroVoc-BERT

PyEuroVoc is a tool for legal document classification with [EuroVoc](https://eur-lex.europa.eu/browse/eurovoc.html) descriptors. It supports 22 languages: Bulgarian (bg), Czech (cs), Danish (da), German (de), Greek (el), English (en), Spanish (es), Estonian (et), Finnish (fi), French (fr), Hungarian (hu), Italian (it), Lithuanian (lt), Latvian (lv), Maltese (mt), Dutch (nl), Polish (pl), Portuguese (pt), Romanian (ro), Slovak (sk), Slovenian (sl), Sweedish (sv). 

The tool uses BERT at its core. The list of BERT variant for each language can be found [here]().

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
{'155': 0.9995473027229309, '230': 0.9377984404563904, '889': 0.9193254113197327, '1519': 0.714003324508667, '5020': 0.5, '5541': 0.5}
```

The number of most probable labels returned by the model is controlled by the `num_labels` parameter (default is 6).

``` python
prediction = model("Commission Decision on a modification of the system of aid applied in Italy in respect of shipbuilding", num_labels=4)
```

Which outputs:
``` python
{'155': 0.9995473027229309, '230': 0.9377984404563904, '889': 0.9193254113197327, '1519': 0.714003324508667}
```
