import unittest
import os
import shutil
import yaml
from pyeurovoc import EuroVocBERT


class MainTests(unittest.TestCase):
    def test_eurovoc(self):
        pyeurovoc_path = os.path.join(os.path.expanduser("~"), ".cache", "pyeurovoc")

        if os.path.exists(pyeurovoc_path):
            print(f"Removing every model from .cache: {pyeurovoc_path}")
            shutil.rmtree(pyeurovoc_path)

        with open(os.path.join("..", "configs", "models.yml"), "r") as yml_file:
            dict_models = yaml.load(yml_file)

            for lang in dict_models:
                print("-" * 100)
                print(f"Testing for language: {lang}")
                model = EuroVocBERT(lang)
                outputs = model("This is a test text.")

            assert type(outputs) == dict and len(outputs) == 6


if __name__ == "__main__":
    unittest.main()