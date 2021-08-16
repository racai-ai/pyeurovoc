import unittest
import requests


IP = "relate.racai.ro"
PORT = "8007"
TEST_FILE = "text.txt"


class MainTests(unittest.TestCase):
    def test_server(self):
        with open(TEST_FILE, "r", encoding="utf-8") as file:
            text = file.read()

        response = requests.post(url=f"http://{IP}:{PORT}/predict", json={"data": text})
        print(response.json())


if __name__ == "__main__":
    unittest.main()