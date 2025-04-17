import pickle

import requests


class SynformerClient:
    """
    Client that communicates with the Synformer REST server.
    It pickles the request payload and sends it to the /predict endpoint.
    """

    def __init__(self, host: str = "localhost", port: int = 8000):
        self.base_url = f"http://{host}:{port}"

    def predict(self, input_dict: dict, encode: bool = False):
        input_dict["encode"] = encode
        data = pickle.dumps(input_dict)
        headers = {"Content-Type": "application/octet-stream"}
        resp = requests.post(f"{self.base_url}/predict", data=data, headers=headers)
        if resp.status_code != 200:
            raise Exception(f"Server returned status code {resp.status_code}")
        return pickle.loads(resp.content)
