# import abc
# from typing import Any

# from synformer.models.encoder.base import EncoderOutput
# from synformer.models.synformer import PredictResult

# import requests
# import pickle

# class SynformerInterface(abc.ABC):
#     """Interface for Synformer models."""

#     @abc.abstractmethod
#     def predict(self, input_dict: dict[str, Any]) -> PredictResult: ...

#     @abc.abstractmethod
#     def encode(self, input_dict: dict[str, Any]) -> EncoderOutput: ...


# class SynformerClient(SynformerInterface):
#     """Client for Synformer models."""

#     def __init__(self, host: str = "localhost", port: int = 8080):
#         self.host = host
#         self.port = port

#     def predict(self, input_dict: dict[str, Any]) -> PredictResult:
#         """Predict."""

#         url = f"http://{self.host}:{self.port}/predict"
#         input_dict_pkl = pickle.dumps(input_dict)
#         response = requests.post(url, data=input_dict_pkl)
#         return pickle.loads(response.content)

#     def encode(self, input_dict: dict[str, Any]) -> EncoderOutput:
#         """Encode."""

#         url = f"http://{self.host}:{self.port}/encode"
#         input_dict_pkl = pickle.dumps(input_dict)
#         response = requests.post(url, data=input_dict_pkl)
#         return pickle.loads(response.content)
