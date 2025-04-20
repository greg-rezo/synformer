import logging
import multiprocessing as mp
import pickle
from base64 import b64decode, b64encode
from contextlib import asynccontextmanager
from dataclasses import fields
from io import BytesIO
from multiprocessing.context import SpawnProcess
from tempfile import TemporaryDirectory
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, Request, Response
from pydantic import BaseModel, ConfigDict, model_validator

from synformer.chem.fpindex import FingerprintIndex
from synformer.chem.matrix import ReactantReactionMatrix
from synformer.models.encoder.base import EncoderOutput
from synformer.models.synformer import PredictResult, Synformer

logger = logging.getLogger(__name__)


class ServerState(BaseModel):
    """State for the Synformer REST server."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: torch.nn.Module
    fpindex: FingerprintIndex
    rxn_matrix: ReactantReactionMatrix

    @property
    def device(self) -> torch.device:
        """Get the device to use for the Synformer model."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    @property
    def on_cpu(self) -> bool:
        """Check if the Synformer model is on the CPU."""
        return self.device == torch.device("cpu")

    @model_validator(mode="after")
    def move_to_device(self):
        """Move the Synformer model to the correct device."""
        self.model = self.model.to(self.device)
        return self


def to_device(obj: Any, device: torch.device) -> Any:
    """Move an object to a device."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(to_device(v, device) for v in obj)
    elif isinstance(obj, (EncoderOutput, PredictResult)):
        return type(obj)(
            **{
                field.name: to_device(getattr(obj, field.name), device)
                for field in fields(obj)
            }
        )
    return obj


# def torch_to_json(obj: Any) -> dict[str, str]:
#     """Convert a torch object to a json serializable object."""
#     stream = BytesIO()
#     torch.save(obj, stream)
#     return {"data": b64encode(stream.getvalue()).decode("utf-8")}


# def json_to_torch(json_str: dict[str, str]) -> Any:
#     """Convert a json string to a torch object."""
#     return torch.load(BytesIO(b64decode(json_str["data"])))


def torch_to_pickle(obj: Any) -> dict[str, str]:
    """Convert a torch object to a json serializable object."""
    stream = BytesIO()
    pickle.dump(obj, stream)
    return {"data": b64encode(stream.getvalue()).decode("utf-8")}


def pickle_to_torch(json_str: dict[str, str]) -> Any:
    """Convert a json string to a torch object."""
    return pickle.load(BytesIO(b64decode(json_str["data"])))


server_state: ServerState | None = None


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     """Lifespan for the Synformer REST server."""
#     global server_state

#     with TemporaryDirectory() as temp_dir:
#         fpindex, rxn_matrix, model = (
#             load_fpindex(out_dir=temp_dir),
#             load_rxn_matrix(out_dir=temp_dir),
#             load_model(out_dir=temp_dir),
#         )

#     server_state = ServerState(
#         model=model,
#         fpindex=fpindex,
#         rxn_matrix=rxn_matrix,
#     )
#     yield


app = FastAPI(title="Synformer REST Server")


@app.get("/")
async def root():
    """Return a message to the user."""
    return {"message": "Synformer REST API"}


@app.get("/healthy")
async def healthy():
    """Return a message to the user."""
    return {"message": "Synformer REST API is healthy"}


# @app.post("/predict")
# async def predict(request: Request):
#     """
#     REST endpoint that accepts a pickled request and returns a pickled response.
#     The request should be a pickled dict containing model inputs and a boolean "encode" flag.
#     """
#     assert server_state is not None

#     data = await request.body()
#     try:
#         req = pickle.loads(data)
#     except Exception as e:
#         logger.error("Error unpickling request", exc_info=e)
#         return Response(
#             content=b"", media_type="application/octet-stream", status_code=400
#         )
#     encode = req.pop("encode", False)

#     # If needed, convert parts of req to torch tensors here.
#     # This example assumes that req (a dict) is already in the appropriate structure.
#     input_dict = {}
#     for k, v in req.items():
#         if isinstance(v, torch.Tensor):
#             input_dict[k] = v.to(device)
#         else:
#             input_dict[k] = v
#     if encode:
#         result = model.encode(input_dict)
#         result.code = result.code.cpu()
#         result.code_padding_mask = result.code_padding_mask.cpu()
#         for k, v in result.loss_dict.items():
#             result.loss_dict[k] = v.cpu()
#     else:
#         result = model.predict(
#             **input_dict,
#             fpindex=server_state["fpindex"],
#             rxn_matrix=server_state["rxn_matrix"],
#         )
#         result.token_logits = result.token_logits.cpu()
#         result.token_sampled = result.token_sampled.cpu()
#         result.reaction_logits = result.reaction_logits.cpu()

#     result_data = pickle.dumps(result)
#     return Response(content=result_data, media_type="application/octet-stream")


@app.post("/predict")
async def predict(inputs: dict[str, Any]) -> dict[str, str]:
    """Make predictions with the synformer model."""
    assert server_state is not None

    # Convert input pickle to torch
    inputs = pickle_to_torch(inputs)
    inputs = to_device(inputs, server_state.device)

    result = server_state.model.predict(
        **inputs,
        fpindex=server_state.fpindex,
        rxn_matrix=server_state.rxn_matrix,
    )

    # Convert results to pickle
    results = to_device(result, torch.device("cpu"))
    results_str = torch_to_pickle(results)

    return results_str


@app.post("/encode")
async def encode(inputs: bytes) -> bytes:
    """Make predictions with the synformer model."""
    assert server_state is not None

    # Convert input pickle to torch
    inputs = pickle.loads(inputs)
    inputs = to_device(inputs, server_state.device)

    result = server_state.model.encode(inputs)

    # Convert results to pickle
    result = to_device(result, torch.device("cpu"))
    result = pickle.dumps(result)

    return result


def _start_server(
    model: Synformer,
    fpindex: FingerprintIndex,
    rxn_matrix: ReactantReactionMatrix,
    host: str = "0.0.0.0",
    port: int = 8000,
):
    global server_state

    logging.basicConfig(level=logging.DEBUG)
    logger.info(f"Starting server on {host}:{port}")

    server_state = ServerState(
        model=model,
        fpindex=fpindex,
        rxn_matrix=rxn_matrix,
    )

    uvicorn.run(app, host=host, port=port, log_level=logging.WARNING)


def launch_server_process(
    model: Synformer,
    fpindex: FingerprintIndex,
    rxn_matrix: ReactantReactionMatrix,
    host: str = "0.0.0.0",
    port: int = 8000,
) -> SpawnProcess:
    """Launch a server process."""
    mp_context = mp.get_context("spawn")
    p = mp_context.Process(
        target=_start_server,
        args=(model, fpindex, rxn_matrix, host, port),
    )
    p.start()
    return p
