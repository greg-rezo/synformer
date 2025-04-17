import logging
import multiprocessing as mp
import pickle
from multiprocessing.context import SpawnProcess
from pathlib import Path

import requests
import torch
import uvicorn
from cloudpathlib import GSPath
from fastapi import FastAPI, Request, Response
from omegaconf import OmegaConf

from synformer.chem.fpindex import FingerprintIndex
from synformer.chem.matrix import ReactantReactionMatrix
from synformer.models.synformer import Synformer

GS_ROOT_DIR = GSPath("gs://rezo-artifacts/synformer/data")
GS_MATRIX_PKL_FILE = GS_ROOT_DIR / "matrix.pkl"
GS_FPINDEX_PKL_FILE = GS_ROOT_DIR / "fpindex.pkl"
GS_MODEL_FILE = GS_ROOT_DIR / "trained_weights/sf_ed_default.ckpt"

logger = logging.getLogger(__name__)

app = FastAPI(title="Synformer REST Server")

# Global server state
server_state = {}


@app.post("/predict")
async def predict(request: Request):
    """
    REST endpoint that accepts a pickled request and returns a pickled response.
    The request should be a pickled dict containing model inputs and a boolean "encode" flag.
    """
    data = await request.body()
    try:
        req = pickle.loads(data)
    except Exception as e:
        logger.error("Error unpickling request", exc_info=e)
        return Response(
            content=b"", media_type="application/octet-stream", status_code=400
        )
    encode = req.pop("encode", False)

    model = server_state.get("model")
    device = server_state.get("device")
    if model is None or device is None:
        logger.error("Server state not initialized")
        return Response(
            content=b"", media_type="application/octet-stream", status_code=500
        )

    # If needed, convert parts of req to torch tensors here.
    # This example assumes that req (a dict) is already in the appropriate structure.
    input_dict = {}
    for k, v in req.items():
        if isinstance(v, torch.Tensor):
            input_dict[k] = v.to(device)
        else:
            input_dict[k] = v
    if encode:
        result = model.encode(input_dict)
        result.code = result.code.cpu()
        result.code_padding_mask = result.code_padding_mask.cpu()
        for k, v in result.loss_dict.items():
            result.loss_dict[k] = v.cpu()
    else:
        result = model.predict(
            **input_dict,
            fpindex=server_state["fpindex"],
            rxn_matrix=server_state["rxn_matrix"],
        )
        result.token_logits = result.token_logits.cpu()
        result.token_sampled = result.token_sampled.cpu()
        result.reaction_logits = result.reaction_logits.cpu()

    result_data = pickle.dumps(result)
    return Response(content=result_data, media_type="application/octet-stream")


def _start_server(
    model: Synformer,
    fpindex: FingerprintIndex,
    rxn_matrix: ReactantReactionMatrix,
    host: str = "0.0.0.0",
    port: int = 8000,
    device: torch.device = torch.device("cuda"),
):
    logging.basicConfig(level=logging.DEBUG)

    model = model.to(device)
    model.eval()
    server_state["model"] = model
    server_state["fpindex"] = fpindex
    server_state["rxn_matrix"] = rxn_matrix
    server_state["device"] = device

    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level=logging.WARNING)


def download_artifacts(out_dir: Path) -> dict[str, Path]:
    """Downloads the artifacts from GCS to local disk."""
    out_dir.mkdir(parents=True, exist_ok=True)
    name_to_gs_file = {
        "matrix": GS_MATRIX_PKL_FILE,
        "fpindex": GS_FPINDEX_PKL_FILE,
        "model": GS_MODEL_FILE,
    }
    name_to_local_file = {}
    for name, gs_file in name_to_gs_file.items():
        relative_path = gs_file.relative_to(GS_ROOT_DIR)
        local_path = out_dir / relative_path
        if not local_path.exists():
            gs_file.download_to(local_path)
        name_to_local_file[name] = local_path
    return name_to_local_file


def load_artifacts(
    out_dir: Path,
) -> tuple[FingerprintIndex, ReactantReactionMatrix, Synformer]:
    """Loads the artifacts from GCS into memory."""
    name_to_local_file = download_artifacts(out_dir)
    fpindex = pickle.load(open(name_to_local_file["fpindex"], "rb"))
    rxn_matrix = pickle.load(open(name_to_local_file["matrix"], "rb"))

    ckpt = torch.load(name_to_local_file["model"], map_location="cpu")
    config = OmegaConf.create(ckpt["hyper_parameters"]["config"])
    model = Synformer(config.model)
    model.load_state_dict({k[6:]: v for k, v in ckpt["state_dict"].items()})

    return fpindex, rxn_matrix, model


def launch_server_process(
    out_dir: Path = Path("."),
    host: str = "0.0.0.0",
    port: int = 8000,
    device: torch.device = torch.device("cuda"),
) -> tuple[SpawnProcess, FingerprintIndex, ReactantReactionMatrix]:
    name_to_local_file = download_artifacts(out_dir)
    model_file = name_to_local_file["model"]
    fpindex = pickle.load(open(name_to_local_file["fpindex"], "rb"))
    rxn_matrix = pickle.load(open(name_to_local_file["matrix"], "rb"))

    mp_context = mp.get_context("spawn")
    p = mp_context.Process(
        target=_start_server,
        args=(model_file, fpindex, rxn_matrix, host, port, device),
    )
    p.start()
    return p, fpindex, rxn_matrix
