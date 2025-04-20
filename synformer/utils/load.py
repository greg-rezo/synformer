"""Load files from GCS."""

import pickle
from pathlib import Path

from cloudpathlib import GSPath

from synformer.chem.fpindex import FingerprintIndex
from synformer.chem.matrix import ReactantReactionMatrix
from synformer.models.synformer import Synformer

GS_ROOT_DIR = GSPath("gs://rezo-artifacts/synformer/data")
GS_MATRIX_PKL_FILE = GS_ROOT_DIR / "matrix.pkl"
GS_FPINDEX_PKL_FILE = GS_ROOT_DIR / "fpindex.pkl"
GS_MODEL_FILE = GS_ROOT_DIR / "trained_weights/sf_ed_default.ckpt"

def _download_resource_file(out_dir: Path, gs_file: GSPath) -> Path:
    """Loads the model from GCS into memory."""
    relative_file = gs_file.relative_to(GS_ROOT_DIR)
    local_file = out_dir / relative_file
    local_file.parent.mkdir(parents=True, exist_ok=True)
    if not local_file.exists():
        gs_file.download_to(local_file)
    return local_file


def load_model(out_dir: Path) -> Synformer:
    """Loads the model from GCS into memory."""
    return Synformer.load_from_checkpoint(
        _download_resource_file(out_dir, GS_MODEL_FILE)
    )


def load_fpindex(out_dir: Path) -> FingerprintIndex:
    """Loads the fingerprint index from GCS into memory."""
    return pickle.load(
        open(
            _download_resource_file(out_dir, GS_FPINDEX_PKL_FILE),
            "rb",
        )
    )


def load_rxn_matrix(out_dir: Path) -> ReactantReactionMatrix:
    """Loads the reaction matrix from GCS into memory."""
    return pickle.load(
        open(
            _download_resource_file(out_dir, GS_MATRIX_PKL_FILE),
            "rb"
        )
    )
