"""This file is used to build the wheel for the project.

Triggers download of assets from the internet on installation.
"""

import os
import shutil

import requests
from setuptools.build_meta import build_wheel as _orig_build_wheel
from setuptools.build_meta import (
    prepare_metadata_for_build_wheel as _orig_metadata_hook,
)

URLS = {
    "fpindex.pkl": "https://huggingface.co/whgao/synformer/resolve/main/fpindex.pkl?download=true",
    "matrix.pkl": "https://huggingface.co/whgao/synformer/resolve/main/matrix.pkl?download=true",
    "trained_weights/sf_ed_default.ckpt": "https://huggingface.co/whgao/synformer/resolve/main/sf_ed_default.ckpt?download=true",
}


def _download_assets():
    target = os.path.join(os.getcwd(), "src", "synformer", "data")
    os.makedirs(target, exist_ok=True)
    for name, url in URLS.items():
        path = os.path.join(target, name)
        if not os.path.exists(path):
            r = requests.get(url, stream=True)
            r.raise_for_status()
            with open(path, "wb") as f:
                shutil.copyfileobj(r.raw, f)


def prepare_metadata_for_build_wheel(config_settings=None, metadata_directory=""):
    return _orig_metadata_hook(config_settings, metadata_directory)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=""):
    _download_assets()
    return _orig_build_wheel(wheel_directory, config_settings, metadata_directory)
