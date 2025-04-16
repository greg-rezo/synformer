import logging
import multiprocessing as mp
import os
import pickle
import signal
import socket
import struct
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from omegaconf import OmegaConf

from synformer.chem.fpindex import FingerprintIndex
from synformer.chem.matrix import ReactantReactionMatrix
from synformer.models.synformer import Synformer

SERVER_PORT = 8765
logger = logging.getLogger(__name__)


class SynformerClient:
    def __init__(
        self,
        fpindex: FingerprintIndex,
        rxn_matrix: ReactantReactionMatrix,
        host: str = "localhost",
        port: int = SERVER_PORT,
    ):
        self.host = host
        self.port = port
        self.fpindex = fpindex
        self.rxn_matrix = rxn_matrix

    def predict(self, input_dict: dict[str, Any], encode: bool = False):
        """Send prediction request to socket server"""
        # Create socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((self.host, self.port))

        try:
            # Serialize request
            input_dict["encode"] = encode
            data = pickle.dumps(input_dict)

            # Send data length and data
            sock.sendall(struct.pack(">I", len(data)))
            sock.sendall(data)

            # Get response length and data
            msg_len_bytes = sock.recv(4)
            msg_len = struct.unpack(">I", msg_len_bytes)[0]

            data = b""
            while len(data) < msg_len:
                packet = sock.recv(min(msg_len - len(data), 4096))
                if not packet:
                    break
                data += packet

            return pickle.loads(data)
        finally:
            sock.close()


class SynformerServer:
    def __init__(
        self,
        fpindex_path: Path,
        rxn_matrix_path: Path,
        model_path: Path,
        host: str = "localhost",
        port: int = SERVER_PORT,
        device: torch.device = torch.device("cuda"),
    ):
        self.host = host
        self.port = port
        self.server = None
        self.fpindex = pickle.load(open(fpindex_path, "rb"))
        self.rxn_matrix = pickle.load(open(rxn_matrix_path, "rb"))

        ckpt = torch.load(model_path, map_location="cpu")
        config = OmegaConf.create(ckpt["hyper_parameters"]["config"])
        self.model = Synformer(config.model).to(device)
        self.model.load_state_dict({k[6:]: v for k, v in ckpt["state_dict"].items()})

        self.device = device

    def start(self):
        """Start the socket server in a new process"""
        self.server_process = mp.Process(
            target=self._run_server, args=(self.model, self.fpindex, self.rxn_matrix)
        )
        self.server_process.start()
        # Wait briefly to ensure server is running
        time.sleep(1)

    def _run_server(
        self,
        model: Synformer,
        fpindex: FingerprintIndex,
        rxn_matrix: ReactantReactionMatrix,
    ):
        """Load model and serve predictions via socket"""
        # Create socket server
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((self.host, self.port))
        self.server.listen(10)

        while True:
            conn, addr = self.server.accept()
            try:
                # Get message length (4 bytes)
                msg_len_bytes = conn.recv(4)
                if not msg_len_bytes:
                    continue
                msg_len = struct.unpack(">I", msg_len_bytes)[0]

                # Receive the data
                data = b""
                while len(data) < msg_len:
                    packet = conn.recv(min(msg_len - len(data), 4096))
                    if not packet:
                        break
                    data += packet

                if len(data) == msg_len:
                    # Process request
                    request = pickle.loads(data)
                    encode = request.pop("encode")
                    if encode:
                        result = self.model.encode(**request)
                    else:
                        result = self.model.predict(**request)

                    # Send response
                    result_data = pickle.dumps(result)
                    conn.sendall(struct.pack(">I", len(result_data)))
                    conn.sendall(result_data)
            finally:
                conn.close()

    def get_client(self):
        return SynformerClient(self.fpindex, self.rxn_matrix, self.host, self.port)
