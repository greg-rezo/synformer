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

SERVER_PORT = 8766
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
            logger.debug(f"Client sending data length: {len(data)}")
            sock.sendall(struct.pack(">I", len(data)))
            sock.sendall(data)

            # Get response length and data
            msg_len_bytes = sock.recv(4)
            logger.debug(f"Client received message: {msg_len_bytes}")
            msg_len = struct.unpack(">I", msg_len_bytes)[0]
            logger.debug(f"Client received message length: {msg_len}")
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
        mp_context = mp.get_context("spawn")
        self.server_process = mp_context.Process(target=self._run_server)
        self.server_process.start()
        # Wait briefly to ensure server is running
        time.sleep(1)

    def _run_server(self):
        """Load model and serve predictions via socket"""
        logging.basicConfig(level=logging.DEBUG)
        logger.debug(f"Server starting on {self.host}:{self.port}")

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
                logger.debug(f"Server received message: {msg_len_bytes}")
                if not msg_len_bytes:
                    continue
                msg_len = struct.unpack(">I", msg_len_bytes)[0]
                logger.debug(f"Server received message length: {msg_len}")

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
                    item_dict = {
                        k: v.to(self.device)
                        for k, v in request.items()
                        if isinstance(v, torch.Tensor)
                    }
                    if encode:
                        result = self.model.encode(item_dict)  # type: ignore
                    else:
                        result = self.model.predict(
                            **item_dict,  # type: ignore
                            fpindex=self.fpindex,
                            rxn_matrix=self.rxn_matrix,
                        )

                    # Send response
                    result_data = pickle.dumps(result)
                    logger.debug(f"Server sending data length: {len(result_data)}")
                    conn.sendall(struct.pack(">I", len(result_data)))
                    conn.sendall(result_data)
            finally:
                conn.close()

    def get_client(self):
        return SynformerClient(self.fpindex, self.rxn_matrix, self.host, self.port)
