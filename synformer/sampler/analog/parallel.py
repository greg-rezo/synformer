import io
import logging
import multiprocessing as mp
import os
import pathlib
import pickle
import subprocess
import time
import warnings
from multiprocessing import synchronize as sync
from typing import TypeAlias

import pandas as pd
import ray
import torch
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from synformer.chem.fpindex import FingerprintIndex
from synformer.chem.matrix import ReactantReactionMatrix
from synformer.chem.mol import FingerprintOption, Molecule

# from synformer.models.model_interface import SynformerInterface
from synformer.models.synformer import Synformer
from synformer.sampler.analog.state_pool import StatePool, TimeLimit

TaskQueueType: TypeAlias = "mp.JoinableQueue[Molecule | None]"
ResultQueueType: TypeAlias = "mp.Queue[tuple[Molecule, pd.DataFrame]]"


logger = logging.getLogger(__name__)


# class Worker(mp.Process):
#     def __init__(
#         self,
#         model_path: pathlib.Path,
#         task_queue: TaskQueueType,
#         result_queue: ResultQueueType,
#         gpu_id: str,
#         gpu_lock: sync.Lock,
#         state_pool_opt: dict | None = None,
#         max_evolve_steps: int = 12,
#         max_results: int = 100,
#         time_limit: int = 120,
#         server=None,
#     ):
#         super().__init__()
#         self._model_path = model_path
#         self._task_queue = task_queue
#         self._result_queue = result_queue
#         self._gpu_id = gpu_id
#         self._gpu_lock = gpu_lock
#         self._server = server

#         self._state_pool_opt = state_pool_opt or {}
#         self._max_evolve_steps = max_evolve_steps
#         self._max_results = max_results
#         self._time_limit = time_limit

#     def run(self) -> None:
#         os.sched_setaffinity(0, range(os.cpu_count() or 1))
#         os.environ["CUDA_VISIBLE_DEVICES"] = self._gpu_id

#         if self._server is None:
#             ckpt = torch.load(self._model_path, map_location="cpu")
#             config = OmegaConf.create(ckpt["hyper_parameters"]["config"])
#             model = Synformer(config.model).to("cuda")
#             model.load_state_dict({k[6:]: v for k, v in ckpt["state_dict"].items()})
#             model.eval()
#             self._model = model

#             self._fpindex: FingerprintIndex = pickle.load(
#                 open(config.chem.fpindex, "rb")
#             )
#             self._rxn_matrix: ReactantReactionMatrix = pickle.load(
#                 open(config.chem.rxn_matrix, "rb")
#             )
#         else:
#             ckpt = torch.load(self._model_path, map_location="cpu")
#             config = OmegaConf.create(ckpt["hyper_parameters"]["config"])
#             model = Synformer(config.model).eval()
#             self._model = model

#             self._fpindex: FingerprintIndex = pickle.load(
#                 open(config.chem.fpindex, "rb")
#             )
#             self._rxn_matrix: ReactantReactionMatrix = pickle.load(
#                 open(config.chem.rxn_matrix, "rb")
#             )

#         try:
#             while True:
#                 next_task = self._task_queue.get()
#                 if next_task is None:
#                     self._task_queue.task_done()
#                     break
#                 result_df = self.process(next_task)
#                 self._task_queue.task_done()
#                 self._result_queue.put((next_task, result_df))
#                 if len(result_df) == 0:
#                     logger.info(f"{self.name}: No results for {next_task.smiles}")
#                 else:
#                     max_sim = result_df["score"].max()
#                     logger.info(
#                         f"{self.name}: max_sim={max_sim:.3f} smiles={next_task.smiles}"
#                     )
#         except KeyboardInterrupt:
#             logger.warning(f"{self.name}: Exiting due to KeyboardInterrupt")
#             return

#     def process(self, mol: Molecule):
#         sampler = StatePool(
#             mol=mol,
#             **self._state_pool_opt,
#         )
#         tl = TimeLimit(self._time_limit)
#         for _ in range(self._max_evolve_steps):
#             sampler.evolve(show_pbar=False, time_limit=tl)
#             max_sim = max(
#                 [
#                     p.molecule.sim(
#                         mol, FingerprintOption.morgan_for_tanimoto_similarity()
#                     )
#                     for p in sampler.get_products()
#                 ]
#                 or [-1]
#             )
#             if max_sim == 1.0:
#                 break

#         df = sampler.get_dataframe()[: self._max_results]
#         return df


# class WorkerPool:
#     def __init__(
#         self,
#         gpu_ids: list[int | str],
#         num_workers_per_gpu: int,
#         task_qsize: int,
#         result_qsize: int,
#         **worker_opt,
#     ) -> None:
#         super().__init__()
#         self._task_queue: TaskQueueType = mp.JoinableQueue(task_qsize)
#         self._result_queue: ResultQueueType = mp.Queue(result_qsize)
#         self._gpu_ids = [str(d) for d in gpu_ids]
#         self._gpu_locks = [mp.Lock() for _ in gpu_ids]
#         num_gpus = len(gpu_ids)
#         num_workers = num_workers_per_gpu * num_gpus
#         self._workers = [
#             Worker(
#                 task_queue=self._task_queue,
#                 result_queue=self._result_queue,
#                 gpu_id=self._gpu_ids[i % num_gpus],
#                 gpu_lock=self._gpu_locks[i % num_gpus],
#                 **worker_opt,
#             )
#             for i in range(num_workers)
#         ]

#         for w in self._workers:
#             w.start()

#     def submit(self, task: Molecule, block: bool = True, timeout: float | None = None):
#         self._task_queue.put(task, block=block, timeout=timeout)

#     def fetch(self, block: bool = True, timeout: float | None = None):
#         return self._result_queue.get(block=block, timeout=timeout)

#     def kill(self):
#         for w in self._workers:
#             w.kill()
#         self._result_queue.close()
#         self._task_queue.close()

#     def end(self):
#         for _ in self._workers:
#             self._task_queue.put(None)
#         self._task_queue.join()
#         for w in tqdm(self._workers, desc="Terminating"):
#             w.terminate()
#         self._result_queue.close()
#         self._task_queue.close()


# class WorkerNoStop(mp.Process):
#     def __init__(
#         self,
#         task_queue: TaskQueueType,
#         result_queue: ResultQueueType,
#         model_interface: SynformerInterface,
#         state_pool_opt: dict | None = None,
#         max_evolve_steps: int = 12,
#         max_results: int = 100,
#         time_limit: int = 120,
#     ):
#         super().__init__()
#         self._model_interface = model_interface
#         self._task_queue = task_queue
#         self._result_queue = result_queue

#         self._state_pool_opt = state_pool_opt or {}
#         self._max_evolve_steps = max_evolve_steps
#         self._max_results = max_results
#         self._time_limit = time_limit

#     def run(self) -> None:
#         try:
#             while True:
#                 next_task = self._task_queue.get()
#                 if next_task is None:
#                     self._task_queue.task_done()
#                     break
#                 result_df = self.process(next_task)
#                 self._task_queue.task_done()
#                 self._result_queue.put((next_task, result_df))
#                 if len(result_df) == 0:
#                     logger.info(f"{self.name}: No results for {next_task.smiles}")
#                 else:
#                     max_sim = result_df["score"].max()
#                     logger.info(
#                         f"{self.name}: max_sim={max_sim:.3f} smiles={next_task.smiles}"
#                     )
#         except KeyboardInterrupt:
#             logger.warning(f"{self.name}: Exiting due to KeyboardInterrupt")
#             return

#     def process(self, mol: Molecule):
#         sampler = StatePool(
#             mol=mol,
#             model_interface=self._model_interface,
#             **self._state_pool_opt,
#         )
#         tl = TimeLimit(self._time_limit)
#         for _ in range(self._max_evolve_steps):
#             sampler.evolve(show_pbar=False, time_limit=tl)

#         df = sampler.get_dataframe()[: self._max_results]
#         return df


# class WorkerPoolNoStop:
#     def __init__(
#         self,
#         model_interface: SynformerInterface,
#         num_workers_per_gpu: int,
#         task_qsize: int,
#         result_qsize: int,
#         **worker_opt,
#     ) -> None:
#         super().__init__()
#         self._task_queue: TaskQueueType = mp.JoinableQueue(task_qsize)
#         self._result_queue: ResultQueueType = mp.Queue(result_qsize)
#         self._model_interface = model_interface
#         num_workers = num_workers_per_gpu
#         self._workers = [
#             WorkerNoStop(
#                 task_queue=self._task_queue,
#                 result_queue=self._result_queue,
#                 model_interface=self._model_interface,
#                 **worker_opt,
#             )
#             for i in range(num_workers)
#         ]

#         for w in self._workers:
#             w.start()

#     def submit(self, task: Molecule, block: bool = True, timeout: float | None = None):
#         self._task_queue.put(task, block=block, timeout=timeout)

#     def fetch(self, block: bool = True, timeout: float | None = None):
#         return self._result_queue.get(block=block, timeout=timeout)

#     def kill(self):
#         for w in self._workers:
#             w.kill()
#         self._result_queue.close()
#         self._task_queue.close()

#     def end(self):
#         for _ in self._workers:
#             self._task_queue.put(None)
#         self._task_queue.join()
#         for w in tqdm(self._workers, desc="Terminating"):
#             w.terminate()
#         self._result_queue.close()
#         self._task_queue.close()


# def _count_gpus():
#     return int(
#         subprocess.check_output(
#             "nvidia-smi --query-gpu=name --format=csv,noheader | wc -l",
#             shell=True,
#             text=True,
#         ).strip()
#     )


# def run_parallel_sampling(
#     input: list[Molecule],
#     output: pathlib.Path,
#     model_path: pathlib.Path,
#     search_width: int = 24,
#     exhaustiveness: int = 64,
#     num_gpus: int = -1,
#     num_workers_per_gpu: int = 2,
#     task_qsize: int = 0,
#     result_qsize: int = 0,
#     time_limit: int = 180,
#     sort_by_scores: bool = True,
# ) -> None:
#     num_gpus = num_gpus if num_gpus > 0 else _count_gpus()
#     pool = WorkerPool(
#         gpu_ids=list(range(num_gpus)),
#         num_workers_per_gpu=num_workers_per_gpu,
#         task_qsize=task_qsize,
#         result_qsize=result_qsize,
#         model_path=model_path,
#         state_pool_opt={
#             "factor": search_width,
#             "max_active_states": exhaustiveness,
#             "sort_by_score": sort_by_scores,
#         },
#         time_limit=time_limit,
#     )
#     output.parent.mkdir(parents=True, exist_ok=True)

#     total = len(input)
#     for mol in input:
#         pool.submit(mol)

#     df_all: list[pd.DataFrame] = []
#     with open(output, "w") as f:
#         for _ in tqdm(range(total), desc="Sampling"):
#             _, df = pool.fetch()
#             if len(df) == 0:
#                 continue
#             df.to_csv(f, float_format="%.3f", index=False, header=f.tell() == 0)
#             df_all.append(df)

#     df_merge = pd.concat(df_all, ignore_index=True)
#     logger.info(
#         df_merge.loc[df_merge.groupby("target").idxmax()["score"]]
#         .select_dtypes(include="number")
#         .sum()
#         / total
#     )

#     count_success = len(df_merge["target"].unique())
#     logger.info(f"Success rate: {count_success}/{total} = {count_success / total:.3f}")

#     recons_targets: set[str] = set()
#     for _, row in df_merge.iterrows():
#         if row["score"] == 1.0:
#             mol_target = Molecule(row["target"])
#             mol_recons = Molecule(row["smiles"])
#             if mol_recons.csmiles == mol_target.csmiles:
#                 recons_targets.add(row["target"])
#     count_recons = len(recons_targets)
#     logger.info(
#         f"Reconstruction rate: {count_recons}/{total} = {count_recons / total:.3f}"
#     )

#     pool.end()


# def run_parallel_sampling_return_smiles(
#     input: list[Molecule],
#     model_path: pathlib.Path,
#     search_width: int = 24,
#     exhaustiveness: int = 64,
#     num_gpus: int = -1,
#     num_workers_per_gpu: int = 2,
#     task_qsize: int = 0,
#     result_qsize: int = 0,
#     time_limit: int = 180,
#     sort_by_scores: bool = True,
# ) -> None:
#     num_gpus = num_gpus if num_gpus > 0 else _count_gpus()
#     logger.info(f"Running parallel sampling with {num_gpus} GPUs")
#     pool = WorkerPool(
#         gpu_ids=list(range(num_gpus)),
#         num_workers_per_gpu=num_workers_per_gpu,
#         task_qsize=task_qsize,
#         result_qsize=result_qsize,
#         model_path=model_path,
#         state_pool_opt={
#             "factor": search_width,
#             "max_active_states": exhaustiveness,
#             "sort_by_score": sort_by_scores,
#         },
#         time_limit=time_limit,
#     )

#     total = len(input)
#     for mol in input:
#         pool.submit(mol)

#     df_all: list[pd.DataFrame] = []

#     for _ in tqdm(range(total), desc="Sampling"):
#         _, df = pool.fetch()
#         if len(df) == 0:
#             continue
#         df_all.append(df)

#     df_merge = pd.concat(df_all, ignore_index=True)
#     pool.end()

#     return df_merge  # type: ignore


# def run_parallel_sampling_return_smiles_no_early_stop(
#     input: list[Molecule],
#     model_interface: SynformerInterface,
#     search_width: int = 24,
#     exhaustiveness: int = 64,
#     num_gpus: int = -1,
#     num_workers_per_gpu: int = 2,
#     task_qsize: int = 0,
#     result_qsize: int = 0,
#     time_limit: int = 180,
#     sort_by_scores: bool = True,
# ) -> None:
#     num_gpus = num_gpus if num_gpus > 0 else _count_gpus()
#     pool = WorkerPoolNoStop(
#         model_interface=model_interface,
#         num_workers_per_gpu=num_workers_per_gpu,
#         task_qsize=task_qsize,
#         result_qsize=result_qsize,
#         state_pool_opt={
#             "factor": search_width,
#             "max_active_states": exhaustiveness,
#             "sort_by_score": sort_by_scores,
#         },
#         time_limit=time_limit,
#     )

#     total = len(input)
#     for mol in input:
#         pool.submit(mol)

#     df_all: list[pd.DataFrame] = []

#     for _ in tqdm(range(total), desc="Sampling"):
#         _, df = pool.fetch()
#         if len(df) == 0:
#             continue
#         df_all.append(df)

#     df_merge = pd.concat(df_all, ignore_index=True)
#     pool.end()

#     return df_merge  # type: ignore


@ray.remote(num_gpus=0.15)
def _run_sampling_molecule(
    mol: Molecule,
    state_pool_opt: dict,
    time_limit: int,
    max_evolve_steps: int,
    max_results: int,
    fpindex: FingerprintIndex,
    rxn_matrix: ReactantReactionMatrix,
    model: Synformer,
) -> bytes:
    logging.basicConfig(level=logging.INFO)
    warnings.filterwarnings("ignore")

    try:
        sampler = StatePool(
            mol=mol,
            **state_pool_opt,
            model=model,
            fpindex=fpindex,
            rxn_matrix=rxn_matrix,
        )
        tl = TimeLimit(time_limit)
        for _ in range(max_evolve_steps):
            sampler.evolve(show_pbar=False, time_limit=tl)
            max_sim = max(
                [
                    p.molecule.sim(
                        mol, FingerprintOption.morgan_for_tanimoto_similarity()
                    )
                    for p in sampler.get_products()
                ]
                or [-1]
            )
            if max_sim == 1.0:
                break

        df = sampler.get_dataframe()
        logger.debug(f"Columns: {df.columns}")

        if len(df) > 0:
            df = df.drop_duplicates(subset="smiles")
            df = df.sort_values(by="score", ascending=False)
            df = df.iloc[:max_results]
        return df.to_parquet()
    except KeyboardInterrupt:
        return pd.DataFrame().to_parquet()


def to_iterator(obj_ids):
    while obj_ids:
        done, obj_ids = ray.wait(obj_ids)
        yield ray.get(done[0])


def run_sampling(
    mols: list[Molecule],
    fpindex: FingerprintIndex,
    rxn_matrix: ReactantReactionMatrix,
    model: Synformer,
    search_width: int = 24,
    exhaustiveness: int = 64,
    time_limit: int = 180,
    max_results: int = 100,
    max_evolve_steps: int = 12,
    sort_by_scores: bool = True,
) -> pd.DataFrame:
    """Sample analogs for a list of molecules.

    Runs in parallel.

    Args:
        mols: A list of molecules to sample analogs for.
        fpindex: The fingerprint index to use for sampling.
        rxn_matrix: The reaction matrix to use for sampling.
        model: The model to use for sampling.
        search_width: Beam search top-k.
        exhaustiveness: Beam search width.
        time_limit: Time limit for each sampling run.
        max_results: Maximum number of results to return.
        max_evolve_steps: Maximum number of evolution steps.
        sort_by_scores: Whether to sort the results by scores.
    """
    # logger.info(f"Running sampling for {len(mols)} molecules")

    state_pool_opt = {
        "factor": search_width,
        "max_active_states": exhaustiveness,
        "sort_by_score": sort_by_scores,
    }

    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init(num_gpus=1)

    # Put shared objects into Ray's object store
    fpindex_ref = ray.put(fpindex)
    rxn_matrix_ref = ray.put(rxn_matrix)
    model_ref = ray.put(model.to("cuda"))

    # Launch remote tasks for each molecule using the Ray remote function
    futures = [
        _run_sampling_molecule.remote(
            mol,
            state_pool_opt,
            time_limit,
            max_evolve_steps,
            max_results,
            fpindex_ref,
            rxn_matrix_ref,
            model_ref,
        )
        for mol in mols
    ]

    # Retrieve results from Ray
    bytes_list = tqdm(
        to_iterator(futures), desc="Retrieving results", total=len(futures)
    )

    dfs = [pd.read_parquet(io.BytesIO(b)) for b in bytes_list]
    return pd.concat(dfs, ignore_index=True)


def project_to_synthesizable(
    smiles_list: list[str],
    fpindex: FingerprintIndex,
    rxn_matrix: ReactantReactionMatrix,
    model: Synformer,
    search_width: int = 24,
    exhaustiveness: int = 64,
    time_limit: int = 180,
    max_results: int = 100,
) -> dict[str, tuple[str, float]]:
    t = time.perf_counter()
    molecules = [Molecule(smi) for smi in smiles_list]
    result_df = run_sampling(
        fpindex=fpindex,
        rxn_matrix=rxn_matrix,
        model=model,
        mols=molecules,
        search_width=search_width,
        exhaustiveness=exhaustiveness,
        time_limit=time_limit,
        sort_by_scores=True,
        max_results=max_results,
    )
    result_df = result_df.sort_values(by="score", ascending=False)
    result_df = result_df.drop_duplicates(subset="target", keep="first")

    logger.info(
        f"Projection returned {len(result_df)} smiles in "
        f"{time.perf_counter() - t:.2f} seconds"
    )
    orig_to_new = {
        row["target"]: (row["smiles"], row["score"]) for _, row in result_df.iterrows()
    }
    return orig_to_new
