import argparse
import logging
import multiprocessing as mp
import random
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import crossover as co
import mutate as mu
import numpy as np
import tdc
import yaml
from pydantic import BaseModel
from rdkit import Chem, rdBase
from rdkit.Chem.rdchem import Mol
from tdc.generation import MolGen
from tdc.metadata import single_molecule_dataset_names

from synformer.chem.fpindex import FingerprintIndex
from synformer.chem.matrix import ReactantReactionMatrix
from synformer.chem.mol import Molecule
from synformer.models.model_server import launch_server_process
from synformer.models.model_server_client import SynformerClient
from synformer.sampler.analog.parallel import run_sampling

GLOBAL_POOL = ProcessPoolExecutor(mp_context=mp.get_context("spawn"))
rdBase.DisableLog("rdApp.error")

logger = logging.getLogger(__name__)


class ScoredMol(BaseModel):
    smiles: str
    score: float | None = None
    _mol: Mol | None = None

    @property
    def mol(self) -> Mol:
        if self._mol is None:
            self._mol = Chem.MolFromSmiles(self.smiles)
        return self._mol  # type: ignore

    def __lt__(self, other: "ScoredMol") -> bool:
        if self.score is None:
            return False
        if other.score is None:
            return True
        return self.score < other.score

    def __hash__(self) -> int:
        return hash(self.smiles)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ScoredMol):
            return False
        return self.smiles == other.smiles


def _sanitize_mol(mol: ScoredMol) -> ScoredMol | None:
    try:
        smiles = Chem.MolToSmiles(mol.mol)
        return ScoredMol(smiles=smiles)
    except ValueError as e:
        logger.info(f"Bad smiles: {e}")
        return None


def sanitize_mols(mols: list[ScoredMol]) -> list[ScoredMol]:
    results = list(GLOBAL_POOL.map(_sanitize_mol, mols))
    return [mol for mol in results if mol is not None]


class Oracle:
    def __init__(self, max_oracle_calls: int = 10000, oracle_name: str = "DRD2"):
        self.oracle_name = oracle_name
        self.max_oracle_calls = max_oracle_calls
        self.mol_buffer: set[ScoredMol] = set()
        self.oracle = tdc.Oracle(name=oracle_name)
        self.sa_scorer = tdc.Oracle(name="SA")
        self.diversity_evaluator = tdc.Evaluator(name="Diversity")
        self.output_dir = Path("output")
        self.log_freq = 10

    def _save_result(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_file_path = self.output_dir / f"results_{self.oracle_name}.yaml"

        mol_data = [mol.model_dump() for mol in self.mol_buffer]
        with open(output_file_path, "w") as f:
            yaml.dump(mol_data, f, sort_keys=False)

    def get_avg_scores(self, top_n: list[int]) -> list[float]:
        mols = sorted(list(self.mol_buffer), reverse=True)
        scores = np.array([mol.score for mol in mols])

        avg_scores = []
        for n in top_n:
            avg_scores.append(scores[:n].mean())
        return avg_scores

    def get_top_smiles(self, top_n: int = 100) -> list[str]:
        mols = sorted(list(self.mol_buffer), reverse=True)
        return [mol.smiles for mol in mols[:top_n]]

    def _log(self):
        smis = self.get_top_smiles(top_n=100)
        n_calls = len(self.mol_buffer)

        avg_top1, avg_top10, avg_top100 = self.get_avg_scores(top_n=[1, 10, 100])

        sa_scores = np.array(self.sa_scorer(smis))
        avg_sa = sa_scores.mean()
        diversity_score = self.diversity_evaluator(smis)

        logger.info(
            f"calls: {n_calls}/{self.max_oracle_calls} | "
            f"top1: {avg_top1:.3f} | "
            f"top10: {avg_top10:.3f} | "
            f"top100: {avg_top100:.3f} | "
            f"sa: {avg_sa:.3f} | "
            f"div: {diversity_score:.3f}"
        )

    def __len__(self):
        return len(self.mol_buffer)

    def _score_mol(self, mol: ScoredMol):
        """
        Function to score one molecule

        Argguments:
            smi: One SMILES string represnets a moelcule.

        Return:
            score: a float represents the property of the molecule.
        """
        assert mol.smiles is not None
        assert len(self.mol_buffer) < self.max_oracle_calls

        if mol not in self.mol_buffer:
            mol.score = float(self.oracle(mol.smiles))  # type: ignore
            self.mol_buffer.add(mol)

    def __call__(self, mols: list[ScoredMol]):
        """Score a list of molecules

        Args:
            smiles_lst: A list of SMILES strings

        Returns:
            A list of scores
        """
        for mol in mols:
            if len(self.mol_buffer) >= self.max_oracle_calls:
                break

            self._score_mol(mol)
            if len(self.mol_buffer) % self.log_freq == 0:
                self._log()
                self._save_result()

    @property
    def finished(self):
        return len(self.mol_buffer) >= self.max_oracle_calls


MINIMUM = 1e-10


def make_mating_pool(mols: list[ScoredMol], size: int):
    """
    Given a population of RDKit Mol and their scores, sample a list of the same size
    with replacement using the population_scores as weights
    Args:
        population_mol: list of RDKit Mol
        population_scores: list of un-normalised scores given by ScoringFunction
        offspring_size: number of molecules to return
    Returns: a list of RDKit Mol (probably not unique)
    """
    # scores -> probs
    scores = []
    for mol in mols:
        assert mol.score is not None
        scores.append(mol.score + MINIMUM)
    sum_scores = sum(scores)
    probs = [p / sum_scores for p in scores]
    return random.choices(mols, weights=probs, k=size)


def reproduce(mating_pool_mols: list[ScoredMol], mutation_rate: float) -> ScoredMol:
    """
    Args:
        mating_pool: list of RDKit Mol
        mutation_rate: rate of mutation
    Returns:
    """
    parent_a_mol = random.choice(mating_pool_mols)
    parent_b_mol = random.choice(mating_pool_mols)
    try:
        new_child_mol = co.crossover(parent_a_mol.mol, parent_b_mol.mol)
        if new_child_mol is not None:
            new_child_mol = mu.mutate(new_child_mol, mutation_rate)
        return ScoredMol(smiles=Chem.MolToSmiles(new_child_mol))
    except ValueError as e:
        logger.debug(f"Crossover failed: {e}")
        return parent_a_mol


def project_to_closest_synthesizable_mols(
    mols: list[ScoredMol],
    fpindex: FingerprintIndex,
    rxn_matrix: ReactantReactionMatrix,
    model_client: SynformerClient,
) -> list[ScoredMol]:
    t = time.perf_counter()
    molecules = [Molecule(mol.smiles) for mol in mols]
    result_df = run_sampling(
        fpindex=fpindex,
        rxn_matrix=rxn_matrix,
        model_client=model_client,
        mols=molecules,
        search_width=24,
        exhaustiveness=64,
        time_limit=180,
        sort_by_scores=True,
        max_results=100,
    )
    result_df = result_df.sort_values(by="score", ascending=False)
    result_df = result_df.drop_duplicates(subset="target", keep="first")
    out_smiles_list = result_df.smiles.to_list()

    logger.info(
        f"Projection returned {len(out_smiles_list)} smiles in "
        f"{time.perf_counter() - t:.2f} seconds"
    )
    return [ScoredMol(smiles=smi, score=None) for smi in out_smiles_list]


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", type=str, default="SA")
    parser.add_argument("--evaluator", type=str, default="Diversity")
    parser.add_argument("--max_oracle_calls", type=int, default=10000)
    parser.add_argument("--population_size", type=int, default=100)
    parser.add_argument("--offspring_size", type=int, default=100)
    parser.add_argument("--mutation_rate", type=float, default=0.1)
    parser.add_argument(
        "--mol_dataset", type=str, default="zinc", choices=single_molecule_dataset_names
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    _, fpindex, rxn_matrix = launch_server_process(
        device="cuda",  # type: ignore
        host="localhost",
        port=8000,
    )
    model_client = SynformerClient(host="localhost", port=8000)

    oracle = Oracle(max_oracle_calls=args.max_oracle_calls, oracle_name=args.oracle)

    logger.info("Loading dataset")
    data = MolGen(name=args.mol_dataset)
    all_smiles = data.get_data().smiles.to_list()  # type: ignore

    # select initial population
    logger.info("Projecting initial population to synthesizable mols")
    population_smiles_list = np.random.choice(all_smiles, args.population_size)
    population_mols = [
        ScoredMol(smiles=smi, score=None) for smi in population_smiles_list
    ]
    population_mols = project_to_closest_synthesizable_mols(
        population_mols,
        fpindex=fpindex,
        rxn_matrix=rxn_matrix,
        model_client=model_client,
    )

    logger.info("Scoring initial population")
    oracle(population_mols)

    prev_avg_score = None
    while True:
        if len(oracle) > 100:
            new_avg_score = oracle.get_avg_scores(top_n=[100])[0]

            if prev_avg_score is not None and (new_avg_score - prev_avg_score) < 1e-3:
                logger.info("Convergence criteria met, exiting")
                break
            prev_avg_score = new_avg_score

            if oracle.finished:
                logger.info("Used all oracle calls, existing")
                break

        # new_population
        logger.info("Making mating pool")
        mating_pool_mols = make_mating_pool(population_mols, args.population_size)

        logger.info("Making offspring")
        offspring_mols = [
            reproduce(mating_pool_mols, args.mutation_rate)
            for _ in range(args.offspring_size)
        ]

        logger.info("Projecting offspring to synthesizable mols")
        offspring_mols = sanitize_mols(offspring_mols)
        offspring_mols = project_to_closest_synthesizable_mols(
            offspring_mols,
            fpindex=fpindex,
            rxn_matrix=rxn_matrix,
            model_client=model_client,
        )

        # stats
        logger.info("Scoring offspring")
        oracle(offspring_mols)

        oracle._save_result()
