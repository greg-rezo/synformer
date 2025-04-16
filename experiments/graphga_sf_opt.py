import logging
import os
import random
import time
from pathlib import Path

import crossover as co
import joblib
import mutate as mu
import numpy as np
import tdc
import yaml
from joblib import delayed
from rdkit import Chem, rdBase
from rdkit.Chem.rdchem import Mol
from tdc.generation import MolGen
from tdc.metadata import single_molecule_dataset_names

from synformer.chem.mol import Molecule
from synformer.models.model_server import SynformerClient, SynformerServer
from synformer.sampler.analog.parallel import run_sampling

rdBase.DisableLog("rdApp.error")

logger = logging.getLogger(__name__)


def sanitize(mol_list):
    new_mol_list = []
    smiles_set = set()
    for mol in mol_list:
        if mol is not None:
            try:
                smiles = Chem.MolToSmiles(mol)
                if smiles is not None and smiles not in smiles_set:
                    smiles_set.add(smiles)
                    new_mol_list.append(mol)
            except ValueError:
                logger.info("bad smiles")
    return new_mol_list


def top_auc(buffer, top_n, finish, freq_log, max_oracle_calls):
    sum = 0
    prev = 0
    called = 0
    ordered_results = list(
        sorted(buffer.items(), key=lambda kv: kv[1][1], reverse=False)
    )
    for idx in range(freq_log, min(len(buffer), max_oracle_calls), freq_log):
        temp_result = ordered_results[:idx]
        temp_result = list(sorted(temp_result, key=lambda kv: kv[1][0], reverse=True))[
            :top_n
        ]
        top_n_now = np.mean([item[1][0] for item in temp_result])
        sum += freq_log * (top_n_now + prev) / 2
        prev = top_n_now
        called = idx
    temp_result = list(sorted(ordered_results, key=lambda kv: kv[1][0], reverse=True))[
        :top_n
    ]
    top_n_now = np.mean([item[1][0] for item in temp_result])
    sum += (len(buffer) - called) * (top_n_now + prev) / 2
    if finish and len(buffer) < max_oracle_calls:
        sum += (max_oracle_calls - len(buffer)) * top_n_now
    return sum / max_oracle_calls


class Oracle:
    def __init__(
        self,
        args=None,
        mol_buffer={},
        max_oracle_calls=10000,
        freq_log=100,
        oracle_name="SA",
        evaluator_name="Diversity",
    ):
        self.task_name = f"{oracle_name}_{evaluator_name}"
        if args is None:
            self.max_oracle_calls = max_oracle_calls
            self.freq_log = freq_log
        else:
            self.args = args
            self.max_oracle_calls = args.max_oracle_calls
            self.freq_log = args.freq_log
        self.mol_buffer = mol_buffer
        self.oracle = tdc.Oracle(name=oracle_name)
        self.evaluator = tdc.Evaluator(name=evaluator_name)
        self.last_log = 0
        self.output_dir = "output"
        os.makedirs(self.output_dir, exist_ok=True)

    @property
    def budget(self):
        return self.max_oracle_calls

    def assign_evaluator(self, evaluator):
        self.evaluator = evaluator

    def sort_buffer(self):
        self.mol_buffer = dict(
            sorted(self.mol_buffer.items(), key=lambda kv: kv[1][0], reverse=True)
        )

    def save_result(self, suffix=None):
        if suffix is None:
            output_file_path = os.path.join(self.output_dir, "results.yaml")
        else:
            output_file_path = os.path.join(
                self.output_dir, "results_" + suffix + ".yaml"
            )

        self.sort_buffer()
        with open(output_file_path, "w") as f:
            yaml.dump(self.mol_buffer, f, sort_keys=False)

    def log_intermediate(
        self,
        mols: list[Mol] | None = None,
        scores: list[float] | None = None,
        finish: bool = False,
    ):
        if finish:
            temp_top100 = list(self.mol_buffer.items())[:100]
            smis = [item[0] for item in temp_top100]
            scores = [item[1][0] for item in temp_top100]
            n_calls = self.max_oracle_calls
        else:
            if mols is None and scores is None:
                if len(self.mol_buffer) <= self.max_oracle_calls:
                    # If not spefcified, log current top-100 mols in buffer
                    temp_top100 = list(self.mol_buffer.items())[:100]
                    smis = [item[0] for item in temp_top100]
                    scores = [item[1][0] for item in temp_top100]
                    n_calls = len(self.mol_buffer)
                else:
                    results = list(
                        sorted(
                            self.mol_buffer.items(),
                            key=lambda kv: kv[1][1],
                            reverse=False,
                        )
                    )[: self.max_oracle_calls]
                    temp_top100 = sorted(
                        results, key=lambda kv: kv[1][0], reverse=True
                    )[:100]
                    smis = [item[0] for item in temp_top100]
                    scores = [item[1][0] for item in temp_top100]
                    n_calls = self.max_oracle_calls
            else:
                # Otherwise, log the input moleucles
                assert mols is not None
                smis = [Chem.MolToSmiles(m) for m in mols]
                n_calls = len(self.mol_buffer)

        # Uncomment this line if want to log top-10 moelucles figures, so as the best_mol key values.
        # temp_top10 = list(self.mol_buffer.items())[:10]

        avg_top1 = np.max(scores)  # type: ignore
        avg_top10 = np.mean(sorted(scores, reverse=True)[:10])  # type: ignore
        avg_top100 = np.mean(scores)  # type: ignore
        avg_sa = np.mean(self.oracle(smis))  # type: ignore
        diversity_top100 = self.evaluator(smis)

        logger.info(
            f"{n_calls}/{self.max_oracle_calls} | "
            f"avg_top1: {avg_top1:.3f} | "
            f"avg_top10: {avg_top10:.3f} | "
            f"avg_top100: {avg_top100:.3f} | "
            f"avg_sa: {avg_sa:.3f} | "
            f"div: {diversity_top100:.3f}"
        )

        # try:
        logger.info(
            {
                "avg_top1": avg_top1,
                "avg_top10": avg_top10,
                "avg_top100": avg_top100,
                "auc_top1": top_auc(
                    self.mol_buffer, 1, finish, self.freq_log, self.max_oracle_calls
                ),
                "auc_top10": top_auc(
                    self.mol_buffer, 10, finish, self.freq_log, self.max_oracle_calls
                ),
                "auc_top100": top_auc(
                    self.mol_buffer, 100, finish, self.freq_log, self.max_oracle_calls
                ),
                "avg_sa": avg_sa,
                "diversity_top100": diversity_top100,
                "n_oracle": n_calls,
            }
        )

    def __len__(self):
        return len(self.mol_buffer)

    def score_smi(self, smi):
        """
        Function to score one molecule

        Argguments:
            smi: One SMILES string represnets a moelcule.

        Return:
            score: a float represents the property of the molecule.
        """
        if len(self.mol_buffer) > self.max_oracle_calls:
            return 0
        if smi is None:
            return 0
        mol = Chem.MolFromSmiles(smi)
        if mol is None or len(smi) == 0:
            return 0
        else:
            smi = Chem.MolToSmiles(mol)
            if smi in self.mol_buffer:
                pass
            else:
                self.mol_buffer[smi] = [
                    float(self.evaluator(smi)),  # type: ignore
                    len(self.mol_buffer) + 1,
                ]
            return self.mol_buffer[smi][0]

    def __call__(self, smiles_lst):
        """
        Score
        """
        if isinstance(smiles_lst, list):
            score_list = []
            for smi in smiles_lst:
                score_list.append(self.score_smi(smi))
                if (
                    len(self.mol_buffer) % self.freq_log == 0
                    and len(self.mol_buffer) > self.last_log
                ):
                    self.sort_buffer()
                    self.log_intermediate()
                    self.last_log = len(self.mol_buffer)
                    self.save_result(suffix=self.task_name)
        else:
            score_list = self.score_smi(smiles_lst)
            if (
                len(self.mol_buffer) % self.freq_log == 0
                and len(self.mol_buffer) > self.last_log
            ):
                self.sort_buffer()
                self.log_intermediate()
                self.last_log = len(self.mol_buffer)
                self.save_result(suffix=self.task_name)
        return score_list

    @property
    def finish(self):
        return len(self.mol_buffer) >= self.max_oracle_calls


MINIMUM = 1e-10


def make_mating_pool(population_mol: list[Mol], population_scores, offspring_size: int):
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
    population_scores = [s + MINIMUM for s in population_scores]
    sum_scores = sum(population_scores)
    population_probs = [p / sum_scores for p in population_scores]
    mating_pool = np.random.choice(
        population_mol, p=population_probs, size=offspring_size, replace=True
    )
    return mating_pool


def reproduce(mating_pool, mutation_rate):
    """
    Args:
        mating_pool: list of RDKit Mol
        mutation_rate: rate of mutation
    Returns:
    """
    parent_a = random.choice(mating_pool)
    parent_b = random.choice(mating_pool)
    try:
        new_child = co.crossover(parent_a, parent_b)
        if new_child is not None:
            new_child = mu.mutate(new_child, mutation_rate)
        return new_child
    except ValueError:
        return parent_a


def projection(smiles_list: list[str], model_client: SynformerClient) -> list[str]:
    t = time.perf_counter()
    input = [Molecule(s) for s in smiles_list]
    result_df = run_sampling(
        mols=input,
        search_width=24,
        exhaustiveness=64,
        time_limit=180,
        sort_by_scores=True,
        model_client=model_client,
    )
    result_df.drop_duplicates(subset="target", inplace=True, keep="first")
    out_smiles_list = result_df.smiles.to_list()

    logger.info(
        f"Projection returned {len(out_smiles_list)} in "
        f"{time.perf_counter() - t:.2f} seconds"
    )
    return out_smiles_list


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", type=str, default="SA")
    parser.add_argument("--evaluator", type=str, default="Diversity")
    parser.add_argument(
        "--model_path",
        type=Path,
        default=Path("data/trained_weights/sf_ed_default.ckpt"),
    )
    parser.add_argument("--max_oracle_calls", type=int, default=10000)
    parser.add_argument("--freq_log", type=int, default=100)
    parser.add_argument("--population_size", type=int, default=100)
    parser.add_argument("--offspring_size", type=int, default=100)
    parser.add_argument("--mutation_rate", type=float, default=0.1)
    parser.add_argument(
        "--mol_dataset", type=str, default="zinc", choices=single_molecule_dataset_names
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    matrix_path = args.model_path.parent.parent / "matrix.pkl"
    fpindex_path = args.model_path.parent.parent / "fpindex.pkl"

    model_server = SynformerServer(
        model_path=args.model_path,
        fpindex_path=fpindex_path,
        rxn_matrix_path=matrix_path,
        device="cuda",  # type: ignore
    )
    model_server.start()

    oracle = Oracle(
        args=args,
        max_oracle_calls=args.max_oracle_calls,
        freq_log=args.freq_log,
        oracle_name=args.oracle,
        evaluator_name=args.evaluator,
    )

    pool = joblib.Parallel(n_jobs=64)

    data = MolGen(name=args.mol_dataset)
    all_smiles = data.get_data().smiles.to_list()  # type: ignore
    starting_population = np.random.choice(all_smiles, args.population_size)

    # select initial population
    # population_smiles = heapq.nlargest(config["population_size"], starting_population, key=oracle)
    population_smiles = starting_population
    population_smiles = projection(
        population_smiles, model_client=model_server.get_client()
    )
    population_mol = [Chem.MolFromSmiles(s) for s in population_smiles]
    population_scores = oracle([Chem.MolToSmiles(mol) for mol in population_mol])

    patience = 0

    while True:
        if len(oracle) > 100:
            oracle.sort_buffer()
            old_score = np.mean(
                [item[1][0] for item in list(oracle.mol_buffer.items())[:100]]
            )
        else:
            old_score = 0

        # new_population
        mating_pool = make_mating_pool(
            population_mol, population_scores, args.population_size
        )
        offspring_mol = list(
            pool(
                delayed(reproduce)(mating_pool, args.mutation_rate)
                for _ in range(args.offspring_size)
            )
        )

        # add new_population
        population_mol += offspring_mol
        population_mol = sanitize(population_mol)
        population_mol = [
            Chem.MolFromSmiles(smi)
            for smi in projection(
                [Chem.MolToSmiles(mol) for mol in population_mol],
                model_client=model_server.get_client(),
            )
        ]

        # stats
        old_scores = population_scores
        population_scores = oracle([Chem.MolToSmiles(mol) for mol in population_mol])
        population_tuples = list(zip(population_scores, population_mol))  # type: ignore
        population_tuples = sorted(population_tuples, key=lambda x: x[0], reverse=True)[
            : args.population_size
        ]
        population_mol = [t[1] for t in population_tuples]
        population_scores = [t[0] for t in population_tuples]

        ### early stopping
        if len(oracle) > 100:
            oracle.sort_buffer()
            new_score = np.mean(
                [item[1][0] for item in list(oracle.mol_buffer.items())[:100]]
            )
            # import ipdb; ipdb.set_trace()
            if (new_score - old_score) < 1e-3:
                patience += 1
                if patience >= 5:
                    oracle.log_intermediate(finish=True)
                    logger.info("convergence criteria met, abort ...... ")
                    break
            else:
                patience = 0

            old_score = new_score

        if oracle.finish:
            break

        oracle.save_result(suffix=args.name)
