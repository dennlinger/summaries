"""
Utility functions used across experiments
"""
from functools import lru_cache
from typing import List, Dict

from tqdm import tqdm
from datasets import load_dataset
from rouge_score.rouge_scorer import RougeScorer
from rouge_score.scoring import BootstrapAggregator
from nltk.stem.cistem import Cistem

from summaries import Analyzer, Cleaner


@lru_cache(maxsize=4)
def get_dataset(name: str, filtered: bool = False):
    if name == "mlsum":
        data = load_dataset("mlsum", "de")

        if filtered:
            reference_column = "text"
            summary_column = "summary"
    else:
        raise ValueError("Unrecognized dataset name passed!")

    if filtered:
        analyzer = Analyzer(lemmatize=True, lang="de")
        cleaner = Cleaner(analyzer, deduplication_method="test_first",
                          length_metric="char", min_length_summary=20, min_length_reference=50,
                          min_compression_ratio=1.25,
                          extractiveness="fully")
        clean_data = cleaner.clean_dataset(summary_column, reference_column,
                                           data["train"], data["validation"], data["test"], enable_tqdm=True)

        return clean_data
    else:
        return data


# The following implementation is borrowed from the Klexikon paper repository (Aumiller and Gertz, 2022):
# https://github.com/dennlinger/klexikon

def get_rouge_scorer_with_cistem(fast=False):
    """
    Replaces the standard Porter stemmer, which works best on English, with the Cistem stemmer, which was specifically
    designed for the German language.
    :return: RougeScorer object with replaced stemmer.
    """
    # Skip LCS computation for 10x speedup during debugging.
    if fast:
        scorer = RougeScorer(["rouge1", "rouge2"], use_stemmer=True)
    else:
        scorer = RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    stemmer = Cistem(case_insensitive=True)  # Insensitive because RougeScorer lowercases anyways.
    scorer._stemmer = stemmer  # Certainly not best practice, but better than re-writing the package ;-)

    return scorer


def get_rouge_scores(gold_summaries: List[str], system_predictions: List[str], fast=False) -> None:
    scorer = get_rouge_scorer_with_cistem(fast=fast)
    aggregator = BootstrapAggregator(confidence_interval=0.95, n_samples=2000)

    print("Computing ROUGE scores...")
    if len(gold_summaries) != len(system_predictions):
        raise ValueError(f"Something went wrong when generating summaries: "
                         f"Found {len(gold_summaries)} samples and "
                         f"{len(system_predictions)} generated texts.")
    for gold, prediction in tqdm(zip(gold_summaries, system_predictions)):
        aggregator.add_scores(scorer.score(gold, prediction))

    result = aggregator.aggregate()
    print_aggregate(result)


def print_aggregate(result: Dict) -> None:
    for key, value_set in result.items():
        print(f"----------------{key} ---------------------")
        print(f"Precision | "
              f"low: {value_set.low.precision * 100:5.2f}, "
              f"mid: {value_set.mid.precision * 100:5.2f}, "
              f"high: {value_set.high.precision * 100:5.2f}")
        print(f"Recall    | "
              f"low: {value_set.low.recall * 100:5.2f}, "
              f"mid: {value_set.mid.recall * 100:5.2f}, "
              f"high: {value_set.high.recall * 100:5.2f}")
        print(f"F1        | "
              f"low: {value_set.low.fmeasure * 100:5.2f}, "
              f"mid: {value_set.mid.fmeasure * 100:5.2f}, "
              f"high: {value_set.high.fmeasure * 100:5.2f}")