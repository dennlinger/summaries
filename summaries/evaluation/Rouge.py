"""
Convenience wrapper around rouge-score library from Google.
Among other things, adds the option to introduce custom stemmers, as well as print functions over the aggregator.
"""

from typing import Optional, List, Dict

from rouge_score.scoring import BootstrapAggregator
from rouge_score.rouge_scorer import RougeScorer
from nltk.stem.cistem import Cistem
from tqdm import tqdm


# The following implementation is borrowed from the Klexikon paper repository (Aumiller and Gertz, 2022):
# https://github.com/dennlinger/klexikon
def get_rouge_scorer(stemmer: Optional[str] = "cistem", fast: bool = False):
    """
    Convenience wrapper around the rouge_score.RougeScorer object.
    Depending on parameter choices, can replace the default stemmer with a language-specific alternative.
    :param fast: Boolean indicating the use of Rouge-L, which takes longer to
    :param stemmer: If specified, will use stemming in the Rouge scorer object; Can also be set to replace the default
        stemmer with a custom (language-specific) version. Available options are: [None, "default", "cistem"]
    :return: RougeScorer object with replaced stemmer.
    """
    # Skip LCS computation for 10x speedup during debugging.

    if stemmer is not None:
        use_stemmer = True
    else:
        use_stemmer = False

    if fast:
        scorer = RougeScorer(["rouge1", "rouge2"], use_stemmer=use_stemmer)
    else:
        scorer = RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=use_stemmer)
    if stemmer == "cistem":
        stemmer = Cistem(case_insensitive=True)  # Insensitive because RougeScorer lowercases anyway.
        scorer._stemmer = stemmer  # Certainly not best practice, but better than re-writing the package ;-)
    elif stemmer is None or stemmer == "default":
        pass
    else:
        raise ValueError("Unrecognized stemmer specified!")

    return scorer


def compute_rouge_scores(gold_summaries: List[str], system_predictions: List[str], scorer: RougeScorer) \
        -> BootstrapAggregator:
    """
    Compute the Rouge scores over a series of summaries with gold references.
    :param gold_summaries: List of gold summary texts.
    :param system_predictions: List of system prediction summaries.
    :param scorer: RougeScorer to compute the actual scores
    :return:
    """
    aggregator = BootstrapAggregator(confidence_interval=0.95, n_samples=2000)

    print("Computing ROUGE scores...")
    if len(gold_summaries) != len(system_predictions):
        raise ValueError(f"Something went wrong when generating summaries: "
                         f"Found {len(gold_summaries)} samples and "
                         f"{len(system_predictions)} generated texts.")
    print("Computing ROUGE scores...")
    for gold, prediction in tqdm(zip(gold_summaries, system_predictions)):
        aggregator.add_scores(scorer.score(gold, prediction))

    result = aggregator.aggregate()
    print_aggregate(result)

    return aggregator


def print_aggregate(result: Dict, latexify: bool = False) -> None:
    """
    Prints the aggregated scores from a BootstrapAggregator object.
    :param result: Dictionary of all relevant scores.
    :param latexify: If enabled, will print the mid values for each Rouge setting in a LaTeX-compatible format.
    :return:
    """
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
        print(f"--------------------------------------------")
        print(f"{key} F1: {value_set.mid.fmeasure * 100:5.2f}")

    # Prints convenience scores for LaTeX tables
    if latexify:
        print(f"${result['rouge1'].mid.fmeasure * 100:5.2f}$ & "
              f"${result['rouge2'].mid.fmeasure * 100:5.2f}$ & "
              f"${result['rougeL'].mid.fmeasure * 100:5.2f}$")

