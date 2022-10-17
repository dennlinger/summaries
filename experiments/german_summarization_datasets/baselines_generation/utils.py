"""
Utility functions used across experiments
"""
from typing import List, Dict, Set
import json
import os

from tqdm import tqdm
from datasets import load_dataset
from rouge_score.rouge_scorer import RougeScorer
from rouge_score.scoring import BootstrapAggregator
from nltk.stem.cistem import Cistem

from summaries import Analyzer, Cleaner


def get_dataset(name: str, filtered: bool = False):
    """
    Loads one of the German summarization datasets, with optional filtering.
    As a distinction from other loading functions in this library, this is only able to load those datasets
    that have a dedicated validation/test set ("mlsum", "klexikon", "eurlexsum", "legalsum").
    Contains hard-coded paths for datasets that are only available offline.
    Download links can be found in the respective comments for datasets.
    :param name: Name of the dataset. Has to be either one of ("mlsum", "klexikon", "eurlexsum", "legalsum").
    :param filtered: Whether the dataset should be filtered. Currently, does not allow for any further specification.
    :return: The samples of a filtered dataset.
    """
    if name == "mlsum":
        data = load_dataset("mlsum", "de")
        # Names of the relevant columns have to be given to work with the filtering method.
        reference_column = "text"
        summary_column = "summary"
    elif "klexikon" in name:
        data = load_dataset("dennlinger/klexikon")

        data = {
            "train": [fuse_sentences_but_keep_sample(sample) for sample in data["train"]],
            "validation": [fuse_sentences_but_keep_sample(sample) for sample in data["validation"]],
            "test": [fuse_sentences_but_keep_sample(sample) for sample in data["test"]]
        }

        reference_column = "wiki_text"
        summary_column = "klexikon_text"
    elif "legalsum" in name:
        # Download link for data can be found here: https://github.com/sebimo/LegalSum
        base_path = "/home/daumiller/LegalSum/"
        train_files = load_split_files(os.path.join(base_path, "train_files.txt"))
        val_files = load_split_files(os.path.join(base_path, "val_files.txt"))
        test_files = load_split_files(os.path.join(base_path, "test_files.txt"))

        train = []
        validation = []
        test = []

        # Iterate through files and assign them to the respective splits.
        # The file containing the assignments is provided by the original dataset authors.
        for fn in tqdm(os.listdir(os.path.join(base_path, "data/"))):
            fp = os.path.join(base_path, "data/", fn)
            with open(fp) as f:
                data = json.load(f)
            sample = construct_sample_from_data(data, fn)

            # Assign the sample to the correct dataset.
            if fn in train_files:
                train.append(sample)
            elif fn in val_files:
                validation.append(sample)
            elif fn in test_files:
                test.append(sample)
            else:
                continue

        data = {
            "train": train,
            "validation": validation,
            "test": test
        }

        reference_column = "reference"
        summary_column = "summary"
    elif "eurlexsum" in name:
        # Download link will be added shortly
        with open("/home/daumiller/german_eurlexsum/german_eurlexsum.json") as f:
            data = json.load(f)
        data = {
            "train": extract_samples(data["train"]),
            "validation": extract_samples(data["validation"]),
            "test": extract_samples(data["test"])
        }
        reference_column = "reference_text"
        summary_column = "summary_text"
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

        print(f"Loading {[len(v) for _, v in clean_data.items()]} samples")
        return clean_data
    else:
        print(f"Loading {[len(v) for _, v in data.items()]} samples")
        return data


def fuse_sentences_but_keep_sample(sample):
    sample["wiki_text"] = "\n".join([line.strip("\n= ") for line in sample["wiki_sentences"] if line.strip("\n= ") != ""])
    sample["klexikon_text"] = "\n".join([line.strip("\n= ") for line in sample["klexikon_sentences"] if line.strip("\n= ") != ""])

    return sample


def extract_samples(split):
    samples = []
    for celex_id, sample in split.items():
        samples.append(sample)

    return samples


def load_split_files(fp: str) -> Set:
    with open(fp) as f:
        files = f.readlines()
    return set([fn.strip("\n ") for fn in files])


def construct_sample_from_data(data: Dict, fn: str) -> Dict:
    # Sentence-based storage requires unfolding
    facts = "\n".join([line.strip("\n ") for line in data["facts"]])
    reasoning = "\n".join([line.strip("\n ") for line in data["reasoning"]])
    # Slightly more complex for summary text, given that we have no idea where it's coming from.
    guiding_principle = ""
    for sub_text in data["guiding_principle"]:
        for text in sub_text:
            clean_text = text.strip("\n ")
            guiding_principle += f"{clean_text}\n"

    reference = f"{facts}\n{reasoning}"
    reference = reference.replace("\xa0", " ")
    guiding_principle = guiding_principle.replace("\xa0", " ")
    sample = {
        "id": data["id"],
        "date": data["date"],
        "court": data["court"],
        "file_name": fn,
        "reference": reference,
        "summary": guiding_principle
    }
    return sample


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


def get_rouge_scores(gold_summaries: List[str], system_predictions: List[str], fast=False) -> BootstrapAggregator:
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
    print_aggregate(result, fast)

    return aggregator


def print_aggregate(result: Dict, fast: bool = False) -> None:
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

    # Necessary to avoid KeyError for RougeL
    if not fast:
        print(f"${result['rouge1'].mid.fmeasure * 100:5.2f}$ & "
              f"${result['rouge2'].mid.fmeasure * 100:5.2f}$ & "
              f"${result['rougeL'].mid.fmeasure * 100:5.2f}$")
