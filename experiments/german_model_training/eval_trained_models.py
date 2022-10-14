"""
Evaluation of trained models on our GPU servers.
This uses the models with available datasets.
"""
import os
import json
from typing import Dict, Set, List
from functools import lru_cache

from tqdm import tqdm
from datasets import load_dataset
from nltk.stem.cistem import Cistem
from rouge_score.rouge_scorer import RougeScorer
from rouge_score.scoring import BootstrapAggregator
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from summaries import Analyzer, Cleaner


@lru_cache(maxsize=4)
def get_dataset(name: str, filtered: bool = False):
    if name == "mlsum":
        data = load_dataset("mlsum", "de")
        ref_col = "text"
        summ_col = "summary"
    elif "klexikon" in name:
        data = load_dataset("dennlinger/klexikon")

        data = {
            "train": [fuse_sentences_but_keep_sample(sample) for sample in data["train"]],
            "validation": [fuse_sentences_but_keep_sample(sample) for sample in data["validation"]],
            "test": [fuse_sentences_but_keep_sample(sample) for sample in data["test"]]
        }

        ref_col = "wiki_text"
        summ_col = "klexikon_text"
    elif "legalsum" in name:
        base_path = "/home/daumiller/LegalSum/"
        train_files = load_split_files(os.path.join(base_path, "train_files.txt"))
        val_files = load_split_files(os.path.join(base_path, "val_files.txt"))
        test_files = load_split_files(os.path.join(base_path, "test_files.txt"))

        train = []
        validation = []
        test = []

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

        ref_col = "reference"
        summ_col = "summary"
    elif "eurlexsum" in name:
        with open("/home/aumiller/german_eurlexsum/german_eurlexsum.json") as f:
            data = json.load(f)
        data = {
            "train": extract_samples(data["train"]),
            "validation": extract_samples(data["validation"]),
            "test": extract_samples(data["test"])
        }
        ref_col = "reference_text"
        summ_col = "summary_text"
    else:
        raise ValueError("Unrecognized dataset name passed!")

    if filtered:
        analyzer = Analyzer(lemmatize=True, lang="de")
        cleaner = Cleaner(analyzer, deduplication_method="test_first",
                          length_metric="char", min_length_summary=20, min_length_reference=50,
                          min_compression_ratio=1.25,
                          extractiveness="fully")
        clean_data = cleaner.clean_dataset(summ_col, ref_col,
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


def extract_samples(split_data: Dict):
    samples = []
    for celex_id, sample in split_data.items():
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


if __name__ == '__main__':
    model_path = "/home/dennis/checkpoint-46895"
    model_name = "German-MultiSumm-base"
    dataset_name = "mlsum"
    reference_column = "text"
    summary_column = "summary"
    filtered = "filtered"

    tokenizer = AutoTokenizer.from_pretrained(model_path, fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    dataset = get_dataset(dataset_name, filtered=True)

    for split in ["validation", "test"]:
        print(f"Computing {filtered} {split} split...")
        samples = dataset[split]

        reference_texts = [sample[reference_column] for sample in samples]
        summary_texts = [sample[summary_column].replace("\n", " ") for sample in samples]

        generated_summaries = []
        print(f"Generating spacy docs for each summary...")
        for reference in tqdm(reference_texts):
            # TODO
            summary = None
            generated_summaries.append(summary)

        with open(f"{dataset_name}_{split}_{filtered}_{model_name}.json", "w") as f:
            json.dump(generated_summaries, f, ensure_ascii=False, indent=2)

        aggregator = get_rouge_scores(summary_texts, generated_summaries)



