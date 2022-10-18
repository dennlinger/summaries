"""
Evaluation of trained models on our GPU servers.
This uses the models with available datasets.
"""
import os
import json
from typing import Dict, Set
from functools import lru_cache
from argparse import Namespace, ArgumentParser

from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

from summaries import Analyzer, Cleaner
from summaries.evaluation import get_rouge_scorer, compute_rouge_scores


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


def batch_generator(iterable, batch_size):
    current_batch = []
    for item in iterable:
        current_batch.append(item)
        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = []
    if current_batch:
        yield current_batch


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("model_path", type=str,
                        help="Path to, or name of, a model checkpoint.")
    parser.add_argument("--output-name", type=str, default="German-MultiSumm",
                        help="Name to use in the output file for the model name.")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size to use during evaluation.")

    return parser.parse_args()


if __name__ == '__main__':
    dataset_name = "mlsum"
    reference_column = "text"
    summary_column = "summary"
    filtered = "filtered"

    args = get_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
    pipe = pipeline("summarization", model=model, tokenizer=tokenizer)# , device=0)
    dataset = get_dataset(dataset_name, filtered=True)

    for split in ["validation", "test"]:
        print(f"Computing {filtered} {split} split...")
        samples = dataset[split]

        prompt = "Zusammenfassung News:"
        reference_texts = [f"{prompt} {sample[reference_column]}" for sample in samples]
        summary_texts = [sample[summary_column].replace("\n", " ") for sample in samples]

        generated_summaries = []
        # TODO: Fix this stupid batching method, and implement it myself.
        # FIXME: Also verify that input texts are actually trimmed to 768 tokens. Based on memory consumption,
        #  it does not seem to be the case.

        for batch in tqdm(batch_generator(reference_texts, batch_size=args.batch_size)):
            summaries = pipe(batch, max_length=256, batch_size=args.batch_size, truncation=True)
            summaries = [generated_sample["summary_text"] for generated_sample in summaries]
            generated_summaries.extend(summaries)

        with open(f"{dataset_name}_{split}_{filtered}_{args.model_name}.json", "w") as f:
            json.dump(generated_summaries, f, ensure_ascii=False, indent=2)

        # Compute Rouge scores
        scorer = get_rouge_scorer(stemmer="cistem")
        aggregator = compute_rouge_scores(summary_texts, generated_summaries, scorer)



