"""
This introduces a torch-compatible dataset class that combines the samples from all the available German models.
Currently, we consider the following:
- MLSUM (de)
- MassiveSumm (de)
- WikiLingua (de)
- Klexikon
- Swisstext/GeWiki
- LegalSum
- EUR-Lex-Sum (Aumiller et al.'s version)

For the respective datasets, we also add prompt to differentiate between the different styles to encourage the model
to learn different summarization patterns across data. We use the following patterns:

- Prompt: "Zusammenfassung News:"; Used sources: MLSUM + Mixin MassiveSumm
    Due to the data quality (and bigger size) of MassiveSumm, we use aggressive filtering and match
     the number of available MLSUM samples with samples from MassiveSumm.
- Prompt: "Zusammenfassung Instruktionen:"; Used sources: WikiLingua
- Prompt: "Zusammenfassung Wikipedia:"; Used sources: Swisstext/GeWiki + Mixin of Klexikon.
    Due to the size of Klexikon, we use all available samples.
- Prompt: "Vereinfachte Zusammenfassung Wikipedia:"; Used sources: Klexikon
- Prompt: "Zusammenfassung Gerichtsentscheid:"; Used sources: LegalSum
- Prompt: "Zusammenfassung Legislatur:"; Used sources: EUR-Lex-Sum

We further use additional samples from each dataset to bake in respective behavior when querying without prompts.
We randomly sample at most 2000 samples per dataset to balance the exposure to different domains.

For validation sets, we similarly sample at most 1000 samples to keep sizes realistic.
"""

import os
import json
import regex
from typing import Set, List, Dict, Tuple

from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import numpy as np

from summaries import Analyzer, Cleaner


def get_dataset(name: str, base_path: str) -> Dict:
    """
    Supports MLSUM, MassiveSumm, Klexikon, LegalSum, WikiLingua, Swisstext, EUR-Lex-Sum
    :param name: Name of the dataset.
    :param base_path: The base path to the files for the dataset files.
    :return: Cleaned samples of respective datasets.
    """

    filter_args = {
        "min_length_summary": 40,
        "min_length_reference": 70
    }

    if name == "mlsum":
        data = load_dataset("mlsum", "de")
        filter_args["reference_column"] = "text"
        filter_args["summary_column"] = "summary"

    elif "klexikon" in name:
        data = load_dataset("dennlinger/klexikon")

        data = {
            "train": [klexikon_processing(sample) for sample in data["train"]],
            "validation": [klexikon_processing(sample) for sample in data["validation"]],
            "test": [klexikon_processing(sample) for sample in data["test"]]
        }

        filter_args["reference_column"] = "wiki_text"
        filter_args["summary_column"] = "klexikon_text"

    elif "legalsum" in name:
        train, validation, test = legalsum_processing(base_path)
        data = {
            "train": train,
            "validation": validation,
            "test": test
        }

        filter_args["reference_column"] = "reference"
        filter_args["summary_column"] = "summary"

    elif "eurlexsum" in name:
        with open(os.path.join(base_path, "german_eurlexsum.json")) as f:
            data = json.load(f)
        data = {
            "train": eurlexsum_processing(data["train"]),
            "validation": eurlexsum_processing(data["validation"]),
            "test": eurlexsum_processing(data["test"])
        }
        filter_args["reference_column"] = "reference_text"
        filter_args["summary_column"] = "summary_text"

    elif "wikilingua" in name:

        data = load_dataset("wiki_lingua", "german")
        train = wikilingua_processing(data["train"])

        data = {
            "train": train,
            "validation": None,
            "test": None
        }
        filter_args["min_length_summary"] = 10
        filter_args["min_length_reference"] = 30
        filter_args["reference_column"] = "text"
        filter_args["summary_column"] = "target"

    elif "massivesumm" in name:
        train = []
        # The data is in JSONL format
        with open(os.path.join(base_path, "deu.all.jsonl"), "r") as f:
            for line in f.readlines():
                sample = json.loads(line.strip("\n "))
                # Technically, this shouldn't filter out any samples, but just to make sure...
                if sample["language"] == "deu":
                    train.append(sample)

        data = {
            "train": train,
            "validation": None,
            "test": None
        }
        filter_args["reference_column"] = "text"
        filter_args["summary_column"] = "summary"

    elif "swisstext" in name:
        train = pd.read_csv(os.path.join(base_path, "data_train.csv"), delimiter=",").to_dict("records")

        data = {
            "train": train,
            "validation": None,
            "test": None
        }
        filter_args["reference_column"] = "source"
        filter_args["summary_column"] = "summary"
    else:
        raise ValueError("Unrecognized dataset name passed!")

    # Remove any unwanted samples
    clean_data = filter_and_normalize(data, filter_args)

    return clean_data


def filter_and_normalize(data, args):
    analyzer = Analyzer(lemmatize=True, lang="de")
    cleaner = Cleaner(analyzer, deduplication_method="test_first", length_metric="char",
                      min_length_summary=args["min_length_summary"], min_length_reference=args["min_length_reference"],
                      min_compression_ratio=1.33,
                      extractiveness="fully")
    clean_data = cleaner.clean_dataset(args["summary_column"], args["reference_column"],
                                       data["train"], data["validation"], data["test"], enable_tqdm=True)

    print(f"Loading {[len(v) for _, v in clean_data.items() if v is not None]} samples")

    clean_data = merge_to_standardized_format(clean_data, args)

    return clean_data


def merge_to_standardized_format(clean_data: Dict[str, List], args: Dict) -> Dict:
    """
    Will generate a sample that has the same three attributes "uid", "text" and "target"
    :param clean_data: The cleaned input dataset, a dictionary consisting of one list per split.
    :param args: Containing the reference and summary column name
    :return: The standardized samples
    """
    data = {}
    for split_name, split in clean_data.items():
        data[split_name] = []

        # Skip dummy splits for training-only datasets
        if split is None:
            continue

        for sample in split:
            new_sample = {
                "text": normalize_text(sample[args["reference_column"]]),
                "target": normalize_text(sample[args["summary_column"]])
            }

            data[split_name].append(new_sample)

    return data


def normalize_text(text: str) -> str:
    """
    Function that tries its best to standardize the text by removing trailing whitespaces and special characters.
    :param text: Input text.
    :return: Normalized text version.
    """
    text = text.replace("\xa0", " ")
    text = text.strip("\n ")
    text = text.replace(u"\u0085", "")
    text = text.replace("[]", "")
    text = regex.sub(r"\n([A-Za-z0-9])", r" \g<1>", text, regex.MULTILINE)
    text = regex.sub(r" {2,10}", " ", text)
    text = regex.sub(r"[\n\t]{2,5}", "\n", text, regex.MULTILINE)
    return text


def klexikon_processing(sample: Dict) -> Dict:
    # TODO: Fix behavior with empty lines so we still have paragraph information.
    sample["wiki_text"] = "\n".join([line.strip("\n= ") for line in sample["wiki_sentences"]])
    sample["klexikon_text"] = "\n".join([line.strip("\n= ") for line in sample["klexikon_sentences"]
                                         if line.strip("\n= ") != ""])

    return sample


def eurlexsum_processing(split):
    samples = []
    for celex_id, sample in split.items():
        samples.append(sample)

    return samples


def legalsum_processing(base_path: str) -> Tuple[List, List, List]:
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

        sample = construct_legalsum_sample_from_data(data)

        # Assign the sample to the correct dataset.
        if fn in train_files:
            train.append(sample)
        elif fn in val_files:
            validation.append(sample)
        elif fn in test_files:
            test.append(sample)
        else:
            continue
    return train, validation, test


def load_split_files(fp: str) -> Set:
    with open(fp) as f:
        files = f.readlines()
    return set([fn.strip("\n ") for fn in files])


def construct_legalsum_sample_from_data(data: Dict) -> Dict:
    # Sentence-based storage requires unfolding
    facts = "\n".join([line.strip("\n ") for line in data["facts"]])
    reasoning = "\n".join([line.strip("\n ") for line in data["reasoning"]])
    # Slightly more complex for summary text, given that we have no idea where it's coming from.
    summary = ""
    for sub_text in data["guiding_principle"]:
        for text in sub_text:
            clean_text = text.strip("\n ")
            summary += f"{clean_text}\n"

    reference = f"{facts}\n{reasoning}"

    if isinstance(data["norms"], list):
        if len(data["norms"]) == 0:
            data["norms"] = {}
        else:
            raise ValueError("Unrecognized format for LegalSum norms encountered!")

    reference = replace_norms(reference, data["norms"])
    summary = replace_norms(summary, data["norms"])

    sample = {
        "reference": reference,
        "summary": summary
    }
    return sample


def replace_norms(summary: str, norms: Dict):
    for norm_id, norm_text in norms.items():
        summary = summary.replace(norm_id, norm_text)

    return summary


def wikilingua_processing(data: Dict) -> List:
    """
    Flattens the data samples from WikiLingua.
    :param data: Huggingface dataset of the training portion
    :return: Flattened samples.
    """
    samples = []
    for sample in data:
        doc = sample["article"]
        try:
            if not len(doc["document"]) == len(doc["summary"]):
                raise ValueError(f"Not the same number of sections! {doc}")
        except KeyError:
            raise ValueError(f"Incomplete sample detected: {doc}")
        for idx in range(len(doc["document"])):
            samples.append({
                "url": sample["url"],
                "section_name": doc["section_name"][idx],
                "text": doc["document"][idx],
                "target": doc["summary"][idx]
            })
    return samples


if __name__ == '__main__':

    debug = False

    sample_template = {
        "uid": 0,
        "text": "This is the reference text",
        "target": "This is the summary"
    }

    rng = np.random.default_rng(seed=424242)

    joint_dataset = []
    joint_validation = []
    uid = 1
    validation_uid = 1

    for name in ["mlsum", "massivesumm", "wikilingua", "klexikon", "swisstext", "legalsum", "eurlexsum"]:
        if name == "mlsum":
            prompt = "Zusammenfassung News:"
            base_path = None
        elif name == "massivesumm":
            prompt = "Zusammenfassung News:"
            base_path = "/home/daumiller/massivesumm/"
        elif name == "wikilingua":
            prompt = "Zusammenfassung Instruktionen:"
            base_path = None
        elif name == "klexikon":
            prompt = "Vereinfachte Zusammenfassung Wikipedia:"
            base_path = None
        elif name == "swisstext":
            prompt = "Zusammenfassung Wikipedia:"
            base_path = "/home/daumiller/swisstext/"
        elif name == "legalsum":
            prompt = "Zusammenfassung Gerichtsentscheid:"
            base_path = "/home/daumiller/LegalSum/"
        elif name == "eurlexsum":
            prompt = "Zusammenfassung Legislatur:"
            base_path = "/home/daumiller/german_eurlexsum/"
        else:
            raise ValueError("Unrecognized dataset! Please specify a prompt and paths/loading behavior")

        cleaned_dataset = get_dataset(name, base_path)

        # Choose at most 2000 samples per dataset to incorporate as "unprompted" examples, learning a somewhat
        # general representation for summarization.
        unprompted_sample_size = min([2000, len(cleaned_dataset["train"])])
        unprompted_dataset = rng.choice(cleaned_dataset["train"], unprompted_sample_size, replace=False)

        dataset_id = 1
        if debug:
            unprompted_dataset = unprompted_dataset[:5]

        for instance in unprompted_dataset:
            instance["uid"] = uid
            instance["dataset_id"] = f"{name}_unprompted_{dataset_id}"
            uid += 1
            dataset_id += 1

            joint_dataset.append(instance)

        # Given the size, we add about the same amount of data as the MLSUM dataset contains.
        if name == "massivesumm":
            cleaned_dataset["train"] = rng.choice(cleaned_dataset["train"], 100_000, replace=False)

        # Add the full (prompted) dataset as well
        dataset_id = 1
        if debug:
            cleaned_dataset["train"] = cleaned_dataset["train"][:5]

        for instance in cleaned_dataset["train"]:
            instance["uid"] = uid
            instance["dataset_id"] = f"{name}_{dataset_id}"
            uid += 1
            dataset_id += 1
            # Add the prompt text
            instance["text"] = f"{prompt} {instance['text']}"

            joint_dataset.append(instance)

        if name == "klexikon":
            dataset_id = 1
            for instance in cleaned_dataset["train"]:
                instance["uid"] = uid
                instance["dataset_id"] = f"{name}_second_prompt_{dataset_id}"
                uid += 1
                dataset_id += 1
                # Add the prompt text
                instance["text"] = instance['text'].replace(prompt, "Zusammenfassung Wikipedia:")

                joint_dataset.append(instance)

        if cleaned_dataset["validation"] is not None:
            dataset_id = 1
            sample_size = min([1000, len(cleaned_dataset["validation"])])
            validation_samples = rng.choice(cleaned_dataset["validation"], sample_size, replace=False)

            for instance in validation_samples:
                instance["uid"] = validation_uid
                instance["dataset_id"] = f"{name}_{dataset_id}"
                validation_uid += 1
                dataset_id += 1
                # Add the prompt text
                instance["text"] = f"{prompt} {instance['text']}"

                joint_validation.append(instance)

    with open("german_summarization.jsonl", "w") as f:
        for instance in joint_dataset:
            json.dump(instance, f, ensure_ascii=False)
            f.write("\n")

    with open("german_summarization_validation.jsonl", "w") as f:
        for instance in joint_validation:
            json.dump(instance, f, ensure_ascii=False)
            f.write("\n")
