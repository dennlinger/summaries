"""
Processing script for the LegalSum dataset by Glaser et al.
Code and download links associated with the original dataset can be found here: https://github.com/sebimo/LegalSum
"""

import os
import json
from collections import Counter
from typing import Set, Dict, List

from tqdm import tqdm

from summaries import Analyzer, Cleaner


def load_split_files(fp: str) -> Set:
    with open(fp) as f:
        files = f.readlines()
    return set([fn.strip("\n ") for fn in files])


def construct_sample_from_data(data: Dict, fn: str) -> Dict:
    """
    The actual reference-summary tuple is spread across multiple fields in the original dataset.
    :param data: Dataframe of a single sample.
    :param fn: File name which will be stored as an additional attribute for future reference.
    :return: Structured sample in parseable format.
    """
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


def get_which_guiding_portion(data: Dict) -> str:
    """
    This is a brief analysis of the sources for which guiding portions are present.
    The original authors are not 100% clear on where exactly they are from.
    """
    guiding_principle = data["guiding_principle"]
    if len(guiding_principle) != 2:
        raise ValueError(f"Sample has different guiding principle structure:\n{guiding_principle}")

    if guiding_principle[0] and guiding_principle[1]:
        # raise ValueError(f"Both text portions are filled:\n{guiding_principle}")
        return "both"
    elif guiding_principle[0]:
        return "first"
    elif guiding_principle[1]:
        return "second"
    else:
        raise ValueError(f"Neither text for guiding principle is filled:\n{guiding_principle}")


def print_split(split: List[Dict], guide_split: List[str], split_name: str):
    first_split = len([guide for guide in guide_split if guide == "first"])
    second_split = len([guide for guide in guide_split if guide == "second"])
    both_split = len([guide for guide in guide_split if guide == "both"])
    print(f"{len(split)} samples are in the {split_name} set.\n"
          f"Of these, {first_split} had text in the first guiding principle ({first_split / len(split) * 100:.2f}%).\n"
          f"{second_split} had text in the second principle ({second_split / len(split) * 100:.2f}%). "
          f"{both_split} had both sections ({both_split / len(split) * 100:.2f}%).\n\n")


if __name__ == '__main__':
    base_path = "./"
    train_files = load_split_files(os.path.join(base_path, "train_files.txt"))
    val_files = load_split_files(os.path.join(base_path, "val_files.txt"))
    test_files = load_split_files(os.path.join(base_path, "test_files.txt"))

    train = []
    validation = []
    test = []
    unused_samples = []

    guide_train = []
    guide_validation = []
    guide_test = []
    guide_unused = []

    for fn in tqdm(os.listdir(os.path.join(base_path, "LegalSum/"))):
        fp = os.path.join(base_path, "LegalSum/", fn)

        with open(fp) as f:
            data = json.load(f)
        sample = construct_sample_from_data(data, fn)

        # Assign the sample to the correct dataset.
        if fn in train_files:
            train.append(sample)
            guide_train.append(get_which_guiding_portion(data))
        elif fn in val_files:
            validation.append(sample)
            guide_validation.append(get_which_guiding_portion(data))
        elif fn in test_files:
            test.append(sample)
            guide_test.append(get_which_guiding_portion(data))
        else:
            unused_samples.append(sample)

    print(f"{len(unused_samples)} files were not assigned to any portion.")
    print_split(train, guide_train, "train")
    print_split(validation, guide_validation, "validation")
    print_split(test, guide_test, "test")

    analyzer = Analyzer(lemmatize=True, lang="de")
    cleaner = Cleaner(analyzer, deduplication_method="test_first",
                      min_length_summary=20, min_length_reference=50, length_metric="char",
                      min_compression_ratio=1.25,
                      # extractiveness=(0.10, 0.90))  # Takes super long...
                      extractiveness="fully")

    clean_dataset = cleaner.clean_dataset("summary", "reference", train, validation, test, enable_tqdm=True)

