"""
What
"""

import os
import json
from typing import Set, Dict, List

from tqdm import tqdm


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

    sample = {
        "id": fn,
        "reference": f"{facts}\n{reasoning}",
        "summary": guiding_principle
    }
    return sample


def get_which_guiding_portion(data: Dict) -> str:
    guiding_principle = data["guiding_principle"]
    if len(guiding_principle) != 2:
        raise ValueError(f"Sample has different guiding principle structure:\n{guiding_principle}")

    if guiding_principle[0] and guiding_principle[1]:
        # raise ValueError(f"Both text portions are filled:\n{guiding_principle}")
        return "both"
    elif guiding_principle:
        return "first"
    elif guiding_principle:
        return "second"
    else:
        raise ValueError(f"Neither text for guiding principle is filled:\n{guiding_principle}")


def print_split(split: List[Dict], guide_split: List[str], split_name: str):
    first_split = len([guide for guide in guide_split if guide == "first"])
    second_split = len([guide for guide in guide_split if guide == "second"])
    both_split = len([guide for guide in guide_split if guide == "both"])
    print(f"{len(split)} samples are in the {split_name} set. "
          f"Of these, {first_split} had text in the first guiding principle ({first_split / len(split) * 100:.2f}%). "
          f"{second_split} had text in the second principle ({second_split / len(split) * 100:.2f}%). "
          f"{both_split} had both sections ({both_split / len(split) * 100:.2f}%).")


if __name__ == '__main__':
    base_path = "/home/dennis/LegalSum/"
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

    for fn in tqdm(os.listdir(os.path.join(base_path, "data/"))):
        fp = os.path.join(base_path, "data/", fn)

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
            # guide_unused.append(get_which_guiding_portion(data))

    print(f"{len(unused_samples)} files were not assigned to any portion.")
    print_split(train, guide_train, "train")
    print_split(validation, guide_validation, "validation")
    print_split(test, guide_test, "test")
