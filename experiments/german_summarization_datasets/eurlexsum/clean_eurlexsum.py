"""
Dataset cleaning of the EUR-Lex-Sum dataset by Aumiller et al.
The dataset can be online: https://huggingface.co/datasets/dennlinger/eur-lex-sum
"""

import json
import os

from summaries import Analyzer, Cleaner

def extract_samples(split):
    samples = []
    for celex_id, sample in split.items():
        samples.append(sample)

    return samples


if __name__ == '__main__':
    # For the offline experiments, we utilize the pre-releaesd version of our dataset.
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "german_eurlexsum.json")
    with open(data_path) as f:
        dataset = json.load(f)
    train = extract_samples(dataset["train"])
    validation = extract_samples(dataset["validation"])
    test = extract_samples(dataset["test"])

    analyzer = Analyzer(lemmatize=True, lang="de")
    cleaner = Cleaner(analyzer, deduplication_method="test_first",
                      min_length_summary=20, min_length_reference=50, length_metric="char",
                      min_compression_ratio=1.25,
                      extractiveness="fully")

    clean_split = cleaner.clean_dataset("summary_text", "reference_text", train, validation, test, enable_tqdm=True)