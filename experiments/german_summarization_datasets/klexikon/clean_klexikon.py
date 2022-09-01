"""
Verify that the cleaner works as intended, and look for issues in Klexikon.
"""
from typing import Dict
from datasets import load_dataset
from summaries import Analyzer, Cleaner


def fuse_sentences_but_keep_sample(sample):
    sample["wiki_text"] = "\n".join(sample["wiki_sentences"])
    sample["klexikon_text"] = "\n".join(sample["klexikon_sentences"])

    return sample


def custom_print_details(summary: str, reference: str, full_sample: Dict,
                         filter_reason: str, analyzer: Analyzer, split: str) \
        -> None:
    """
    Example of a print_details function implementation.
    This will print the reference and summary if the sample has been filtered out for any reason.
    """

    if filter_reason == "duplicate":
        print(f"Split: {split}")
        print(f"u_id: {full_sample['u_id']}, title: {full_sample['title']}, wiki_url: {full_sample['wiki_url']}, "
              f"klexikon_url: {full_sample['klexikon_url']}")

        print(f"{full_sample}")


if __name__ == '__main__':
    klexikon = load_dataset("dennlinger/klexikon")

    analyzer = Analyzer(lemmatize=True, lang="de")
    cleaner = Cleaner(analyzer, min_length_summary=20, min_length_reference=50, length_metric="char",
                      # extractiveness=(0.10, 0.90))
                      extractiveness="fully")

    train = [fuse_sentences_but_keep_sample(sample) for sample in klexikon["train"]]
    validation = [fuse_sentences_but_keep_sample(sample) for sample in klexikon["validation"]]
    test = [fuse_sentences_but_keep_sample(sample) for sample in klexikon["test"]]

    clean_klexikon = cleaner.clean_dataset("klexikon_text", "wiki_text", train, validation, test, enable_tqdm=True,
                                           print_details=custom_print_details)