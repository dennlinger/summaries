"""
Verify that the cleaner works as intended, and look for issues in Klexikon.
"""

from datasets import load_dataset
from summaries.preprocessing import Cleaner
from summaries.analysis import Analyzer


def fuse_sentences_but_keep_sample(sample):
    sample["wiki_text"] = "\n".join(sample["wiki_sentences"])
    sample["klexikon_text"] = "\n".join(sample["klexikon_sentences"])

    return sample


if __name__ == '__main__':
    klexikon = load_dataset("dennlinger/klexikon")

    analyzer = Analyzer(lemmatize=True, lang="de")
    cleaner = Cleaner(analyzer, min_length_summary=20, min_length_reference=50, length_metric="char",
                      extractiveness=(0.10, 0.90))
                      # extractiveness="fully")

    train = [fuse_sentences_but_keep_sample(sample) for sample in klexikon["train"]]
    validation = [fuse_sentences_but_keep_sample(sample) for sample in klexikon["validation"]]
    test = [fuse_sentences_but_keep_sample(sample) for sample in klexikon["test"]]

    clean_klexikon = cleaner.clean_dataset("klexikon_text", "wiki_text", train, validation, test, enable_tqdm=True)