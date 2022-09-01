"""
Analysis and cleaning of the EUR-LexSum dataset by Klaus et al. (SIGIR'22).
See https://github.com/svea-klaus/Legal-Document-Summarization for the data
and https://doi.org/10.1145/3477495.3531872 for the paper.
"""
import os

from summaries import Analyzer, Cleaner


def merge_files(base_path: str, split: str):
    with open(os.path.join(base_path, f"{split}.source")) as source, \
        open(os.path.join(base_path, f"{split}.target")) as target:
        source_file_lines = source.readlines()
        target_file_lines = target.readlines()

    samples = []
    for source_line, target_line in zip(source_file_lines, target_file_lines):
        samples.append({"reference": source_line.strip("\n "), "summary": target_line.strip("\n ")})

    return samples


if __name__ == '__main__':
    base_path = "/home/dennis/english_eurlexsum/"

    train = merge_files(base_path, "train")
    validation = merge_files(base_path, "val")
    test = merge_files(base_path, "test")

    analyzer = Analyzer(lemmatize=True, lang="en")
    cleaner = Cleaner(analyzer, deduplication_method="first", min_length_summary=50, min_length_reference=50,
                      extractiveness="fully")

    clean_splits = cleaner.clean_dataset(summary_text_column_name="summary", reference_text_column_name="reference",
                                         train_set=train, validation_set=validation, test_set=test)