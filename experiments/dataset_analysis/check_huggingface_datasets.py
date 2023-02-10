from typing import Dict

from datasets import load_dataset

from summaries import Analyzer, Cleaner


def custom_print_details(summary: str, reference: str, full_sample: Dict,
                         filter_reason: str, analyzer: Analyzer, split: str) \
        -> None:
    """
    Example of a print_details function implementation.
    This will print the reference and summary if the sample has been filtered out for any reason.
    """
    if filter_reason == "summary_duplicate":
        try:
            print(f"{full_sample['summary']}")
        except KeyError:
            print(f"{full_sample['highlights']}")


if __name__ == '__main__':
    analyzer = Analyzer(lemmatize=True, lang="en")
    cleaner = Cleaner(analyzer, deduplication_method="test_first",
                      min_length_summary=20, min_length_reference=50, length_metric="char",
                      min_compression_ratio=1.25,
                      extractiveness="fully")

    dataset = load_dataset("cnn_dailymail", "3.0.0")
    print("CNN/DailyMail (HF Version 3.0.0):")
    clean_dataset = cleaner.clean_dataset("highlights", "article",
                                          dataset["train"], dataset["validation"], dataset["test"],
                                          enable_tqdm=False, print_details=custom_print_details)

    dataset = load_dataset("xsum")
    print("XSUM (HF Version):")
    clean_dataset = cleaner.clean_dataset("summary", "document",
                                          dataset["train"], dataset["validation"], dataset["test"],
                                          enable_tqdm=False, print_details=custom_print_details)

    # Delete variables to not crash pycharm
    del dataset
    del clean_dataset

