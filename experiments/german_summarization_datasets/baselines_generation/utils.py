"""
Utility functions used across experiments
"""

from datasets import load_dataset

from summaries import Analyzer, Cleaner


def get_dataset(name: str, filtered: bool = False):
    if name == "mlsum":
        data = load_dataset("mlsum", "de")

        if filtered:
            reference_column = "text"
            summary_column = "summary"
    else:
        raise ValueError("Unrecognized dataset name passed!")

    if filtered:
        analyzer = Analyzer(lemmatize=True, lang="de")
        cleaner = Cleaner(analyzer, deduplication_method="test_first",
                          length_metric="char", min_length_summary=20, min_length_reference=50,
                          min_compression_ratio=1.25,
                          extractiveness="fully")
        clean_data = cleaner.clean_dataset(summary_column, reference_column,
                                           data["train"], data["validation"], data["test"], enable_tqdm=True)
        return clean_data
    else:
        return data
