"""
Applying cleaner to MSLUM.
"""
from typing import Dict
from datasets import load_dataset

from summaries.analysis import Analyzer
from summaries.preprocessing import Cleaner


def custom_print_details(summary: str, reference: str, full_sample: Dict,
                         filter_reason: str, analyzer: Analyzer, split: str) \
        -> None:
    """
    Example of a print_details function implementation.
    This will print the reference and summary if the sample has been filtered out for any reason.
    """
    if filter_reason is not None and len(summary) <= 20:
        print(f"{full_sample}")


if __name__ == '__main__':
    dataset = load_dataset("mlsum", "de")

    analyzer = Analyzer(lemmatize=True, lang="de")
    # Analysis with minimal length requirements set
    cleaner = Cleaner(analyzer, deduplication_method="test_first",
                      length_metric="char", min_length_summary=20, min_length_reference=50,
                      extractiveness="fully")
    # # Alternative analysis that does not impose length requirements
    # cleaner = Cleaner(analyzer, extractiveness="fully")

    clean_massivesumm = cleaner.clean_dataset("summary", "text",
                                              dataset["train"], dataset["validation"], dataset["test"],
                                              enable_tqdm=True)

    # # To investigate samples a bit more, you can additionally pass a function, as defined above:
    # clean_massivesumm = cleaner.clean_dataset("summary", "text", train, enable_tqdm=True,
    #                                           print_details=custom_print_details)

    print(f"Train: ${len(dataset['train'])}$ & ${len(clean_massivesumm[0])}$ & ${len(clean_massivesumm[0]) / len(dataset['train']) * 100:.2f}$")
    print(f"Val: ${len(dataset['validation'])}$ & ${len(clean_massivesumm[1])}$ & ${len(clean_massivesumm[1]) / len(dataset['validation']) * 100:.2f}$")
    print(f"Test: ${len(dataset['test'])}$ & ${len(clean_massivesumm[2])}$ & ${len(clean_massivesumm[2]) / len(dataset['test']) * 100:.2f}$")