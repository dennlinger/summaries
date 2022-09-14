"""
Applying cleaner to Massivesumm.
"""
from typing import Dict
import json

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
    data_path = "/home/dennis/massivesumm/deu.all.jsonl"

    train = []
    # The data is in JSONL format
    with open(data_path, "r") as f:
        for line in f.readlines():
            sample = json.loads(line.strip("\n "))
            # Technically, this shouldn't filter out any samples, but just to make sure...
            if sample["language"] == "deu":
                train.append(sample)

    print(f"{len(train)} samples before filtering.")

    analyzer = Analyzer(lemmatize=True, lang="de")
    # Analysis with minimal length requirements set
    cleaner = Cleaner(analyzer, deduplication_method="test_first",
                      length_metric="char", min_length_summary=20, min_length_reference=50,
                      min_compression_ratio=1.25,
                      extractiveness="fully")
    # # Alternative analysis that does not impose length requirements
    # cleaner = Cleaner(analyzer, extractiveness="fully")

    clean_massivesumm = cleaner.clean_dataset("summary", "text", train, enable_tqdm=True)

    # # To investigate samples a bit more, you can additionally pass a function, as defined above:
    # clean_massivesumm = cleaner.clean_dataset("summary", "text", train, enable_tqdm=True,
    #                                           print_details=custom_print_details)
