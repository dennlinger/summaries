"""
Applying cleaner to Massivesumm.
"""
import json

from summaries.analysis import Analyzer
from summaries.preprocessing import Cleaner
from summaries.preprocessing.Cleaner import example_print_details

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
    cleaner = Cleaner(analyzer, min_length_summary=20, min_length_reference=50, length_metric="char",
                      extractiveness=(0.10, 0.90))
                      # extractiveness="fully")

    clean_massivesumm = cleaner.clean_dataset("summary", "text", train, enable_tqdm=True,
                                              print_details=example_print_details)
