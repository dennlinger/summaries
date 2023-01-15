"""
Script to work with the summarization data for the Swisstext summarization challenge.
Notably, this includes only a training and test set, but no validation set.
Also, since the copyright of this dataset is unclear, there exists no processed version on Huggingface.

The dataset can be downloaded from https://www.swisstext.org/2019/shared-task/german-text-summarization-challenge.html
"""
import os

import pandas as pd

from summaries import Analyzer, Cleaner


if __name__ == '__main__':
    base_path = "./"

    train_set = pd.read_csv(os.path.join(base_path, "data_train.csv"), delimiter=",").to_dict("records")
    # The test set only contains the text, not the summaries...
    # test_set = pd.read_csv(os.path.join(base_path, "data_test.csv"), delimiter=",").to_dict("records")

    analyzer = Analyzer(lemmatize=True, lang="de")
    cleaner = Cleaner(analyzer, deduplication_method="test_first",
                      min_length_summary=20, min_length_reference=50, length_metric="char",
                      min_compression_ratio=1.25,
                      extractiveness="fully")

    clean_data = cleaner.clean_dataset(summary_text_column_name="summary",
                                       reference_text_column_name="source",
                                       train_set=train_set,
                                       enable_tqdm=True)
