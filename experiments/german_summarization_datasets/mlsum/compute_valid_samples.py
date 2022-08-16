"""
Basic computation of number of valid samples, if we ignore faulty ones.
"""

import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from summaries.analysis import Analyzer

if __name__ == '__main__':

    dataset = load_dataset("mlsum", "de")

    analyzer = Analyzer(lang="de")
    # for partition_name in ["train", "validation", "test"]:
    for partition_name in ["validation", "test"]:
        partition = dataset[partition_name]

        invalid_lengths = 0
        empty_samples = 0
        fully_extractive_samples = 0
        lcs_scores = []
        multiple_aspects_invalid = 0

        for idx, sample in enumerate(tqdm(partition)):
            multiple_invalids = False
            if analyzer.is_fully_extractive(sample["summary"], sample["text"]):
                fully_extractive_samples += 1
                multiple_invalids = True
            # Only perform LCS check for samples that don't directly match, since we know the LCS score is 1.0
            # for samples that are directly extractive. This is retained as a list so we can analyze the
            # percentiles and evaluate a suitable threshold post-hoc.
            else:
                lcs_scores.append(analyzer.lcs_overlap_fraction(sample["summary"], sample["text"]))

            if analyzer.is_either_text_empty(sample["summary"], sample["text"]):
                empty_samples += 1
                if multiple_invalids:
                    multiple_aspects_invalid += 1
            if analyzer.is_summary_longer_than_reference(sample["summary"], sample["text"]):
                invalid_lengths += 1
                if multiple_invalids:
                    multiple_aspects_invalid += 1

        print(f"Stats for the {partition_name} partition of the MLSUM dataset:")
        print(f"A total of {fully_extractive_samples} samples were fully extractive "
              f"({fully_extractive_samples / len(partition) * 100:.2f}%).")
        print(f"A total of {invalid_lengths} samples had invalid lengths "
              f"({invalid_lengths / len(partition) * 100:.2f}%).")
        print(f"A total of {empty_samples} samples had either empty summary or reference texts "
              f"({empty_samples / len(partition) * 100:.2f}%).")
        print(f"Of those samples, {multiple_aspects_invalid} samples had several issues.")

        # Check the percentiles of partially extractive ones
        print(f"Samples that were not fully extractive had a median LCS score of {np.median(lcs_scores):.4f}")
        print(f"90th percentile: {np.percentile(lcs_scores, 90)}\n\n")

        # Output: See "./valid_sample_report.txt
