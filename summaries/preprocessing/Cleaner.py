"""
Utilizing the functions available in `summaries.analysis.Analyzer`, this module is able to process samples
and remove unwanted instances (e.g., empty samples) and return a cleaned version of a particular dataset.
"""
from typing import Optional, List, Union, Tuple, Dict
import warnings

from datasets import Dataset

from ..analysis import Analyzer


class Cleaner:
    analyzer: Analyzer
    deduplication_method: str
    min_length_reference: int
    min_length_summary: int
    length_metric: str
    ngram_similarity_range: Optional[Tuple[float, float]]

    # TODO: Add a parameter to make removal of incompatible length samples
    def __init__(self,
                 analyzer: Optional[Analyzer] = None,
                 deduplication_method: str = "first",
                 min_length_summary: int = 0,
                 min_length_reference: int = 0,
                 length_metric: str = "char",
                 ngram_similarity_range: Optional[Tuple[float, float]] = None) -> None:
        """
        Initializes a `Cleaner` object with specified parameters.
        :param analyzer: The analyzer defines properties for the analysis, such as the language processing and
        :param deduplication_method: Currently accepts "first", which will retain the first sample
            with a particular text, and discard other duplicates or leaks, or "none", which will do no filtering.
        :param min_length_summary: Minimum length of the summary text (see `length_metric`).
        :param min_length_reference: Minimum length of the reference text (see `length_metric`).
        :param length_metric: Which way to calculate lengths (either one of ["char", "whitespace", "token"]).
        :param ngram_similarity_range: If specified, will restrict all samples to have a minimal/maximal ngram
            similarity between the gold summary and the reference summary.
            This can be utilized to avoid, for example, fully extractive samples (by specifying upper bound < 1.0).
        """
        # FIXME: Extend docstrings to explain deduplication with first a bit better. Because currently, this method
        #  will retain training instances over validation / test instances, based on the iteration order.
        if analyzer is not None:
            self.analyzer = analyzer
        else:
            warnings.warn(f"Falling back to an English Analyzer object that has lemmatization enabled. "
                          f"If these specifications do not match your expectations, please provide a custom Analyzer "
                          f"to your Cleaner!")
            self.analyzer = Analyzer(lemmatize=True, lang="en")
        self.valid_deduplication_methods = ["first", "none"]
        if deduplication_method not in self.valid_deduplication_methods:
            raise ValueError(f"Supplied deduplication method is invalid! "
                             f"Supported deduplication methods: {self.valid_deduplication_methods}")
        self.deduplication_method = deduplication_method

        if length_metric not in self.analyzer.valid_length_methods:
            raise ValueError(f"Supplied length method is invalid! "
                             f"Supported length methods: {self.analyzer.valid_length_methods}")
        self.length_metric = length_metric
        self.min_length_reference = min_length_reference
        self.min_length_summary = min_length_summary

        if ngram_similarity_range is not None:
            if ngram_similarity_range[0] < 0.0 or ngram_similarity_range[1] > 1.0:
                raise ValueError("Similarity range must fall in the closed interval [0.0, 1.0]!")
            self.ngram_similarity_range = ngram_similarity_range
        else:
            self.ngram_similarity_range = None

    def clean_dataset(self,
                      summary_text_column_name: str,
                      reference_text_column_name: str,
                      train_set: Optional[Union[List[Dict], Dataset]] = None,
                      validation_set: Optional[Union[List[Dict], Dataset]] = None,
                      test_set: Optional[Union[List[Dict], Dataset]] = None) -> Tuple[List, List, List]:
        """
        Function that removes samples based on the available Analyzer module. By default, removes the following:
            - Samples with incompatible lengths (summary longer than reference)
            - Samples with ngram overlaps that do not fall into the ngram_similarity_range (if specified)
            - Duplicate samples (according to deduplcation_strategy). This will retain one sample for duplications.
              This will also remove duplicates across the different splits, to avoid leakage.
        :param summary_text_column_name: Key to identify the summary text in a sample.
        :param reference_text_column_name: Key to identify the reference text in a sample.
        :param train_set: List of dicts, or alternatively Huggingface dataset, representing the training samples.
        :param validation_set: List of dicts, or alternatively Huggingface dataset, representing the validation samples.
        :param test_set: List of dicts, or alternatively Huggingface dataset, representing the test samples.
        :return: Returns a tuple of (cleaned_train, cleaned_val, cleaned_test),
            where each of those is a list of samples that satisfy all filtering criteria.
        """

        passed_sets = self.analyzer.get_passed_splits_with_names(train_set, validation_set, test_set)

        # Necessary to return all splits in the end, and retain which ones we have seen.
        cleaned_splits = {"train": None, "validation": None, "test": None}

        previously_seen_references = set()
        previously_seen_summaries = set()

        # TODO: There could be a better way to initialize and track this.
        #  Could also be extended to account for removal by split?
        filter_count_with_reason = {"too_short": 0, "identity_sample": 0, "longer_summary": 0,
                                    "ngram_range": 0, "duplicate": 0}

        # Iterate through all available datasets
        for split, name in passed_sets:
            cleaned_splits[name] = []
            for sample in split:
                # Basically skip at any point we encounter some invalidating property. Only add in the end.
                current_summary = sample[summary_text_column_name]
                current_reference = sample[reference_text_column_name]

                # Check for samples to retain minimal length
                if self.analyzer.is_text_too_short(current_reference,
                                                   min_length=self.min_length_reference,
                                                   length_metric=self.length_metric) or \
                   self.analyzer.is_text_too_short(current_summary,
                                                   min_length=self.min_length_summary,
                                                   length_metric=self.length_metric):
                    filter_count_with_reason["too_short"] += 1
                    continue
                # Check for differences in input/output per sample
                if self.analyzer.is_identity_sample(current_summary, current_reference, comparison_method="exact"):
                    filter_count_with_reason["identity_sample"] += 1
                    continue
                # TODO: Introduce parameter that determines whether this is a filtering criterion
                if self.analyzer.is_summary_longer_than_reference(current_summary,
                                                                  current_reference,
                                                                  length_metric=self.length_metric):
                    filter_count_with_reason["too_short"] += 1
                    continue
                # If no range for the similarities is specified, no need to check it.
                if self.ngram_similarity_range is not None:
                    sample_similarity = self.analyzer.ngram_overlap_fraction(current_summary, current_reference, n=2)
                    # Check whether the actual similarity is outside the specified range
                    if self.ngram_similarity_range[0] > sample_similarity or \
                       self.ngram_similarity_range[1] < sample_similarity:
                        filter_count_with_reason["ngram_range"] += 1
                        continue
                # Deduplication is a bit more tricky, especially once we add more supported methods.
                if self.deduplication_method == "first":
                    if current_summary in previously_seen_summaries or \
                       current_reference in previously_seen_references:
                        filter_count_with_reason["duplicate"] += 1
                        continue
                # TODO: Catch other deduplication methods here
                else:
                    pass

                # If all checks have been "survived", add the sample as it is deemed valid.
                cleaned_splits[name].append(sample)

                # Also retain their checks for deduplication if necessary. This can be only done at this stage,
                # because it might be the case that we introduce more filters later on, which could conflict with
                # the current deduplication step.
                if self.deduplication_method == "first":
                    previously_seen_summaries.add(current_summary)
                    previously_seen_references.add(current_reference)

        print(f"{sum(filter_count_with_reason.values())} samples were removed from the dataset.")
        print(f"Breakdown by filter category:")
        for reason, count in filter_count_with_reason.items():
            print(f"Reason '{reason}': {count} samples removed.")

        # FIXME: Currently "converts" Huggingface dataset inputs to List-based outputs for simplicity of internal
        #  handling, since we otherwise have to differentiate at some point.
        return cleaned_splits["train"], cleaned_splits["validation"], cleaned_splits["test"]
