"""
Utilizing the functions available in `summaries.analysis.Analyzer`, this module is able to process samples
and remove unwanted instances (e.g., empty samples) and return a cleaned version of a particular dataset.
"""
from typing import Optional, List, Union, Tuple, Dict, Callable
import warnings

from tqdm import tqdm
from datasets import Dataset

from ..analysis import Analyzer



class Cleaner:
    analyzer: Analyzer
    deduplication_method: str
    min_compression_ratio: float
    min_length_summary: int
    min_length_reference: int
    length_metric: str
    extractiveness: Optional[Union[Tuple[float, float], str]]

    # TODO: Add a parameter to make removal of incompatible length samples
    def __init__(self,
                 analyzer: Optional[Analyzer] = None,
                 deduplication_method: str = "test_first",
                 min_compression_ratio: float = 1.0,
                 min_length_summary: int = 0,
                 min_length_reference: int = 0,
                 length_metric: str = "char",
                 extractiveness: Optional[Union[Tuple[float, float], str]] = None) -> None:
        """
        Initializes a `Cleaner` object with specified parameters.
        :param analyzer: The analyzer defines properties for the analysis, such as the language processing and
        :param deduplication_method: Currently accepts the following options:
            * "first", which will retain the first sample with a particular text, and discard other duplicates or leaks,
            * "test_first", which will reverse the order of passed sets from (train, val, test) to (test, val, train),
            * "none", which will perform no duplicate filtering.
        :param min_compression_ratio: Minimum compression ratio (len(reference)/len(summary) that should be ensured.
            1.0 means summaries have to be strictly shorter than the references.
        :param min_length_summary: Minimum length of the summary text (see `length_metric`).
        :param min_length_reference: Minimum length of the reference text (see `length_metric`).
        :param length_metric: Which way to calculate lengths (either one of ["char", "whitespace", "token"]).
        :param extractiveness: If specified, will restrict all samples to have a minimal/maximal ngram
            similarity between the gold summary and the reference summary. This can be utilized to avoid bad alignments,
            or samples which overlap too much with the source. Since this is very expensive, also takes the value of
            "fully", which will approximate filtering of fully extractive samples (orders of magnitudes faster).
        """
        # FIXME: Extend docstrings to explain deduplication with first a bit better. Because currently, this method
        #  will retain training instances over validation / test instances, based on the iteration order.
        if analyzer is not None:
            self.analyzer = analyzer
        else:
            warnings.warn(f"Falling back to an English Analyzer object that has lemmatization enabled. "
                          f"This affects the cleaning if either of the following conditions is met:\n"
                          f"  * `length_metric` is set to `token`\n"
                          f"  * `extractiveness` is set to something other than `fully`")
            self.analyzer = Analyzer(lemmatize=True, lang="en")
        self.valid_deduplication_methods = ["first", "test_first", "none"]
        if deduplication_method not in self.valid_deduplication_methods:
            raise ValueError(f"Supplied deduplication method is invalid! "
                             f"Supported deduplication methods: {self.valid_deduplication_methods}")
        self.deduplication_method = deduplication_method

        if length_metric not in self.analyzer.valid_length_methods:
            raise ValueError(f"Supplied length method is invalid! "
                             f"Supported length methods: {self.analyzer.valid_length_methods}")
        self.length_metric = length_metric
        if min_compression_ratio < 0.0:
            raise ValueError("The minimum compression ratio must be greater or equal to zero!")
        elif min_compression_ratio < 1.0:
            warnings.warn("The currently set minimum compression ratio allows summaries longer than the reference!")
        self.min_compression_ratio = min_compression_ratio
        if min_length_reference < 0 or min_length_summary < 0:
            raise ValueError("Please provide a positive integer (or zero) for the minimum length!")
        self.min_length_summary = min_length_summary
        self.min_length_reference = min_length_reference

        # Determine the input type of the extractiveness parameter
        if extractiveness is not None:
            if isinstance(extractiveness, tuple):
                if extractiveness[0] < 0.0 or extractiveness[1] > 1.0:
                    raise ValueError("Extractiveness range must fall in the closed interval [0.0, 1.0]!")
                self.extractiveness = extractiveness
            elif isinstance(extractiveness, str):
                if extractiveness == "fully":
                    self.extractiveness = extractiveness
                else:
                    raise ValueError(f"Invalid extractiveness option supplied: {extractiveness}")
            else:
                raise ValueError("Incorrect data type for extractiveness provided. Must be tuple or string.")
        else:
            self.extractiveness = None

    def clean_dataset(self,
                      summary_text_column_name: str,
                      reference_text_column_name: str,
                      train_set: Optional[Union[List[Dict], Dataset]] = None,
                      validation_set: Optional[Union[List[Dict], Dataset]] = None,
                      test_set: Optional[Union[List[Dict], Dataset]] = None,
                      print_details: Optional[Callable] = None,
                      enable_tqdm: bool = False,
                      print_breakdown: bool = True) -> Dict:
        """
        Function that removes samples based on the available Analyzer module. By default, removes the following:
            - Samples with incompatible lengths (summary longer than reference)
            - Samples with ngram overlaps that do not fall into the extractiveness (if specified)
            - Duplicate samples (according to deduplcation_strategy). This will retain one sample for duplications.
              This will also remove duplicates across the different splits, to avoid leakage.
        :param summary_text_column_name: Key to identify the summary text in a sample.
        :param reference_text_column_name: Key to identify the reference text in a sample.
        :param train_set: List of dicts, or alternatively Huggingface dataset, representing the training samples.
        :param validation_set: List of dicts, or alternatively Huggingface dataset, representing the validation samples.
        :param test_set: List of dicts, or alternatively Huggingface dataset, representing the test samples.
        :param print_details: Optional function that will be called on samples. Could, for example, include a print
            condition to inspect samples that will be filtered on specific criteria.
        :param enable_tqdm: Will show a progress bar if enabled.
        :param print_breakdown: If enabled, will output a breakdown of where samples were filtered.
        :return: Returns a Dict of ("train": cleaned_train, "validation": cleaned_val, "test": cleaned_test),
            where each of those is a list of samples that satisfy all filtering criteria (or None, if none was passed).
        """

        passed_sets = self.analyzer.get_passed_splits_with_names(train_set, validation_set, test_set)

        # Ensure correct iteration order for "test first" deduplication
        if self.deduplication_method == "test_first":
            passed_sets = reversed(passed_sets)

        # Necessary to return all splits in the end, and retain which ones we have seen.
        cleaned_splits = {"train": None, "validation": None, "test": None}

        previously_seen_references = set()
        previously_seen_summaries = set()
        previously_seen_both = set()

        # TODO: There could be a better way to initialize and track this.
        #  Could also be extended to account for removal by split?
        filter_count_with_reason = {"reference_too_short": {"train": 0, "validation": 0, "test": 0},
                                    "summary_too_short": {"train": 0, "validation": 0, "test": 0},
                                    "identity_sample": {"train": 0, "validation": 0, "test": 0},
                                    "compression_ratio": {"train": 0, "validation": 0, "test": 0},
                                    "extractiveness": {"train": 0, "validation": 0, "test": 0},
                                    "exact_duplicate": {"train": 0, "validation": 0, "test": 0},
                                    "both_duplicate": {"train": 0, "validation": 0, "test": 0},
                                    "summary_duplicate": {"train": 0, "validation": 0, "test": 0},
                                    "reference_duplicate": {"train": 0, "validation": 0, "test": 0}}

        # Iterate through all available datasets
        for split, split_name in passed_sets:
            cleaned_splits[split_name] = []

            # Iteration checks, which are required to maintain logic.
            prev_sample = None
            filter_reason = None
            current_reference = None
            current_summary = None

            for sample in tqdm(split, disable=not enable_tqdm):

                # Check whether the previous sample was filtered. This way, we still retain only one filter reason.
                # FIXME: With multiple filter reason, this view gets distorted?
                if prev_sample is not None:
                    # Potentially run inspection function supplied by users
                    if callable(print_details):
                        print_details(current_summary, current_reference, prev_sample,
                                      filter_reason, self.analyzer, split_name)

                # Basically skip at any point we encounter some invalidating property. Only add in the end.
                current_summary = sample[summary_text_column_name]
                current_reference = sample[reference_text_column_name]
                prev_sample = sample
                filter_reason = None

                # Check for samples to retain minimal length
                if self.analyzer.is_text_too_short(current_reference,
                                                   min_length=self.min_length_reference,
                                                   length_metric=self.length_metric):
                    filter_reason = "reference_too_short"
                    filter_count_with_reason[filter_reason][split_name] += 1
                    continue

                if self.analyzer.is_text_too_short(current_summary,
                                                   min_length=self.min_length_summary,
                                                   length_metric=self.length_metric):
                    filter_reason = "summary_too_short"
                    filter_count_with_reason[filter_reason][split_name] += 1
                    continue
                # Check for differences in input/output per sample
                if self.analyzer.is_identity_sample(current_summary, current_reference, comparison_method="exact"):
                    filter_reason = "identity_sample"
                    filter_count_with_reason[filter_reason][split_name] += 1
                    continue
                # Filter based on the compression ratio
                if self.analyzer.compression_ratio(current_summary, current_reference, self.length_metric) < \
                   self.min_compression_ratio:
                    filter_reason = "compression_ratio"
                    filter_count_with_reason[filter_reason][split_name] += 1
                    continue
                # If no range for the similarities is specified, no need to check it.
                if self.extractiveness is not None:
                    if isinstance(self.extractiveness, tuple):
                        # TODO: Let users decide on which overlap function to use?
                        curr_similarity = self.analyzer.ngram_overlap_fraction(current_summary, current_reference, n=2)
                        # Check whether the actual similarity is outside the specified range
                        if self.extractiveness[0] > curr_similarity or \
                           self.extractiveness[1] < curr_similarity:
                            filter_reason = "extractiveness"
                            filter_count_with_reason[filter_reason][split_name] += 1
                            continue
                    elif self.extractiveness == "fully":
                        if self.analyzer.is_fully_extractive(current_summary, current_reference):
                            filter_reason = "extractiveness"
                            filter_count_with_reason[filter_reason][split_name] += 1
                            continue
                # Deduplication is a bit more tricky, especially once we add more supported methods.
                if self.deduplication_method in ["first", "test_first"]:
                    if (current_reference, current_summary) in previously_seen_both:
                        filter_reason = "exact_duplicate"
                        filter_count_with_reason[filter_reason][split_name] += 1
                        continue
                    elif current_reference in previously_seen_references and \
                            current_summary in previously_seen_summaries:
                        filter_reason = "both_duplicate"
                        filter_count_with_reason[filter_reason][split_name] += 1
                        continue
                    elif current_reference in previously_seen_references:
                        filter_reason = "reference_duplicate"
                        filter_count_with_reason[filter_reason][split_name] += 1
                        continue
                    elif current_summary in previously_seen_summaries:
                        filter_reason = "summary_duplicate"
                        filter_count_with_reason[filter_reason][split_name] += 1
                        continue
                # TODO: Catch other deduplication methods here
                else:
                    pass

                # If all checks have been "survived", add the sample as it is deemed valid.
                cleaned_splits[split_name].append(sample)

                # Also retain their checks for deduplication if necessary. This can be only done at this stage,
                # because it might be the case that we introduce more filters later on, which could conflict with
                # the current deduplication step.
                if self.deduplication_method in ["first", "test_first"]:
                    previously_seen_summaries.add(current_summary)
                    previously_seen_references.add(current_reference)
                    previously_seen_both.add((current_reference, current_summary))

        if print_breakdown:
            total_filtered = sum([sum(reason.values()) for reason in filter_count_with_reason.values()])
            print(f"{total_filtered} samples were removed from the dataset.")
            print(f"Breakdown by filter category:")
            for reason, count in filter_count_with_reason.items():
                print(f"'{reason}': {count} samples removed across splits.")

            # if train_set is not None:
            #     format_print(train_set, filter_count_with_reason, "train", cleaned_splits)
            # if validation_set is not None:
            #     format_print(validation_set, filter_count_with_reason, "validation", cleaned_splits)
            # if test_set is not None:
            #     format_print(test_set, filter_count_with_reason, "test", cleaned_splits)

        # FIXME: Currently "converts" Huggingface dataset inputs to List-based outputs for simplicity of internal
        #  handling, since we otherwise have to differentiate at some point.
        return cleaned_splits


# def format_print(split, filter_count_with_reason, split_name, cleaned_splits):
#     print(f"& {split_name} & ${len(split)}$ & "
#           f"${filter_count_with_reason['reference_too_short'][split_name]}$ & "
#           f"${filter_count_with_reason['summary_too_short'][split_name]}$ & "
#           f"${filter_count_with_reason['identity_sample'][split_name]}$ & "
#           f"${filter_count_with_reason['compression_ratio'][split_name]}$ & "
#           f"${filter_count_with_reason['extractiveness'][split_name]}$ & "
#           f"${filter_count_with_reason['exact_duplicate'][split_name] + filter_count_with_reason['both_duplicate'][split_name]}$ & "
#           f"${filter_count_with_reason['reference_duplicate'][split_name]}$ & "
#           f"${filter_count_with_reason['summary_duplicate'][split_name]}$ & "
#           f"${len(cleaned_splits[split_name])}$ & "
#           f"$({len(cleaned_splits[split_name]) / len(split) * 100:.2f}\\%)$ \\\\")


def example_print_details(summary: str, reference: str, full_sample: Dict,
                          filter_reason: str, analyzer: Analyzer, split: str) \
        -> None:
    """
    Example of a print_details function implementation.
    This will print the reference and summary if the sample has been filtered out for any reason.
    """
    if filter_reason is not None:
        print(reference)
        print(summary)
        print(f"\n\n\n{full_sample}")
