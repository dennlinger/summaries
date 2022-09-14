"""
A collection of different functions for analyzing summaries (both reference texts and generated ones).
This can either be done through visual inspection of diversity (Density Plots),
analyzing the textual coherence of texts (finding repetitions),
or simply checking for the amount of overlap with a reference text.
"""
from typing import Optional, List, Union, Tuple
from collections import Counter

from rouge_score.rouge_scorer import _create_ngrams, _score_lcs, _score_ngrams
from spacy.language import Language
from datasets import Dataset

from ..utils import get_nlp_model, interpret_lang_code


class Analyzer:
    lemmatize: bool
    valid_comparison_methods: List[str]
    processor: Optional[Language]
    lang_code: Optional[str]
    print_cutoff_length: Optional[int]

    def __init__(self,
                 lemmatize: bool = False,
                 processor: Optional[Language] = None,
                 lang: Optional[str] = None,
                 print_cutoff_length: Optional[int] = None):
        """
        Initializes an Analyzer object.
        :param lemmatize: Boolean to indicate whether lemmas should be used for token-level functions.
        :param processor: Language-specific spaCy model. Either this or `lang` has to be specified.
        :param lang: As an alternative, a language code (e.g., "en" or "de") could be provided, and a spaCy model is
            loaded based on this information.
        :param print_cutoff_length: If samples are especially long, it can make sense to limit the number of chars
            per sample that will be printed out.
        """
        self.lemmatize = lemmatize
        self.valid_comparison_methods = ["exact"]
        self.valid_length_methods = ["char", "whitespace", "token"]

        self.print_cutoff_length = print_cutoff_length

        # Automatically load a model if none is passed.
        if processor is None:
            if lang is None:
                raise ValueError("Either a language model (`processor`) or a language code (`lang`) must be specified!")
            else:
                # TODO: Technically a redundant call to interpret_lang_code, since get_nlp_model will check again.
                lang_code = interpret_lang_code(lang)
                self.lang_code = lang_code
            self.processor = get_nlp_model("sm", lang=lang_code)
        else:
            self.processor = processor

    def analyze_dataset(self,
                        reference_text_column_name: str,
                        summary_text_column_name: str,
                        train_set: Optional[Union[List, Dataset]] = None,
                        validation_set: Optional[Union[List, Dataset]] = None,
                        test_set: Optional[Union[List, Dataset]] = None,
                        operations: Optional[List[str]] = None,
                        comparison_method: str = "exact") -> None:
        """
        Function that essentially encapsulates all available analysis methods. Will run whichever methods are specified
        on all samples, which uses simple matching approach to map supplied arguments.
        :param reference_text_column_name:
        :param summary_text_column_name:
        :param train_set:
        :param validation_set:
        :param test_set:
        :param operations: List of analysis methods to run. If empty, will run the full suit.
        :param comparison_method:
        :return:
        """
        raise NotImplementedError("Currently this function is not fully implementec!")

        if not (train_set or validation_set or test_set):
            raise ValueError("No data samples provided (all sets were empty)!")

        # TODO: Check whether the passed operations are all valid!
        # for operation in operations:
        #     if operation not in valid_operations:
        #         raise ValueError(f"Could not identify operation '{operation}'!"
        #                          f"The only valid operations currently supported are:\n{valid_operations}")

        if "detect duplicates" in operations:
            self.detect_duplicates(reference_text_column_name, summary_text_column_name, train_set, validation_set,
                                   test_set, comparison_method)
        passed_splits = self.get_passed_splits_with_names(train_set, validation_set, test_set)
        if "identify empty" in operations:
            total_count = 0
            for split, name in passed_splits:
                split_count = 0
                for sample in split:
                    if self.is_either_text_empty(sample[summary_text_column_name], sample[reference_text_column_name]):
                        split_count += 1
                        total_count += 1
                        # TODO: Instead we could print out the entire sample, but this would not work with the cutoff
                        #  length limitation, unless we somehow identify all the sample attributes beforehand!
                        print(sample[reference_text_column_name][:self.print_cutoff_length])
                        print(sample[summary_text_column_name][:self.print_cutoff_length])
                print(f"{split_count} {split} samples have either an empty reference or summary.")
            print(f"{total_count} samples across all splits had an empty reference or summary.")
        if "identify identity" in operations:
            self.find_identity_samples(reference_text_column_name, summary_text_column_name, train_set, validation_set,
                                       test_set, comparison_method)

    def count_ngram_repetitions(self, text: str, n: int = 3) -> Counter:
        """
        Determines the number of repeated ngrams in a particular text (usually applied to generated summaries.
        :param text: Input text that should be analyzed and split.
        :param n: n-gram length which is used to determine repeating n-grams.
        :return: Count of how many repetitions were counted for what n-gram.
        """

        tokens = self._get_tokens(text)
        ngrams = _create_ngrams(tokens, n=n)

        encountered_ngrams = set()
        duplicate_ngrams = []

        # Simply iterate ngrams and see which ones have not been encountered before.
        for ngram in ngrams:
            if ngram in encountered_ngrams:
                duplicate_ngrams.append(ngram)
            else:
                encountered_ngrams.add(ngram)

        return Counter(duplicate_ngrams)

    def lcs_overlap_fraction(self, summary: str, reference: str) -> float:
        """
        Will compute the fraction of tokens in the generated summary that are shared in the longest common subsequence
        with the reference text. Note that this is quite computationally expensive.
        """
        summary_tokens = self._get_tokens(summary)
        reference_tokens = self._get_tokens(reference)

        return _score_lcs(reference_tokens, summary_tokens).precision

    def ngram_overlap_fraction(self, summary: str, reference: str, n: int = 2) -> float:
        """
        Will compute the fraction of ngrams in the generated summary that are shared with the reference text.
        """
        summary_tokens = self._get_tokens(summary)
        reference_tokens = self._get_tokens(reference)

        summary_ngrams = _create_ngrams(summary_tokens, n=n)
        reference_ngrams = _create_ngrams(reference_tokens, n=n)

        return _score_ngrams(reference_ngrams, summary_ngrams).precision

    def novel_ngrams_fraction(self, summary: str, reference: str, n: int = 2) -> float:
        """
        Convenience function, returns the inverse (1 - x) of self.ngram_overlap_fraction
        """
        return 1 - self.ngram_overlap_fraction(summary=summary, reference=reference, n=n)

    @staticmethod
    def is_fully_extractive(summary: str, reference: str) -> bool:
        """
        Determines whether a summary is fully extractive, by checking whether it verbatim appears in the reference.
        This is suitable for quickly evaluating datasets, but should be only used as an approximate metric.
        """

        if summary in reference:
            return True
        else:
            return False

    def is_summary_longer_than_reference(self, summary: str, reference: str, length_metric: str = "char") -> bool:
        """
        Checks whether a particular sample has a longer summary than reference, indicated by a compression ratio <= 1.0.
        This is a frequent sanity check for quality of samples. Also note that we flag samples that are the exact
        same length. This can happen in rare instances, where the samples are the exact same text.
        :param summary: Summary text.
        :param reference: Reference text.
        :param length_metric: Can be either of "char", "whitespace" or "token". The first two are faster approximations,
            and should generally indicate the same result as the slower "token" method, which uses the processor to
            tokenize the input text. This is mainly important for texts where there is a high difference expected
            between "proper" tokenization and approximations.
        :return: Will return a boolean indicating whether the summary is longer than the reference.
        """
        compression_ratio = self.compression_ratio(summary, reference, length_metric)
        if compression_ratio <= 1.0:
            return True
        else:
            return False

    def compression_ratio(self, summary: str, reference: str, length_metric: str = "char") -> float:
        """
        Computes the compression ratio between a given summary and reference text.
        :param summary: Summary text.
        :param reference: Reference text.
        :param length_metric: Can be either of "char", "whitespace" or "token".
        :return: Will return the compression ratio length_metric(reference) / length_metric(summary).
        """
        length_summary = self._get_length(summary, length_metric)
        length_reference = self._get_length(reference, length_metric)
        if length_reference == 0:
            raise ZeroDivisionError("Empty reference text would cause ZeroDivisionError!")
        return length_reference / length_summary

    def _get_length(self, text: str, length_metric: str) -> int:
        """
        Compute the length of a given summary and reference under a particular length metric.
        :param text: Input text
        :param length_metric: Can be either of "char", "whitespace" or "token".
        :return: Returns the lengths of the summary and reference text, respectively.
        """
        if length_metric == "char":
            return len(text)
        elif length_metric == "whitespace":
            return len(text.split(" "))
        elif length_metric == "token":
            return len(self._get_tokens(text))
        else:
            raise ValueError(f"Unexpected length metric passed! Supported length metrics: {self.valid_length_methods}")

    def is_text_too_short(self, text: str, min_length: int, length_metric: str = "char") \
            -> bool:
        """
        Checks whether a particular text satisfies simple length requirements. This should be generally checked with
        different values for references and summaries in the case of summarization datasets.
        :param text: Input text that should be checked.
        :param min_length: Minimum length requirement for the text (in the respective units of `length_method`).
        :param length_metric: Can be either of "char", "whitespace" or "token". The first two are faster approximations,
            and should generally indicate the same result as the slower "token" method, which uses the processor to
            tokenize the input text. This is mainly important for texts where there is a high difference expected
            between "proper" tokenization and approximations.
        :return: Will return a boolean indicating whether the text is too short or not.
        """
        text_length = self._get_length(text, length_metric)

        if text_length >= min_length:
            return False
        else:
            return True

    @staticmethod
    def is_either_text_empty(summary: str, reference: str) -> bool:
        """
        Simple utility to check whether a sample has either an empty summary or reference text, both of which
        invalidate a sample.
        :param summary: Summary text.
        :param reference: Reference text.
        :return: Boolean indicating whether either summary or reference are empty
        """
        # TODO: Propose a clear list of stripped symbols, which also extend to unicode (e.g., \xa0), or HTML (e.g.,
        #  &nbsp;) characters
        if not summary.strip("\n\t ") or not reference.strip("\n\t "):
            return True
        else:
            return False

    def _get_tokens(self, text: str) -> List[str]:
        """
        Helper method that reduces overhead of changing code parts with, for example, lemmatization choices.
        :param text: Input text that should be tokenized.
        :return: List of tokens in the original text.
        """

        if self.lemmatize:
            return [token.lemma_ for token in self.processor(text)]
        else:
            return [token.text for token in self.processor(text)]

    def is_identity_sample(self, summary_text: str, reference_text: str, comparison_method: str = "exact") -> bool:
        """
        Function to determine whether the reference and summary texts are the same.
        :param summary_text: Text of the summary.
        :param reference_text: Text of the reference article.
        :param comparison_method: Which method to use to determine similarity. Currently only supports "exact",
            but future versions could utilize approximate matchers to be more resilient to, e.g., Unicode issues.
        :return: Nothing is returned, but numbers of affected samples are printed to console.
        """
        # FIXME: This currently will also register samples that are simply two empty strings, which is technically
        #  already handled by self.is_either_text_empty().
        if comparison_method not in self.valid_comparison_methods:
            raise ValueError(f"Currently only the following comparison methods are supported: "
                             f"{self.valid_comparison_methods}")

        if comparison_method == "exact":
            if reference_text == summary_text:
                return True
            else:
                return False
        else:
            raise ValueError("Unexpected method encountered!")

    @staticmethod
    def get_char_occurrences(text: Union[List[str], str]) -> Counter:
        """
        Determines the alphabet used (and each character's occurrence) across either a single or multiple texts.
        :param text: Either a single text or a list of strings containing multiple documents.
        :return: Counter with the character occurrence across text.
        """
        if isinstance(text, str):
            return Counter(text)
        elif isinstance(text, list):
            all_occs = Counter()
            for snippet in text:
                all_occs += Counter(snippet)
            return all_occs
        else:
            raise ValueError("Unrecognised type passed! Supports either str or List[str]!")

    @staticmethod
    def get_passed_splits_with_names(train: Union[List, Dataset, None],
                                     validation: Union[List, Dataset, None],
                                     test: Union[List, Dataset, None]) -> List[Tuple]:
        """
        Utility to aggregate only the splits that were actually passed, i.e., will return all splits that are non-empty.
        Maintains the general order of train -> validation -> test, and associates them with those names.
        :return: List of all the present splits in the format (split, split_name)
        """
        passed_splits = []

        # TODO: Verify behavior in test cases with empty list inputs, etc.
        if train:
            passed_splits.append((train, "train"))
        if validation:
            passed_splits.append((validation, "validation"))
        if test:
            passed_splits.append((test, "test"))

        return passed_splits

    # TODO: See whether it makes sense to enable integer-type columns, in case they are passed as Tuples etc.
    def detect_duplicates(self,
                          reference_text_column_name: str,
                          summary_text_column_name: str,
                          train_set: Optional[Union[List, Dataset]] = None,
                          validation_set: Optional[Union[List, Dataset]] = None,
                          test_set: Optional[Union[List, Dataset]] = None,
                          comparison_method: str = "exact") -> None:
        """
        Method to spot duplicates in datasets. Depending on the size of datasets passed, this method may consume
        a significant amount of main memory. Currently, no disk streaming is supported.
        :param reference_text_column_name: Name of the column that contains the reference text, for a sample.
            This assumes that each sample is structured as a Dict.
        :param summary_text_column_name: Name of the column that contains the summary text.
        :param train_set: List of samples, or split of Huggingface dataset, containing the training samples.
            It is assumed that this is the largest split.
        :param validation_set: List of samples, or split of Huggingface dataset, containing validation samples.
        :param test_set: List of samples, or split of Huggingface dataset, containing test samples.
        :param comparison_method: Which method to use to determine similarity. Currently only supports "exact",
            but future versions could utilize approximate matchers to be more resilient to, e.g., Unicode issues.
        :return: Nothing is returned, but numbers of affected samples are printed to console.
        """
        if comparison_method not in self.valid_comparison_methods:
            raise ValueError(f"Currently only the following comparison methods are supported: "
                             f"{self.valid_comparison_methods}")

        self.detect_leakage(reference_text_column_name, summary_text_column_name,
                            train_set, validation_set, test_set, comparison_method)

        # Iterate over all the passed splits (and ONLY those)
        passed_splits = self.get_passed_splits_with_names(train_set, validation_set, test_set)
        for split, name in passed_splits:
            self._detect_intra_leaks(split, reference_text_column_name, summary_text_column_name, name)

    def _detect_intra_leaks(self, split: Union[List, Dataset],
                            reference_column_name: str,
                            summary_column_name: str,
                            split_name: str):
        seen_samples_ref, seen_samples_summary = self._get_seen_samples_dict(split,
                                                                             reference_column_name,
                                                                             summary_column_name)

        for reference_text, occurrences in seen_samples_ref:
            if len(occurrences) > 1:
                print(f"The following reference text occurred {len(occurrences)} times in the {split_name} split:")
                print(f"{reference_text[:self.print_cutoff_length]}")

        for summary_text, occurrences in seen_samples_summary:
            if len(occurrences) > 1:
                print(f"The following summary text occurred {len(occurrences)} times in the {split_name} split:")
                print(f"{summary_text[:self.print_cutoff_length]}")

    def _get_seen_samples_dict(self, dataset: Union[List, Dataset],
                               reference_column_name: str,
                               summary_column_name: str):
        seen_samples_ref = {}
        seen_samples_summary = {}

        for sample in dataset:
            seen_samples_ref = self._add_sample(sample, sample[reference_column_name], seen_samples_ref)
            seen_samples_summary = self._add_sample(sample, sample[summary_column_name], seen_samples_summary)

        return seen_samples_ref, seen_samples_summary

    @staticmethod
    def _add_sample(sample, sample_text, d):
        if sample_text in d.keys():
            d[sample_text].append(sample)
        else:
            d[sample_text] = [sample]

        return d

    def detect_leakage(self,
                       reference_text_column_name: str,
                       summary_text_column_name: str,
                       train_set: Optional[Union[List, Dataset]] = None,
                       validation_set: Optional[Union[List, Dataset]] = None,
                       test_set: Optional[Union[List, Dataset]] = None,
                       comparison_method: str = "exact") -> None:
        """
        :param reference_text_column_name: Name of the column that contains the reference text, for a sample.
            This assumes that each sample is structured as a Dict.
        :param summary_text_column_name: Name of the column that contains the summary text.
            Similar to `detect_duplicates`, however, only concerned with duplicates *across* different splits.
            This means it will NOT detect duplicates that occur within the same set (e.g., two same samples in train).
            Note that this also assumes that all splits have the same naming convention!
        :param train_set: List of samples, or split of Huggingface dataset, containing the training samples.
            It is assumed that this is the largest split.
        :param validation_set: List of samples, or split of Huggingface dataset, containing validation samples.
        :param test_set: List of samples, or split of Huggingface dataset, containing test samples.
        :param comparison_method: Which method to use to determine similarity. Currently only supports "exact",
            but future versions could utilize approximate matchers to be more resilient to, e.g., Unicode issues.
        :return: Number of affected samples is printed to console.
        """
        if comparison_method not in self.valid_comparison_methods:
            raise ValueError(f"Currently only the following comparison methods are supported: "
                             f"{self.valid_comparison_methods}")

        passed_splits = self.get_passed_splits_with_names(train_set, validation_set, test_set)
        # If only one (or no) split is provided, then no leakage can occur.
        if len(passed_splits) < 2:
            print("No inter-set leaks were detected, since less than two splits were provided.")
            return None

        # Since we don't care about intra-set duplicates, this simplifies to set analysis of occurring texts.
        seen_ref_samples = []
        seen_summ_samples = []

        for split, name in passed_splits:
            seen_ref_samples.append(set([sample[reference_text_column_name] for sample in split]))
            seen_summ_samples.append(set([sample[summary_text_column_name] for sample in train_set]))

        # Now simply cross-check which elements occur in both.
        reference_leakage = 0
        summary_leakage = 0
        for idx, (_, name) in enumerate(passed_splits):
            for ref_text in seen_ref_samples[idx]:
                # Essentially iterate over the other splits and check containment.
                for other_idx, other_split_ref_texts in enumerate(seen_ref_samples[idx+1:], start=idx+1):
                    if ref_text in other_split_ref_texts:
                        reference_leakage += 1
                        print(f"Reference leakage between {name} and {passed_splits[other_idx][1]} set spotted:")
                        print(f"{ref_text[:self.print_cutoff_length]}")

            # TODO: See if this code duplication can be eliminated!
            # Do the same, but for the summary texts
            for summ_text in seen_summ_samples[idx]:
                for other_idx, other_split_summ_texts in enumerate(seen_summ_samples[idx+1:], start=idx+1):
                    if summ_text in other_split_summ_texts:
                        summary_leakage += 1
                        print(f"Reference leakage between {name} and {passed_splits[other_idx][1]} set spotted:")
                        print(f"{summ_text[:self.print_cutoff_length]}")

        print(f"A total of {reference_leakage} reference text leaks and {summary_leakage} summary leaks were found.")

