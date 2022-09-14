"""
Collection of statistics-related functions, such as plotting distributions, calculating means, etc.
A large portion of these functions was originally developed by Jiahui Li during her research project on
analyzing English summarization datasets.
"""

from typing import List, Optional, Union, Dict, Tuple
from dataclasses import dataclass
import warnings


from spacy.language import Language
import matplotlib.pyplot as plt
from seaborn import histplot
from datasets import Dataset
from tqdm import tqdm
import numpy as np

from ..utils import find_closest_reference_matches, interpret_lang_code, get_nlp_model


class LengthStats:
    """
    Basic data class for computing basic statistics over a sequence of lengths.
    """
    lengths: List[int]
    mean: Union[float, np.ndarray]
    median: Union[float, np.ndarray]
    std: Union[float, np.ndarray]

    def __init__(self, lengths: List[int]) -> None:
        """
        Computes basic properties and assigns them to the object fields.
        :param lengths: List of the lengths of a sequence of samples.
        """
        self.lengths = lengths
        self.mean = np.mean(self.lengths)
        self.median = np.median(self.lengths)
        self.std = np.std(self.lengths)
        # TODO: Maybe add something like percentiles in the future? How to pass custom attributes?


class Stats:
    """
    Class encapsulating a series of statistics-based analysis methods.
    """
    lemmatize: bool
    length_metric: str
    processor: Language

    def __init__(self,
                 lemmatize: bool = False,
                 length_metric: str = "char",
                 processor: Optional[Language] = None,
                 lang: Optional[str] = None):
        """
        Initializes an Analyzer object.
        :param lemmatize: Boolean to indicate whether lemmas should be used for token-level functions.
        :param processor: Language-specific spaCy model. Either this or `lang` has to be specified.
        :param lang: As an alternative, a language code (e.g., "en" or "de") could be provided, and a spaCy model is
            loaded based on this information.
        """
        self.lemmatize = lemmatize
        self.valid_comparison_methods = ["exact"]
        self.valid_length_methods = ["char", "whitespace", "token"]

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

        if length_metric not in self.valid_length_methods:
            raise ValueError(f"Invalid length metric specified. Valid options are: {self.valid_length_methods}")
        self.length_metric = length_metric

    def density_plot(self, references: List[List[str]], summaries: List[List[str]], out_fn: Optional[str] = None,
                     max_num_bins: int = 100) -> None:
        """
        Generates density plots based on normalized position of reference sentences. A density
        :param references: List of reference texts, split into sentences.
        :param summaries: List of summaries; can be either gold or system summaries. Split into sentences.
        :param out_fn: File name where to store the plot. Will only store the plot if `out_fn` is provided.
        :param max_num_bins: Maximum number of bins to use in the resulting plot.
        :return: No output, but resulting plot will be saved to disk if `out_fn` is specified.
        """

        reference_positions = []
        max_article_length = 0
        for reference_doc, summary_doc in tqdm(zip(references, summaries)):

            # Need maximum length to determine lower bound of bins in histogram
            if len(list(reference_doc)) > max_article_length:
                max_article_length = len(list(reference_doc))

            # Compute likely source-target alignments
            reference_positions.extend(find_closest_reference_matches(summary_doc, reference_doc, 2,
                                                                      processor=self.processor,
                                                                      lemmatize=self.lemmatize,
                                                                      optimization_attribute="fmeasure"))

        self._generate_density_histogram(reference_positions, min(max_num_bins, max_article_length), out_fn)

    # TODO: This can maybe be combined with the histogram for lengths?
    @staticmethod
    def _generate_density_histogram(positions: List[float], bins: int, out_fn: Optional[str] = None):
        """
        Auxiliary wrapper function for the density plots
        """
        plt.figure()
        # Simple hack to only plot a KDE if there are "sufficient" samples available.
        if len(positions) > 10:
            plot_kde = True
        else:
            plot_kde = False
        ax = histplot(positions, bins=bins, stat="probability", kde=plot_kde, binrange=(0, 1))
        ax.set(xlim=(0, 1))
        ax.set(ylim=(0, 0.25))
        if out_fn:
            plt.savefig(out_fn, dpi=200)
        plt.show()

    def compute_length_statistics(self,
                                  summary_column: str,
                                  reference_column: str,
                                  train_split: Optional[Union[List[Dict], Dataset]] = None,
                                  validation_split: Optional[Union[List[Dict], Dataset]] = None,
                                  test_split: Optional[Union[List[Dict], Dataset]] = None,
                                  aggregate_over_splits: bool = False) -> Dict:
        """
        Function that computes central metrics, including mean, median, etc.
        :param summary_column: Name of the column that contains the summary text
        :param reference_column: Name of the column that contains the summary text.
        :param train_split: Training split of the data.
        :param validation_split: Validation split of the data.
        :param test_split: Test split of the data.
        :param aggregate_over_splits: If enabled, will combine samples across all splits instead of computing separate
            stats for each provided split.
        :return: Dictionary with the stats for different (or aggregated) splits.
        """
        passed_splits = self.get_passed_splits_with_names(train_split, validation_split, test_split)
        results = {}

        # If results should be kept for splits, separate them out
        if not aggregate_over_splits:
            for split, split_name in passed_splits:

                # FIXME: This pre-computation of tokens comes at the cost of more memory. Might be an option to enable
                #  in future versions to let the user decide.
                reference_texts = [sample[reference_column] for sample in split]
                summary_texts = [sample[summary_column] for sample in split]

                reference_stats = LengthStats(self._get_lengths(reference_texts))
                summary_stats = LengthStats(self._get_lengths(summary_texts))

                results[split_name] = {"reference": reference_stats, "summary": summary_stats}
        # Otherwise first merge samples and then compute stats
        else:
            all_references = []
            all_summaries = []
            for split, split_name in passed_splits:
                all_references.extend([sample[reference_column] for sample in split])
                all_summaries.extend([sample[summary_column] for sample in split])
            reference_stats = LengthStats(self._get_lengths(all_references))
            summary_stats = LengthStats(self._get_lengths(all_summaries))

            results["aggregated"] = {"reference": reference_stats, "summary": summary_stats}

        return results

    def _get_lengths(self, samples: List[str]) -> List[int]:
        if self.length_metric == "char":
            return [len(sample) for sample in samples]
        elif self.length_metric == "whitespace":
            return [len(sample.split(" ")) for sample in samples]
        elif self.length_metric == "token":
            return [len(sample) for sample in self._get_spacy_tokens_parallel(samples)]

    def _get_spacy_tokens_parallel(self, samples: List[str], num_workers: int = 8) -> List:
        # FIXME: This might break for exceptionally large number of samples?
        return list(tqdm(self.processor.pipe(samples, n_process=num_workers), total=len(samples)))

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

    # TODO: Potentially make lengths to Union[List[str], LengthStats], since we don't need anything besides lengths.
    @staticmethod
    def plot_lengths_histogram(lengths: LengthStats, bins: int = 100, out_fn: Optional[str] = None) -> None:
        """
        In combination with compute_length_statistics(), this function then takes the result and plots
        :param lengths: Length statistics of a dataset.
        :param bins: Number of bins for the histogram
        :param out_fn: If specified, will store the figure in the specified file path
        :return: None, but will plot
        """
        plt.figure()
        # FIXME: Potentially change `stat` to something else?
        #  Also, binrange is currently incorrectly set.
        raise NotImplementedError("Currently binrange is not set correctly")
        ax = histplot(lengths, bins=bins, stat="probability", kde=False, binrange=(0, 1))

        # Heuristic to determine cutoff along x-axis for extremely long tails
        if max(lengths.lengths) > lengths.mean + 2 * lengths.std:
            ax.set(xlim=(0, lengths.mean + 2 * lengths.std))
        else:
            ax.set(xlim=(0, max(lengths.lengths) + 2))
        # TODO: How to automatically detect whether the maximum is larger?
        ax.set(ylim=(0, 0.25))
        if out_fn:
            plt.savefig(out_fn, dpi=200)
        plt.show()
