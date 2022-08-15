"""
A collection of different functions for analyzing summaries (both reference texts and generated ones).
This can either be done through visual inspection of diversity (Density Plots),
analyzing the textual coherence of texts (finding repetitions),
or simply checking for the amount of overlap with a reference text.
"""
from typing import Optional, List, Union
from collections import Counter

from rouge_score.rouge_scorer import _create_ngrams, _score_lcs, _score_ngrams
from spacy.language import Language
import matplotlib.pyplot as plt
from seaborn import histplot
from tqdm import tqdm

from ..utils import get_nlp_model, interpret_lang_code, find_closest_reference_matches


class Analyzer:
    processor: Optional[Language]
    lang_code: Optional[str]

    def __init__(self, processor: Optional[Language] = None, lang: Optional[str] = None):
        # Automatically load a model if none is passed.
        if processor is None:
            if lang is None:
                raise ValueError("Either a language model (`processor`) or a language code (`lang`) must be specified!")
            else:
                lang_code = interpret_lang_code(lang)
                self.lang_code = lang_code
            self.processor = get_nlp_model("sm", lang=lang_code)
        else:
            self.processor = processor

    def density_plot(self, references: List[List[str]], summaries: List[List[str]], out_fn: Optional[str] = None,
                     max_num_bins: int = 100) -> None:
        """
        Generates density plots based on normalized position of reference sentences. A density
        :param references: List of reference texts, split into sentences.
        :param summaries: List of summaries; can be either gold or system summaries. Split into sentences.
        :param out_fn: File name where to store the plot.
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
            reference_positions.extend(find_closest_reference_matches(summary_doc, reference_doc, 2, self.processor))

        self._generate_plot(reference_positions, min(max_num_bins, max_article_length), out_fn)

    @staticmethod
    def _generate_plot(positions: List[float], bins: int, out_fn: Union[None, str]):
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

    def count_ngram_repetitions(self, text: str, n: int = 3) -> Counter:
        """
        Determines the number of repeated ngrams in a particular text (usually applied to generated summaries.
        :param text: Input text that should be analyzed and split.
        :param n: n-gram length which is used to determine repeating n-grams.
        :param processor: spaCy module which will be used for tokenization.
        :param lang: As an alternative to a processor, it is sufficient to pass a language code.
        :return: Count of how many repetitions were counted for what n-gram.
        """

        tokens = [token.text for token in self.processor(text)]
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
        Will compute the fraction of tokens in the generated summary that are shared in a longest common subsequence
        with the reference text (verbatim copy).
        """
        summary_tokens = [token.lemma_ for token in self.processor(summary)]
        reference_tokens = [token.lemma_ for token in self.processor(reference)]

        # TODO: Verify that this is actually equivalent to the recall, and not precision.
        return _score_lcs(reference_tokens, summary_tokens).precision

    def ngram_overlap_fraction(self, summary: str, reference: str, n: int = 2) -> float:
        """
        Will compute the fraction of ngrams in the generated summary that are shared with the reference text.
        """
        summary_tokens = [token.lemma_ for token in self.processor(summary)]
        reference_tokens = [token.lemma_ for token in self.processor(reference)]

        summary_ngrams = _create_ngrams(summary_tokens, n=n)
        reference_ngrams = _create_ngrams(reference_tokens, n=n)

        # TODO: Verify that this is actually equivalent to the recall, and not precision.
        return _score_ngrams(reference_ngrams, summary_ngrams).recall

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
