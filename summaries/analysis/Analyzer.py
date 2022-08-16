"""
A collection of different functions for analyzing summaries (both reference texts and generated ones).
This can either be done through visual inspection of diversity (Density Plots),
analyzing the textual coherence of texts (finding repetitions),
or simply checking for the amount of overlap with a reference text.
"""
from typing import Optional, List, Union
from collections import Counter
import warnings

from rouge_score.rouge_scorer import _create_ngrams, _score_lcs, _score_ngrams
from spacy.language import Language
import matplotlib.pyplot as plt
from seaborn import histplot
from tqdm import tqdm

from ..utils import get_nlp_model, interpret_lang_code, find_closest_reference_matches


class Analyzer:
    processor: Optional[Language]
    lang_code: Optional[str]
    lemmatize: bool

    def __init__(self, lemmatize: bool = False, processor: Optional[Language] = None, lang: Optional[str] = None):
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

        self.lemmatize = lemmatize

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

            # FIXME: Currently does not respect the self.lemmatize option!
            warnings.warn("`density_plot` currently does not respect the choice of `Analyzer.lemmatize`"
                          "and will always lemmatize!")
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
        Checks whether a particular sample has a longer summary than reference, indicating a compression ratio <= 1.0.
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

        valid_methods = ["char", "whitespace", "token"]
        if length_metric not in valid_methods:
            raise ValueError(f"length_metric should be either of {valid_methods}.")

        if length_metric == "char":
            length_summary = len(summary)
            length_reference = len(reference)
        elif length_metric == "whitespace":
            length_summary = len(summary.split(" "))
            length_reference = len(reference.split(" "))
        elif length_metric == "token":
            length_summary = len(self._get_tokens(summary))
            length_reference = len(self._get_tokens(reference))
        else:
            raise ValueError("Unexpected length metric passed!")

        if length_summary >= length_reference:
            return True
        else:
            return False

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

