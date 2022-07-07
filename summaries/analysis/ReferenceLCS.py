"""
This utility allows users to investigate the overlap between a generated system summary, and a corresponding reference
text. Traditionally, only the system and gold *summaries* are compared, however, we recently noticed that often
the text in (abstractive) systems is purely extractive, which is not really helpful.
"""
from typing import Optional

from spacy.language import Language
from rouge_score.rouge_scorer import _score_lcs, _score_ngrams, _create_ngrams

from ..utils import get_nlp_model


class ReferenceLCS:
    processor: Language

    # TODO: Potentially allow the usage of lemma=True/False to allow customization.
    def __init__(self, processor: Optional[Language]) -> None:
        if processor:
            self.processor = processor
        else:
            self.processor = get_nlp_model(size="sm", disable=tuple("ner"), lang="de")

    def lcs_fraction(self, summary: str, reference: str) -> float:
        """
        Will compute the fraction of tokens in the generated summary that are shared in a longest common subsequence
        with the reference text (verbatim copy). Equivalent to the LCS precision of the summary wrt the reference.
        """
        summary_tokens = [token.lemma_ for token in self.processor(summary)]
        reference_tokens = [token.lemma_ for token in self.processor(reference)]

        return _score_lcs(reference_tokens, summary_tokens).precision

    def ngram_fraction(self, summary: str, reference: str, n: int = 2) -> float:
        """
        Will compute the fraction of ngrams in the generated summary that are shared with the reference text.
        This is equivalent to the n-gram precision of the summary wrt the reference text.
        """
        summary_tokens = [token.lemma_ for token in self.processor(summary)]
        reference_tokens = [token.lemma_ for token in self.processor(reference)]

        summary_ngrams = _create_ngrams(summary_tokens, n=n)
        reference_ngrams = _create_ngrams(reference_tokens, n=n)

        return _score_ngrams(reference_ngrams, summary_ngrams).recall


