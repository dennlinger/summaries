"""
Generates a list of relevant sentences in the source text that best align with a target summary's individual sentences.
For this purposes, for each sentence a maximizing fragment (according to ROUGE-2) is chosen from the input text.
Currently only works for SDS cases (i.e., single reference text).
"""
from copy import deepcopy
from collections import Counter
from typing import List, Union, Tuple, Set

from rouge_score.rouge_scorer import _create_ngrams, _score_ngrams

from ..utils import max_rouge_n_match, RelevantSentence, safe_divisor
from .AlignerBase import Aligner


VALID_ATTRIBUTES = ["precision", "recall", "fmeasure"]


class RougeNAligner(Aligner):

    optimization_attribute: str
    n: int

    def __init__(self, n: int = 2, optimization_attribute: str = "recall", lang: str = "de"):
        """
        Initializes a RougeNAligner, serving as a base class for differing implementations.
        :param n: N-gram length to consider. Default choice in literature is ROUGE-2, although ROUGE-1 can be used, too.
        :param optimization_attribute: Either one of "precision", "recall", or "fmeasure" to optimize for.
        :param lang: Language code for the underlying lemmatizer.
        """
        super(RougeNAligner, self).__init__(lang=lang)

        self.n = n
        if optimization_attribute not in VALID_ATTRIBUTES:
            raise ValueError(f"optimization_attribute must be either of {VALID_ATTRIBUTES}!")
        self.optimization_attribute = optimization_attribute

    def _process_string_inputs(self, summary: str, reference: str) -> List[RelevantSentence]:
        raise NotImplementedError("Please instantiate either a SentenceRougeNAligner or GreedyRougeNAligner")

    def _process_sentencized_inputs(self, summary: List[str], reference: List[str]) -> List[RelevantSentence]:
        raise NotImplementedError("Please instantiate either a SentenceRougeNAligner or GreedyRougeNAligner")


class SentenceRougeNAligner(RougeNAligner):

    def __init__(self, n: int = 2, optimization_attribute: str = "recall", lang: str = "de"):
        """
        Initializes a SentenceRougeNAligner, optimizing for each sentence in the target separately.
        :param n: N-gram length to consider. Default choice in literature is ROUGE-2, although ROUGE-1 can be used, too.
        :param optimization_attribute: Either one of "precision", "recall", or "fmeasure" to optimize for.
        :param lang: Language code for the underlying lemmatizer.
        """
        super(SentenceRougeNAligner, self).__init__(n=n, optimization_attribute=optimization_attribute, lang=lang)

    def _process_string_inputs(self, summary: str, reference: str) -> List[RelevantSentence]:
        """
        Method that additionally uses spacy to sentencize content before matching.
        """
        relevant_sentences = []
        summary_doc = self.processor(summary)
        reference_doc = self.processor(reference)

        # TODO: Could extend the n-gram creation to actual split words, which would improve for compounds!
        reference_ngrams = [_create_ngrams([token.lemma_ for token in sentence], n=self.n)
                            for sentence in reference_doc.sents]
        reference_sentences = [sentence.text for sentence in reference_doc.sents]

        for sentence in summary_doc.sents:
            relevant_sentences.append(max_rouge_n_match(sentence,
                                                        reference_sentences,
                                                        reference_ngrams,
                                                        self.n,
                                                        self.optimization_attribute))

        return relevant_sentences

    def _process_sentencized_inputs(self, summary: List[str], reference: List[str]) -> List[RelevantSentence]:
        """
        Method to process already sentencized inputs. Despite the additionally processed text,
        this method is slightly slower, since obtaining lemmas still requires processing with spacy.
        """
        relevant_sentences = []
        summary_doc = [self.processor(sentence) for sentence in summary]
        reference_doc = [self.processor(sentence) for sentence in reference]

        reference_ngrams = [_create_ngrams([token.lemma_ for token in sentence], n=2)
                            for sentence in reference_doc]
        # This simplifies for already split sentences
        reference_sentences = reference

        for sentence in summary_doc:
            relevant_sentences.append(max_rouge_n_match(sentence,
                                                        reference_sentences,
                                                        reference_ngrams,
                                                        self.n,
                                                        self.optimization_attribute))

        return relevant_sentences


class GreedyRougeNAligner(RougeNAligner):
    # TODO: Implement optimization_attribute as ENUM type?
    def __init__(self, n: int = 2, optimization_attribute: str = "fmeasure", lang: str = "de"):
        """
        Initializes a GreedyRougeNAligner, which considers the full summary when optimizing the metric.
        :param n: N-gram length to consider. Default choice in literature is ROUGE-2, although ROUGE-1 can be used, too.
        :param optimization_attribute: Either one of "precision", "recall", or "fmeasure" to optimize for.
        :param lang: Language code for the underlying lemmatizer.
        """
        super(GreedyRougeNAligner, self).__init__(n=n, optimization_attribute=optimization_attribute, lang=lang)

    def _process_string_inputs(self, summary: str, reference: str) -> List[RelevantSentence]:
        """
        Method that additionally uses spacy to sentencize content before matching.
        """
        relevant_sentences = []
        summary_doc = self.processor(summary)
        reference_doc = self.processor(reference)

        # TODO: Could extend the n-gram creation to actual split words, which would improve for compounds!
        reference_ngrams = [_create_ngrams([token.lemma_ for token in sentence], n=self.n)
                            for sentence in reference_doc.sents]
        reference_sentences = [sentence.text for sentence in reference_doc.sents]

        summary_ngrams = _create_ngrams([token.lemma_ for token in summary_doc], n=self.n)

        # Iteratively add new reference sentences until ROUGE scores are saturated
        current_best_ngrams = Counter()
        previously_added_sentences = set()
        oracle_summary = []
        # Iterate until no further improvement can be found, or we have as many sentences as the reference
        while len(oracle_summary) < len(reference_sentences):
            best_addition_idx, improvement_in_score = find_best_extension_sentence(current_best_ngrams,
                                                                                   reference_ngrams,
                                                                                   summary_ngrams,
                                                                                   previously_added_sentences)
            if best_addition_idx is None:
                break
            else:
                # Update current hypothesis...
                current_best_ngrams.update(reference_ngrams[best_addition_idx])
                # ... and also add sentence to oracle summary
                oracle_summary.append(RelevantSentence(reference_sentences[best_addition_idx],
                                                       improvement_in_score,
                                                       best_addition_idx / safe_divisor(reference_sentences)))

        return relevant_sentences

    def _process_sentencized_inputs(self, summary: List[str], reference: List[str]) -> List[RelevantSentence]:
        """
        Method to process already sentencized inputs. Despite the additionally processed text,
        this method is slightly slower, since obtaining lemmas still requires processing with spacy.
        """
        relevant_sentences = []

        # TODO: Could extend the n-gram creation to actual split words, which would improve for compounds!
        reference_ngrams = [_create_ngrams([token.lemma_ for token in self.processor(sentence)], n=self.n)
                            for sentence in reference]
        reference_sentences = [sentence.text for sentence in reference_doc.sents]
        summary_ngrams = _create_ngrams([token.lemma_ for token in summary_doc], n=self.n)

        # Iteratively add new reference sentences until ROUGE scores are saturated
        current_best_ngrams = Counter()
        previously_added_sentences = set()
        oracle_summary = []
        # Iterate until no further improvement can be found, or we have as many sentences as the reference
        while len(oracle_summary) < len(reference_sentences):
            best_addition_idx, improvement_in_score = find_best_extension_sentence(current_best_ngrams,
                                                                                   reference_ngrams,
                                                                                   summary_ngrams,
                                                                                   previously_added_sentences)
            if best_addition_idx is None:
                break
            else:
                # Update current hypothesis...
                current_best_ngrams.update(reference_ngrams[best_addition_idx])
                # ... and also add sentence to oracle summary
                oracle_summary.append(RelevantSentence(reference_sentences[best_addition_idx],
                                                       improvement_in_score,
                                                       best_addition_idx / safe_divisor(reference_sentences)))

        return relevant_sentences


def find_best_extension_sentence(current_best_ngrams: Counter,
                                 reference_sentence_ngrams: List[Counter],
                                 gold_summary_ngrams: Counter,
                                 previously_added_sentences: Set[int]) -> Tuple[Union[int, None], float]:
    """
    Method to identify the sentence (index) that is the best addition to the current oracle summary.
    The matching criteria is maximizing the currently added ROUGE sentence; if no sentence increases ROUGE scores,
    the return value will be "None".
    :param current_best_ngrams: Counter of ngrams in the current best hypothesis. Will be used as basis for extension.
    :param reference_sentence_ngrams: Sentence-level Counters containing ngram counts for each sentence.
    :param gold_summary_ngrams: The ngrams contained in the gold summary.
    :param previously_added_sentences: A set of indices of previously added sentences. Will skip all sentences that
        are present in this set during the current iteration.
    :return: Index position of the best addition or alternatively None if no improvement is found.
        Also returns the ROUGE improvement of the best addition.
    """
    # Current best hypothesis is the baseline for all future comparisons
    max_pos = None
    base_score = _score_ngrams(gold_summary_ngrams, current_best_ngrams)
    max_score = base_score

    for idx, sentence_ngrams in enumerate(reference_sentence_ngrams):
        # Skip sentences that have been previously considered
        if idx in previously_added_sentences:
            continue

        # Consider the union of our best current approach plus one additional (current) sentence
        hypothesis = deepcopy(current_best_ngrams).update(sentence_ngrams)
        alternative_score = _score_ngrams(gold_summary_ngrams, hypothesis)

        if alternative_score >= max_score:
            max_pos = idx

    # Also return by how much the current hypothesis improved.
    improvement = max_score - base_score
    return max_pos, improvement
