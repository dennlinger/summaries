"""
In comparison to the RougeNAligner, the GreedyAligner iteratively adds more sentences from the reference text,
which optimize a set similarity measure (e.g., ROUGE-2). This is then used to determine an optimal length for the
alignment.
"""

from typing import List, Set

from rouge_score.rouge_scorer import _create_ngrams

from ..utils import max_rouge_n_match, RelevantSentence, valid_optimization_attributes
from .AlignerBase import Aligner


class GreedyAligner(Aligner):

    optimization_attribute: str
    n: int
    lemmatize: bool

    def __init__(self, n: int = 2, optimization_attribute: str = "recall", lemmatize: bool = False, lang: str = "de"):
        """
        Initializes a RougeNAligner
        :param n: N-gram length to consider. Default choice in literature is ROUGE-2.
        :param optimization_attribute: Either one of "precision", "recall", or "fmeasure" to optimize for.
            In theory could also support further similarity measures, which are not currently implemented.
        :param lemmatize: Boolean flag indicating whether to use lemmas instead of the raw token texts.
        :param lang: Language code for the underlying tokenizer & optional lemmatizer.
        """
        super(GreedyAligner, self).__init__(lang=lang)

        self.n = n
        if optimization_attribute not in valid_optimization_attributes:
            raise ValueError(f"optimization_attribute must be either of {valid_optimization_attributes}!")
        self.optimization_attribute = optimization_attribute
        self.lemmatize = lemmatize

    def _process_string_inputs(self, summary: str, reference: str) -> List[RelevantSentence]:
        """
        Method that additionally uses spacy to sentencize content before matching.
        """
        relevant_sentences = []
        summary_doc = self.processor(summary)
        reference_doc = self.processor(reference)

        # TODO: Could extend the n-gram creation to actual split words, which would improve for compounds!
        if self.lemmatize:
            reference_ngrams = [_create_ngrams([token.lemma_ for token in sentence], n=self.n)
                                for sentence in reference_doc.sents]
            summary_ngrams = _create_ngrams([token.lemma_ for token in summary_doc], n=self.n)
        else:
            reference_ngrams = [_create_ngrams([token.text for token in sentence], n=self.n)
                                for sentence in reference_doc.sents]
            summary_ngrams = _create_ngrams([token.text for token in summary_doc], n=self.n)

        reference_sentences = [sentence.text for sentence in reference_doc.sents]

        return self._extract_greedy_hypothesis()

    def _process_sentencized_inputs(self, summary: List[str], reference: List[str]) -> List[RelevantSentence]:
        """
        Method to process already sentencized inputs. Despite the additionally processed text,
        this method is slightly slower, since obtaining lemmas still requires processing with spacy.
        """
        relevant_sentences = []
        summary_doc = [self.processor(sentence) for sentence in summary]
        reference_doc = [self.processor(sentence) for sentence in reference]

        if self.lemmatize:
            reference_ngrams = [_create_ngrams([token.lemma_ for token in sentence], n=2)
                                for sentence in reference_doc]
        else:
            reference_ngrams = [_create_ngrams([token.text for token in sentence], n=2)
                                for sentence in reference_doc]

        # This simplifies for already split sentences
        reference_sentences = reference

        # TODO: Implement

        return relevant_sentences

    @staticmethod
    def _extract_greedy_hypothesis(reference_sentences: List[str],
                                   reference_ngrams: List[Set],
                                   summary_ngrams: List[Set],
                                   metric) -> List[str]:

        hypothesis = []
        previous_best_score = 0
        remaining_sentences = reference_sentences

        while remaining_sentences:
            new_best_score = 0
            best_addition = ""
            for sentence, ngrams in zip(reference_sentences, reference_ngrams):
                # TODO: Issue is that we need to re-lemmatize each iteration for this one!
                #  This is because the addition of new sentences technically introduces "boundary n-grams",
                #  which belong to both the previous hypothesis and the current sentence. These are relatively few,
                #  but the pre-split sentences could also be accidentally used by spacy's sentencizer.
                score = metric(f"{' '.join(hypothesis)} {sentence}", summary_ngrams)

                if score > new_best_score:
                    new_best_score = score
                    best_addition = sentence

            # Additional sentence was no longer improving the score; terminal condition
            if new_best_score < previous_best_score:
                return hypothesis
            else:
                # Update hypothesis
                hypothesis.append(best_addition)
                previous_best_score = new_best_score

                # TODO: Also need to remove the corresponding sentence from the ngram list, if we keep it?
                # Also remove this sentence from the candidate set so it cannot be added in future iterations
                remove_added_sentence_from_candidates(new_best_hypothesis_sentence, remaining_reference_sentences)