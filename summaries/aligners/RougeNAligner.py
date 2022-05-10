"""
Generates a list of relevant sentences in the source text that best align with a target summary's individual sentences.
For this purposes, for each sentence a maximizing fragment (according to ROUGE-2) is chosen from the input text.
Currently only works for SDS cases (i.e., single reference text).
"""
from typing import List

from rouge_score.rouge_scorer import _create_ngrams

from ..utils import max_rouge_n_match, RelevantSentence
from .AlignerBase import Aligner


class RougeNAligner(Aligner):

    optimization_attribute: str
    n: int

    def __init__(self, n: int = 2, optimization_attribute: str = "recall", lang: str = "de"):
        """
        Initializes a RougeNAligner
        :param n: N-gram length to consider. Default choice in literature is ROUGE-2, although ROUGE-1 can be used, too.
        :param optimization_attribute: Either one of "precision", "recall", or "fmeasure" to optimize for.
        :param lang: Language code for the underlying lemmatizer.
        """
        super(RougeNAligner, self).__init__(lang=lang)

        self.n = n
        if optimization_attribute not in ["precision", "recall", "fmeasure"]:
            raise ValueError(f"optimization_attribute must be either of {optimization_attribute}!")
        self.optimization_attribute = optimization_attribute

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


