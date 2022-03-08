"""
Generates a list of relevant sentences in the source text that best align with a target summary's individual sentences.
For this purposes, for each sentence a maximizing fragment (according to ROUGE-2) is chosen from the input text.
Currently only works for SDS cases (i.e., single reference text).
"""
from typing import List, Union

from spacy.language import Language
from rouge_score.rouge_scorer import _create_ngrams

from ..utils import max_rouge_2_match, RelevantSentence
from .AlignerBase import Aligner


class Rouge2Aligner(Aligner):

    optimization_attribute: str

    def __init__(self, optimization_attribute: str = "recall"):
        super(Rouge2Aligner, self).__init__()

        self.optimization_attribute = optimization_attribute

    def _process_string_inputs(self, summary: str, reference: str) -> List[RelevantSentence]:
        """
        Method that additionally uses spacy to sentencize content before matching.
        """
        relevant_sentences = []
        summary_doc = self.processor(summary)
        reference_doc = self.processor(reference)

        reference_ngrams = [_create_ngrams([token.lemma_ for token in sentence], n=2)
                            for sentence in reference_doc.sents]
        reference_sentences = [sentence.text for sentence in reference_doc.sents]

        for sentence in summary_doc.sents:
            relevant_sentences.append(max_rouge_2_match(sentence,
                                                        reference_sentences,
                                                        reference_ngrams,
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
            relevant_sentences.append(max_rouge_2_match(sentence,
                                                        reference_sentences,
                                                        reference_ngrams,
                                                        self.optimization_attribute))

        return relevant_sentences


