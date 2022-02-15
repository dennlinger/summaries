"""
Generates a list of relevant sentences in the source text that best align with a target summary's individual sentences.
For this purposes, for each sentence a maximizing fragment (according to ROUGE-2 recall) is chosen from the input text.
Currently only works for SDS cases (i.e., single reference text).
"""
from typing import List

from spacy.language import Language
from rouge_score.rouge_scorer import _create_ngrams

from ..utils import max_rouge_2_match, RelevantSentence, get_nlp_model


class Rouge2Aligner:

    processor: Language

    def __init__(self):
        self.processor = get_nlp_model(size="sm", disable=("ner",), lang="de")

    def extract_source_sentences(self, summary: str, reference: str) -> List[RelevantSentence]:

        relevant_sentences = []
        summary_doc = self.processor(summary)
        reference_doc = self.processor(reference)

        reference_ngrams = [_create_ngrams([token.lemma_ for token in sentence], n=2)
                            for sentence in reference_doc.sents]
        reference_sentences = [sentence.text for sentence in reference_doc.sents]

        for sentence in summary_doc.sents:
            relevant_sentences.append(max_rouge_2_match(sentence, reference_sentences, reference_ngrams))

        return relevant_sentences


