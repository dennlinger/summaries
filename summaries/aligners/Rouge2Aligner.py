"""
Generates a list of relevant sentences in the source text that best align with a target summary's individual sentences.
For this purposes, for each sentence a maximizing fragment (according to one ROUGE-2 metric) is chosen from the input text.
Currently only works for SDS cases (i.e., single reference text).
"""
from typing import List, Union

from spacy.language import Language
from rouge_score.rouge_scorer import _create_ngrams

from ..utils import max_rouge_2_match, RelevantSentence, get_nlp_model


class Rouge2Aligner:

    processor: Language
    optimization_attribute: str

    def __init__(self, optimization_attribute: str = "recall"):
        self.processor = get_nlp_model(size="sm", disable=("ner",), lang="de")
        self.optimization_attribute = optimization_attribute

    def extract_source_sentences(self, summary: Union[List[str], str], reference: Union[List[str], str]) \
            -> List[RelevantSentence]:
        """
        Note that this only works for a *singular* input/output document pair right now, i.e., MDS is not supported!
        Lists of strings would represent a pre-split sentencized input here.
        """
        # Differentiate processing based on input type of arguments
        if isinstance(summary, str) and isinstance(reference, list):
            raise ValueError("Assumed to be MDS scenario, which is not yet supported. "
                             "If you were trying to pass references as a list of sentences, "
                             "please do the same for the summary for it to work.")
        elif isinstance(summary, str) and isinstance(reference, str):
            return self._process_string_inputs(summary, reference)
        elif isinstance(summary, list) and isinstance(reference, list):
            return self._process_sentencized_inputs(summary, reference)
        else:
            raise ValueError("Unrecognized combination of input types!")

    def _process_string_inputs(self, summary: str, reference: str):
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

    def _process_sentencized_inputs(self, summary: List[str], reference: List[str]):
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


