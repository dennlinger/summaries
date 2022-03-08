"""
Base class for Aligner methods.
"""
from typing import List, Union

from spacy.language import Language

from ..utils import get_nlp_model, RelevantSentence


class Aligner:
    processor: Language

    def __init__(self):
        self.processor = get_nlp_model(size="sm", disable=("ner",), lang="de")

    def extract_source_sentences(self, summary: Union[List[str], str], reference: Union[List[str], str]) \
            -> List[RelevantSentence]:
        """
        Note that this only works for a *singular* input/output document pair right now, i.e., MDS is not supported!
        Lists of strings would represent a pre-split sentencized input here.
        :param summary: Either a list of pre-split sentences, or a full text containing the gold summary.
        :param reference: Either a list of pre-split sentences, or a full text containing the reference text,
            from which the matching sentences will be extracted.
        :return Returns a list of sentences selected from "reference", where the first sentence is the best match
        for the first target sentence, the second resulting sentence best matches the second gold sentence, etc.
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
        # TODO: Check for ways to extend this to MDS scenarios as well.
        else:
            raise ValueError("Unrecognized combination of input types!")

    def _process_string_inputs(self, summary: str, reference: str) -> List[RelevantSentence]:
        raise NotImplementedError("Method _process_string_inputs has to be implemented by derivative class!")

    def _process_sentencized_inputs(self, summary: List[str], reference: List[str]) -> List[RelevantSentence]:
        raise NotImplementedError("Method _process_sentencized_inputs has to be implemented by derivative class!")
