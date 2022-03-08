"""
Similar to other Aligners, this method chooses a 1:1 mapping for each target sentence,
by maximizing a similarity objective. In this case, the similarity is defined through a sentence-transformers model.
"""
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from spacy.language import Language

from ..utils import RelevantSentence
from .AlignerBase import Aligner


class SentenceTransformerAligner(Aligner):
    processor: Language
    model: SentenceTransformer

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        """
        :param model_name: Alternative suggestion: paraphrase-multilingual-mpnet-base-v2
        """
        super(SentenceTransformerAligner, self).__init__()
        # Checks for GPU by default.
        self.model = SentenceTransformer(model_name)

    def _process_string_inputs(self, summary: str, reference: str) -> List[RelevantSentence]:
        """
        For streamlined interface, this simply prepends a sentence splitting step, and then calls the alternative
        method that actually performs the similarity computation.
        """
        split_summary = [sentence.text for sentence in self.processor(summary).sents]
        split_reference = [sentence.text for sentence in self.processor(reference).sents]

        return self._process_sentencized_inputs(split_summary, split_reference)

    def _process_sentencized_inputs(self, summary: List[str], reference: List[str]) -> List[RelevantSentence]:
        """
        Computes the similarities between embeddings of all summary/reference pairs.
        Returns len(summary) sentences, where each sentence represents the most similar reference sentence
        to a respective summary sentence.
        Note that this is not deduplicated or otherwise applying any threshold to the similarity.
        """
        # see https://www.sbert.net/docs/usage/semantic_textual_similarity.html
        # Compute embedding for both lists
        summary_embeddings = self.model.encode(summary, convert_to_tensor=True)
        reference_embeddings = self.model.encode(reference, convert_to_tensor=True)

        # Compute cosine-similarities
        cosine_scores = cos_sim(summary_embeddings, reference_embeddings)

        result_sentences = []
        # Iterating through rows means all similarities for one particular summary sentence
        for row in cosine_scores.detach().numpy():

            # Determine specific values for the most relevant sentence
            # TODO: Maybe we could add a threshold here?
            most_similar_sentence_idx = np.argmax(row)
            relevant_sentence = reference[most_similar_sentence_idx]
            max_score = np.max(row)
            relative_position = most_similar_sentence_idx / len(reference)
            result_sentences.append(RelevantSentence(relevant_sentence, max_score, relative_position))

        return result_sentences





