"""
Similar to other Aligners, this method chooses a 1:1 mapping for each target sentence,
by maximizing a similarity objective. In this case, the similarity is defined through a sentence-transformers model.
"""
from typing import Union, List

from sentence_transformers import SentenceTransformer
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



