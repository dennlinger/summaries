"""
Retrieval module based on (mainly German) DPR models.
"""
from typing import List, Union

import numpy as np
from transformers import Pipeline, pipeline
from sentence_transformers.util import dot_score

from .RetrieverBase import Retriever
from ..index import Index


class DPRRetriever(Retriever):
    processor: Pipeline
    query_processor: Pipeline

    def __init__(self, query_encoder: Union[Pipeline, str], context_encoder: Union[Pipeline, str]):
        if isinstance(context_encoder, str):
            context_encoder = pipeline(task="feature-extraction", model=context_encoder)
        super(DPRRetriever, self).__init__(processor=context_encoder)

        # same thing for query encoder, which will be used in slightly different context
        self.query_processor = query_encoder

    def retrieve(self, query: str, index: Index, limit: int = 3) -> List[int]:
        """
        Returns top-k sentences that maximize dot product of the two sentence embeddings
        """
        query_repr = self.encoded_query(query)
        similarities = dot_score(query_repr, index.encoded_documents)
        central_indices = np.argsort(similarities)[-limit:]

        return central_indices

    def encoded_query(self, text: str) -> np.array:
        # Pipelines return nested lists, which we have to re-convert
        return np.array(self.query_processor(text)[0])



