"""
Retrieval module based on (mainly German) DPR models.
"""
from typing import List, Union

import numpy as np
from transformers import Pipeline, pipeline, DPRContextEncoder, DPRQuestionEncoder
from sentence_transformers.util import dot_score

from .RetrieverBase import Retriever
from ..index import Index


class DPRRetriever(Retriever):
    processor: Pipeline
    query_processor: Pipeline

    def __init__(self, query_encoder: Union[Pipeline, str], context_encoder: Union[Pipeline, str]):
        if isinstance(context_encoder, str):
            context_encoder_model = DPRContextEncoder.from_pretrained(context_encoder)
            context_encoder = pipeline(task="feature-extraction",
                                       model=context_encoder_model,
                                       tokenizer=context_encoder)
        super(DPRRetriever, self).__init__(processor=context_encoder)

        # same thing for query encoder, which will be used in slightly different context
        if isinstance(query_encoder, str):
            query_encoder_model = DPRQuestionEncoder.from_pretrained(query_encoder)
            query_encoder = pipeline(task="feature-extraction",
                                     model=query_encoder_model,
                                     tokenizer=query_encoder)
        self.query_processor = query_encoder

    def retrieve(self, query: str, index: Index, limit: int = 3) -> List[int]:
        """
        Returns top-k sentences that maximize dot product of the two sentence embeddings
        """
        # Ensure we're not taking more sentences than available
        limit = min(len(index.encoded_documents), limit)

        # Encode query
        query_repr = self.encoded_query(query)

        # Compute similarities
        similarities = dot_score(query_repr, index.encoded_documents).squeeze(dim=0)
        central_indices = np.argsort(similarities)[-limit:]

        return central_indices.tolist()

    def encoded_query(self, text: str) -> np.array:
        # Pipelines return nested lists, which we have to re-convert
        return np.array(self.query_processor(text)[0])



