"""
Simple term frequency-based retriever model.
"""

from typing import List
from collections import Counter

from spacy.language import Language

from .RetrieverBase import Retriever
from ..index import Index


class FrequencyRetriever(Retriever):
    """
    Simple Retriever model that operates on aggregated (non-normalized) term frequency.
    Not intended for main usage, but easy to debug.
    """

    def __init__(self):
        super(FrequencyRetriever, self).__init__()

    def retrieve(self, query: str, index: Index, processor: Language, limit: int = 3) -> List[str]:
        query_tokens = self.split_tokens(query, processor)
        occurrences = []
        for token in query_tokens:
            try:
                occurrences.extend(index.term_lookup[token])
            except KeyError:
                # FIXME: Handle empty case by returning nothing or similar
                raise KeyError(f"Query {query} not found in index!")

        group_by_doc_id = Counter(occurrences)
        most_relevant_sentence_ids = [doc_id for doc_id, freq in dict(group_by_doc_id.most_common(limit))]
        return [index.sources[idx] for idx in most_relevant_sentence_ids]

    @staticmethod
    def split_tokens(query: str, processor: Language) -> List[str]:
        if not query:
            raise ValueError(f"Got empty query!")
        doc = processor(query)
        separate_tokens = [token.text for token in doc]

        if separate_tokens == []:
            raise ValueError(f"Processing query {query} returned no split tokens!")
        return separate_tokens
