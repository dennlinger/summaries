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
    processor: Language

    def __init__(self, processor: Language):
        super(FrequencyRetriever, self).__init__(processor)

    def retrieve(self, query: str, index: Index, limit: int = 3) -> List[int]:
        # Handle case from outer call that does not set limit, so reset to default
        if limit <= 0:
            raise ValueError("Negative limit set!")
        query_tokens = self.split_tokens(query)
        occurrences = []
        for token in query_tokens:
            try:
                occurrences.extend(index.term_lookup[token])
            except KeyError:
                # FIXME: Handle empty case by returning nothing or similar
                raise KeyError(f"Query {query} not found in index!")

        group_by_doc_id = Counter(occurrences)
        most_relevant_sentence_ids = [doc_id for doc_id, freq in group_by_doc_id.most_common(limit)]
        return most_relevant_sentence_ids

    def split_tokens(self, query: str) -> List[str]:
        if not query:
            raise ValueError(f"Got empty query!")
        doc = self.processor(query)
        separate_tokens = [token.text for token in doc]

        if separate_tokens == []:
            raise ValueError(f"Processing query {query} returned no split tokens!")
        return separate_tokens
