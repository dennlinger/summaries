"""
Simple term frequency-based retriever model.
"""

from typing import List
from collections import Counter

from .RetrieverBase import Retriever
from ..index import Index


class FrequencyRetriever(Retriever):

    def __init__(self):
        super(FrequencyRetriever, self).__init__()

    def retrieve(self, query: str, index: Index, limit=3) -> List[str]:
        try:
            # FIXME: Clash between phrase-based query (keyhprase) and term-based index!!!!
            occurrences = index.term_lookup[query]
        except KeyError:
            # FIXME: Handle empty case by returning nothing or similar
            raise KeyError(f"Query {query} not found in index!")

        group_by_doc_id = Counter(occurrences)
        most_relevant_sentence_ids = [doc_id for doc_id, freq in dict(group_by_doc_id.most_common(limit))]
        return [index.sources[idx] for idx in most_relevant_sentence_ids]


