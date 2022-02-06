"""
Index class with optimized structure for in-memory retrieval.
TODO: Figure out how efficiently such an approach scales to larger source data sets.
"""

from typing import List, Dict

from ..utils import get_nlp_model


class Index:
    term_lookup: Dict[str, List[int]]
    doc_lookup: Dict[int, List[str]]
    sources: List[str]

    def __init__(self, sources: List[str], processor):
        self.sources = sources
        self.doc_lookup = {idx: [token.text for token in processor(sentence)] for idx, sentence in enumerate(sources)}

        # FIXME: Check tokenization method for faster processing
        self.build_term_lookup()

    def build_term_lookup(self):
        self.term_lookup = {}

        for idx, doc in self.doc_lookup.items():
            for token in doc:
                occurrences = self.term_lookup.get(token, list())
                occurrences.append(idx)
                self.term_lookup[token] = occurrences
