"""
Base class for retriever components.
"""

from spacy.language import Language

from ..index import Index


class Retriever:

    def __init__(self):
        pass

    def retrieve(self, query: str, index: Index, processor: Language, limit: int = 3):
        NotImplementedError("Retrieval function not implemented!")
