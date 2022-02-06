"""
Base class for retriever components.
"""

from ..index import Index


class Retriever:

    def __init__(self):
        pass

    def retrieve(self, query: str, index: Index, limit: int):
        NotImplementedError("Retrieval function not implemented!")
