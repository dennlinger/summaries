"""
Base class for retriever components.
"""
from typing import Union

from spacy.language import Language
from transformers import Pipeline

from ..index import Index


class Retriever:
    processor: Union[Language, Pipeline]

    def __init__(self, processor: Union[Language, Pipeline]):
        self.processor = processor

    def retrieve(self, query: str, index: Index, limit: int = 3):
        NotImplementedError("Retrieval function not implemented!")
