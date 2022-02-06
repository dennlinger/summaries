"""
Base class file, containing the wholistic pipeline.
"""

from typing import Union, List

import spacy

from .extractors import Extractor, YakeExtractor
from .retrievers import Retriever, FrequencyRetriever
from .index import Index
from .utils import get_nlp_model


class AspectSummarizer:

    extractor: Extractor
    retriever: Retriever
    index: Index
    processor: spacy.Language.pipeline  # FIXME: probably the wrong type?

    def __init__(self, extractor: Union[str, Extractor] = "yake", retriever: Union[str, Retriever] = "tf"):
        self.extractor = self._assign_extractor(extractor)
        self.retriever = self._assign_retriever(retriever)
        # FIXME: Proper loading of model
        self.processor = get_nlp_model("sm", disable=tuple("ner"), lang="de")

    def _assign_extractor(self, extractor: Union[str, Extractor]) -> Extractor:
        """
        Determine whether an existing Extractor object was passed, or otherwise create appropriate Extractor object.
        :param extractor: Object representing either a extractor or
        :return: valid Extractor object
        """
        if isinstance(extractor, Extractor):
            return extractor
        elif extractor == "yake":
            return YakeExtractor(num_topics=5, max_ngram_size=3, lang="de")
        else:
            raise NotImplementedError("Extractor component not yet supported!")

    def _assign_retriever(self, retriever: Union[str, Retriever]) -> Retriever:
        if isinstance(retriever, Retriever):
            return retriever
        elif retriever == "tf":
            return FrequencyRetriever()
        else:
            raise NotImplementedError("Retriever component not yet supported!")

    def summarize(self, source_text: Union[str, List[str]]):
        """
        Supports SDS (single document summarization) and MDS scenarios in one.
        :param source_text: Document text(s) that should be summarized.
        :return: Aspect-based summary of the text
        """
        topics = self.extractor.extract_keywords(source_text)
        self.build_index(source_text)
        summary_sentences = []
        for topic in topics:
            summary_sentences.append(self.retriever.retrieve(topic, self.index))

        return "\n".join(summary_sentences)

    def build_index(self, sources: Union[str, List[str]]):
        if isinstance(sources, str):
            sources = self.split_sentences(sources)
        self.index = Index(sources, self.processor)

    def split_sentences(self, text: str) -> List[str]:
        sentences = [str(sentence) for sentence in self.processor(text).sents]

        return sentences

