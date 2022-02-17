"""
Base class file, containing the holistic pipeline.
"""

from typing import Union, List

from spacy.language import Language

from .extractors import Extractor, YakeExtractor
from .retrievers import Retriever, FrequencyRetriever
from .index import Index
from .utils import get_nlp_model


class AspectSummarizer:

    extractor: Extractor
    retriever: Retriever
    index: Index
    processor: Language

    def __init__(self, extractor: Union[str, Extractor] = "yake", retriever: Union[str, Retriever] = "frequency"):
        # TODO: Enable passing of config files with attributes
        self.extractor = self._assign_extractor(extractor)
        self.retriever = self._assign_retriever(retriever)
        # FIXME: Proper loading of model
        self.processor = get_nlp_model("sm", disable=tuple("ner"), lang="de")

    def _assign_extractor(self, extractor: Union[str, Extractor]) -> Extractor:
        """
        Determine whether an existing Extractor object was passed, or otherwise create appropriate Extractor object.
        :param extractor: Either existing extractor object or string identifier for a model.
        :return: valid Extractor object
        """
        if isinstance(extractor, Extractor):
            return extractor
        elif extractor == "yake":
            return YakeExtractor(num_topics=5, max_ngram_size=3, lang="de")
        else:
            raise NotImplementedError("Extractor component not yet supported!")

    def _assign_retriever(self, retriever: Union[str, Retriever]) -> Retriever:
        """
        Similar to _assign_extractor, assigns the correct retriever object based on input
        :param retriever: Object representing an existing retriever or string identifier for a model.
        :return: valid Retriever object
        """
        if isinstance(retriever, Retriever):
            return retriever
        elif retriever == "frequency":
            return FrequencyRetriever()
        else:
            raise NotImplementedError("Retriever component not yet supported!")

    def summarize(self, source_text: Union[str, List[str]]):
        """
        Extractively summarizes documents based on a simple topic-aggregation strategy.
        :param source_text: Document text(s) that should be summarized.
        :return: Aspect-based summary of the text
        """
        topics = self.extractor.extract_keywords(source_text)
        self.build_index(source_text)
        summary_sentence_ids = []
        for topic, score in topics:
            summary_sentence_ids.extend(self.retriever.retrieve(topic, self.index, self.processor))
        # Heuristic to order the chosen sentence in order of occurrence (i.e., our assigned ID)
        summary_sentence_ids = sorted(set(summary_sentence_ids))
        summary_sentences = [self.index.doc_lookup[idx] for idx in summary_sentence_ids]
        return "\n".join(summary_sentences)

    def build_index(self, sources: Union[str, List[str]]):
        """
        Depending on single or multiple documents, will build an inverted index based on "sentence documents",
        i.e., each sentence representing a different document.
        :param sources: Single or multiple documents from which to summarize
        :return: Sentence-based Index
        """
        if isinstance(sources, str):
            sources = [sources]
        else:
            # TODO: Currently our indexing strategy of assigning a u_id doesn't work with multiple document.
            #  If we want to utilize ordering on the final ids, then this implicitly forces a document order,
            #  which might be unwanted.
            raise NotImplementedError("Multi-document summarization currently not supported!")
        sources = self.split_into_sentence_documents(sources)
        self.index = Index(sources, self.processor)

    def split_into_sentence_documents(self, source_documents: List[str]) -> List[str]:
        sentences = []
        for doc_text in source_documents:
            sentences.extend([sentence.text for sentence in self.processor(doc_text).sents])
        return sentences

