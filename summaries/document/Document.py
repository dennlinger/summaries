"""
Representation of an input document.
In theory extensible enough to represent both SDS and MDS scenarios.
"""

import warnings
from typing import Optional, Callable, Union
from itertools import count

from spacy.tokens import Span
from spacy.language import Doc


VALID_SEGMENT_CODES = ["document", "paragraph", "sentence"]


class Document:
    document_id: int
    segments: list["Segment"]
    raw_text: str

    # Meta information
    text_lang: str
    dct: Optional[str]   # Document Creation Timestamp
    source: Optional[str]

    def __init__(self,
                 doc: Doc,
                 document_id: int,
                 lemmatize: bool = False,
                 segment_level: Optional[str] = None,
                 custom_splitting_fn: Optional[Callable] = None,
                 meta_info: Optional[dict] = None):
        """
        :param doc: The processed document text.
        :param document_id: An integer, uniquely identifying this document.
        :param lemmatize: Whether to store a lemmatized text in the segments
        :param segment_level: A string indicating at which level the document should be parsed.
        :param custom_splitting_fn: Should return a list of Sentences, and a list of Paragraph (if determine_paragraphs
            is set to True).
        :param meta_info: Optional dictionary containing document-level meta information, such as temporal information.
        """
        self.document_id = document_id
        self.raw_text = doc.text

        if segment_level not in VALID_SEGMENT_CODES and custom_splitting_fn is None:
            raise ValueError(f"Supported segment_level codes are: {VALID_SEGMENT_CODES}")
        elif segment_level is None and custom_splitting_fn is None:
            raise ValueError("Either specify a segmentation level, or otherwise pass a custom splitting function.")
        elif segment_level is not None and custom_splitting_fn is not None:
            raise ValueError("Inconclusive choice on segmentation! Either provide segmentation level, "
                             "or a custom splitting function, but not both.")

        if custom_splitting_fn:
            self.segments = custom_splitting_fn(self.raw_text)

        if segment_level == "document":
            self.segments = [Segment(doc, self, lemmatize)]
        elif segment_level == "paragraph":
            raise NotImplementedError("Need to figure out how to split paragraphs within spaCy!")
        elif segment_level == "sentence":
            id_count = count(0)
            self.segments = [Segment(sentence, self, next(id_count), lemmatize) for sentence in doc.sents]

        if meta_info is not None:
            if "dct" in meta_info:
                self.dct = meta_info["dct"]
            if "source" in meta_info:
                self.source = meta_info["source"]

            if len(meta_info.keys()) > 2:
                warnings.warn("Found more than two keys in meta information dictionary. "
                              "Currently only DCT and source fields are parsed.")

    def __iter__(self):
        for segment in self.segments:
            yield segment

    def __len__(self):
        return len(self.segments)


class Segment:
    raw: str
    parent_doc: Document
    segment_id: int
    relevance_vector: list[float]

    time_stamps: Optional[dict]
    lemmatized: Optional[list[str]]

    def __init__(self,
                 spacy_processed_text: Union[list[Span], Span, Doc],
                 parent_doc: Document,
                 segment_id: int,
                 lemmatize: bool = False):

        self.parent_doc = parent_doc
        self.segment_id = segment_id
        self.relevance_vector = []

        # TODO: Figure out a way to process paragraphs?
        # # Differentiate iterator to make processing consistent
        # if isinstance(spacy_processed_text, Span):
        #     spacy_processed_text = [spacy_processed_text]
        # elif isinstance(spacy_processed_text, Doc):
        #     spacy_processed_text = spacy_processed_text.sents

        # TODO: Determine whether saving the spaCy objects could make sense (e.g., for PoS tags etc.)
        # self.raw = " ".join([sent.text for sent in spacy_processed_text])
        self.raw = spacy_processed_text.text

        # TODO: Set flag to determine whether the flattening is even necessary (lemmatization flag)
        if lemmatize:
            self.lemmatized = self._flatten_lemmata(spacy_processed_text)
        else:
            self.lemmatized = None

    def __repr__(self):
        return f"Raw text: {self.raw}\nLemmatized text: {self.lemmatized}\nDocument parent: {self.parent_doc}"

    @staticmethod
    def _flatten_lemmata(text) -> list[str]:
        """
        Formulation is a bit counter-intuitive, but this is essentially a (faster) nested list iteration.
        This is required, because we might have multiple spacy-sentences that are considered a single sentence by
        our post-processing.
        """
        # TODO: Fix processing of individual sentences vs paragraphs!
        # return [token.lemma_ for sent in text for token in sent]
        return [token.lemma_ for token in text]

    @property
    def uuid(self):
        return f"{self.parent_doc.document_id}-{self.segment_id}"


def determine_text_language(text: str) -> str:
    """
    Determines a text's language with some package (TBD).
    """
    # TODO: Probably call langid or something
    raise NotImplementedError("Function to determine language currently not available!")




