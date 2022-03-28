"""
Representation of the smallest considered text unit (sentences),
containing both raw and processed forms for easier access.
"""
from typing import Union, List, Optional

from spacy.tokens import Span


class Sentence:
    raw: str
    lemmatized: List[str]
    paragraph_id: Optional[int]

    def __init__(self, spacy_processed_text: Union[List[Span], Span], paragraph_id: Optional[int] = None):

        # Dummy wrapper to make processing consistent
        if isinstance(spacy_processed_text, Span):
            spacy_processed_text = [spacy_processed_text]

        # TODO: Determine whether saving the spaCy objects could make sense (e.g., for PoS tags etc.)
        self.raw = " ".join([sent.text for sent in spacy_processed_text])
        self.lemmatized = self._flatten_lemmata(spacy_processed_text)

        self.paragraph_id = paragraph_id

    def __repr__(self):
        return f"Raw text: {self.raw}\nLemmatized text: {self.lemmatized}\nParagraph ID: {self.paragraph_id}"

    @staticmethod
    def _flatten_lemmata(text):
        """
        Formulation is a bit counter-intuitive, but this is essentially a (faster) nested list iteration.
        This is required, because we might have multiple spacy-sentences that are considered a single sentence by
        our post-processing.
        """
        return [token.lemma_ for sent in text for token in sent]
