"""
Representation of an input document.
In theory extensible enough to represent both SDS and MDS scenarios.
"""

from typing import Optional, Callable, List, Tuple, Union

from ..better_split import better_sentence_split
from ..utils import get_nlp_model
from .Paragraph import Paragraph
from .Sentence import Sentence


class Document:
    text: List[Sentence]
    paragraphs: List[Paragraph]
    raw_text: str
    text_lang: str

    def __init__(self, raw_text: str, determine_paragraphs: bool = False, text_lang: Optional[str] = None,
                 custom_splitting_fn: Optional[Callable] = None):
        """
        :param raw_text:
        :param determine_paragraphs:
        :param custom_splitting_fn: Should return a list of Sentences, and a list of Paragraph (if determine_paragraphs
            is set to True).
        """

        self.raw_text = raw_text

        # Automatically determine language of the text, if not provided.
        if not text_lang:
            self.text_lang = determine_text_language(raw_text)
        else:
            self.text_lang = text_lang

        if custom_splitting_fn:
            self.text, self.paragraphs = custom_splitting_fn(raw_text, determine_paragraphs, self.text_lang)
        else:
            self.text, self.paragraphs = sample_splitting_func(raw_text, determine_paragraphs, self.text_lang)


def determine_text_language(text: str) -> str:
    """
    Determines a text's language with some package (TBD).
    """
    # TODO: Probably call langid or something
    return "de"


def sample_splitting_func(text: str, determine_paragraphs: bool, lang: str) \
        -> Tuple[List[Sentence], Optional[List[Paragraph]]]:
    """
    Example of a sample splitting function. In this case, it only takes the default spacy sentence splits.
    """

    nlp = get_nlp_model("sm", lang=lang)

    doc = nlp(text)
    # If desired, splits into separate paragraphs.
    if determine_paragraphs:

        sentences = []
        paragraphs = []

        curr_paragraph = []

        for sentence in doc.sents:
            if sentence.text.startswith("=") or not sentence.text.strip("\n "):
                # Add only if there are sentences/lines in there anyway
                if curr_paragraph:
                    paragraphs.append(Paragraph(curr_paragraph))
                # Reset in either case, because we have just found a boundary condition
                curr_paragraph = []
            else:
                curr_sentence = Sentence(sentence)
                sentences.append(curr_sentence)
                curr_paragraph.append(curr_sentence)

        return sentences, paragraphs

    # Otherwise, just return all sentences that are not empty or headings
    else:
        return [Sentence(sentence) for sentence in doc.sents
                if not sentence.text.startswith("=") and sentence.text.strip("\n ")], None




