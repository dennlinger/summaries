"""
A baseline that takes the first three segments (sentences) of an input text as a baseline summary.
There is no clear citation for this method, however, this appears frequently past 2016 with the popularity surge
of the CNN/DailyMail dataset.
"""
from typing import Union, List, Optional

from spacy.language import Language
import spacy

from ..utils import get_nlp_model


def lead3_baseline(reference_text: Union[List[str], str],
                   processor: Optional[Language] = None,
                   lang: Optional[str] = None) -> str:
    """
    Baseline method that extractively summarizes an article by taking the first three sentences.
    :param reference_text: Reference text to be summarized. Can be either of raw reference text (str) or
        pre-split sentences (List[str]).
    :param processor: In the case of raw input text, this is an optional spaCy processor to use for sentence-splitting.
        If not specified, will automatically try and determine a suitable processor based on the `lang` parameter.
    :param lang: If raw input texts are specified and no `processor` is passed, `lang` will be used to determine
        the language of the reference text and load an appropriate spaCy model.
    :return: Summary text consisting of the first three sentences of the reference text (str).
    """

    return leadk_baseline(reference_text, k=3, processor=processor, lang=lang)


def leadk_baseline(reference_text: Union[List[str], str],
                   k: int,
                   processor: Optional[Language] = None,
                   lang: Optional[str] = None) -> str:
    """
    Baseline method that extractively summarizes an article by taking the first k sentences.
    :param reference_text: Reference text to be summarized. Can be either of raw reference text (str) or
        pre-split sentences (List[str]).
    :param k: Number of $k$ first sentences to consider as the summary
    :param processor: In the case of raw input text, this is an optional spaCy processor to use for sentence-splitting.
        If not specified, will automatically try and determine a suitable processor based on the `lang` parameter.
    :param lang: If raw input texts are specified and no `processor` is passed, `lang` will be used to determine
        the language of the reference text and load an appropriate spaCy model.
    :return: Summary text consisting of the first k sentences of the reference text (str).
    """

    if isinstance(reference_text, list):
        # Check for empty inputs...
        if not bool(reference_text):
            raise ValueError("No reference text supplied!")
        # ... as well as empty individual sentences in the input
        for el in reference_text[:k]:
            # TODO: Fix to remove all kinds of whitespaces, not just the specified ones
            if not bool(el.strip("\n\t ")):
                raise ValueError("Empty sentence in the first three supplied sentences detected!")

        return " ".join(reference_text[:3])

    # Catch invalid parameter combinations
    if not processor and not lang:
        raise ValueError("Either specify a language or pass a spaCy object!")

    # Check for empty string-like input
    if not bool(reference_text):
        raise ValueError("Empty reference text supplied!")

    # Automatically load a model if a language is specified
    if lang and not processor:
        processor = get_nlp_model("sm", lang=lang)

    doc = processor(reference_text)
    summary = [sent.text for sent in doc.sents][:k]

    # Check for empty sentences caused by the automated splitter
    for sentence in summary:
        if not sentence.strip("\n\t "):
            raise ValueError("Empty sentence detected in the automatically split text! "
                             "Consider using pre-split sentences instead")

    return " ".join(summary)
