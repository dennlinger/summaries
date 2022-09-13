"""
A baseline that takes the first three segments (sentences) of an input text as a baseline summary.
There is no clear citation for this method, however, this appears frequently past 2016 with the popularity surge
of the CNN/DailyMail dataset.
"""
from typing import Union, List, Optional

from spacy.language import Language

from ..utils import get_nlp_model


def lead_3(reference_text: Union[List[str], str],
           processor: Optional[Language] = None,
           lang: Optional[str] = None) -> str:
    """
    Baseline method that extractively summarizes an article by taking the first two sentences.
    :param reference_text: Reference text to be summarized. Can be either of raw reference text (str) or
        pre-split sentences (List[str]).
    :param processor: In the case of raw input text, this is an optional spaCy processor to use for sentence-splitting.
        If not specified, will automatically try and determine a suitable processor based on the `lang` parameter.
    :param lang: If raw input texts are specified and no `processor` is passed, `lang` will be used to determine
        the language of the reference text and load an appropriate spaCy model.
    :return: Summary text consisting of the first three sentences of the reference text (str).
    """

    return lead_k(reference_text, k=3, processor=processor, lang=lang)


def lead_k(reference_text: Union[List[str], str],
           k: int,
           processor: Optional[Language] = None,
           lang: Optional[str] = None) -> str:
    """
    Baseline method that extractively summarizes an article by taking the first two sentences.
    :param reference_text: Reference text to be summarized. Can be either of raw reference text (str) or
        pre-split sentences (List[str]).
    :param k: Number of $k$ first sentences to consider as the summary
    :param processor: In the case of raw input text, this is an optional spaCy processor to use for sentence-splitting.
        If not specified, will automatically try and determine a suitable processor based on the `lang` parameter.
    :param lang: If raw input texts are specified and no `processor` is passed, `lang` will be used to determine
        the language of the reference text and load an appropriate spaCy model.
    :return: Summary text consisting of the first three sentences of the reference text (str).
    """

    # If we have pre-split inputs, the computation is simple
    if isinstance(reference_text, list):
        # Check for empty inputs...
        if not bool(reference_text):
            raise ValueError("No reference text supplied!")
        # ... as well as empty individual sentences in the input
        summary = get_sentences(reference_text, k)

        return " ".join(summary)

    # Catch invalid parameter combinations
    if not processor and not lang:
        raise ValueError("Either specify a language or pass a spaCy object, or pass pre-split sentences!")

    # Check for empty string-like input
    if not bool(reference_text):
        raise ValueError("Empty reference text supplied!")

    # Automatically load a model if a language is specified
    if lang and not processor:
        processor = get_nlp_model("sm", lang=lang)

    doc = processor(reference_text)
    summary = get_sentences([sentence.text for sentence in doc.sents], k)

    return " ".join(summary)


def get_sentences(sentences: List[str], k: int) -> List[str]:
    """
    Utility function that extracts a summary from (potentially poorly) split sentences.
    :param sentences: List strings indicating individual sentences.
    :param k: Number of sentences to choose. If len(sentences) <= k, will return the cleaned version of this text.
    :return: Returns summary consisting of k sentences.
    """
    summary = []
    # Check for empty sentences caused by the automated splitter
    for sentence in sentences:
        # Exit
        if len(summary) >= k:
            break
        # Sanity check to get only valid sentences
        clean_sentence = sentence.strip("\n\t ")
        if not clean_sentence:
            continue
        else:
            summary.append(clean_sentence)
    return summary
