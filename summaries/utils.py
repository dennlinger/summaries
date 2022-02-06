"""
Collection of helper functions.
"""

from functools import lru_cache
from typing import Tuple

import spacy

supported_languages = {"en", "de"}


def interpret_lang_code(lang: str) -> str:
    """
    Helper function to consistently deal with language tags.
    Internally, ISO tags will be used.
    :param lang: (str) language code passed to the system, which has an unspecified context/encoding
    :return: language tag in lower-cased ISO language (without region).
    """
    # Interpret BCP 47 codes by removing country-specific tag and lower-casing
    lang = lang.split("-")[0].lower()
    if lang not in supported_languages:
        raise ValueError("Unrecognized language tag!")

    return lang


@lru_cache(maxsize=4)
def get_nlp_model(size: str, disable: Tuple[str] = tuple(), lang: str = "de"):
    """
    Wrapper function to provide cached loading of spacy models.
    Automatically determines the correct model to load based in input arguments
    :param size: One of "sm", "md", or "lg". Determines which sized model to load.
    :param disable: Argument to disable particular components of the spacy pipeline.
    :param lang: Identifier to determine which language to use.
    :return: Spacy model pipeline.
    """
    if interpret_lang_code(lang) == "de":
        model_identifier = "de_core_news_"
    elif interpret_lang_code(lang) == "en":
        model_identifier = "en_core_web_"
    else:
        raise NotImplementedError("No associated language model for this language defined!")

    if size not in {"sm", "md", "lg"}:
        raise ValueError("Incorrect size argument passed for language model!")
    model_identifier += size

    return spacy.load(model_identifier, disable=disable)
