"""
Improved sentence splitting by fixing common issues in (full-text) splitting with spacy.
Some parts taken from my Klexikon processing:
https://github.com/dennlinger/klexikon/blob/master/klexikon/processing/fix_lead_sentences.py
"""

import regex
from typing import List

from .utils import get_nlp_model


def better_sentence_split(text: str, lang: str = "de") -> List[str]:
    """
    Returns a sentence-split list of sentences.
    """

    nlp = get_nlp_model(size="sm", lang=lang)

    spacy_sentences = [sentence.text for sentence in nlp(text).sents]

    better_sentences = []
    likely_incomplete_sentence = False
    curr = ""
    for sentence in spacy_sentences:
        sentence = sentence.strip("\n ")
        # Always merge semicolon sentences
        if sentence.endswith(";") or no_period_symbol(sentence) or year_like_sentence_end(sentence):
            curr += sentence + " "
            likely_incomplete_sentence = True

        # Preferably also merge too short sentences (based on either word or character length)
        elif len(sentence.split(" ")) < 3 or len(sentence) < 10:
            curr += sentence + " "
            likely_incomplete_sentence = True

        # If no further problem has been detected, append current "line"
        elif likely_incomplete_sentence:
            curr += sentence
            better_sentences.append(curr.strip(" "))
            curr = ""
            likely_incomplete_sentence = False
        # instance where just no incomplete sentence has been encountered so far, simply reset
        else:
            better_sentences.append(sentence)
            curr = ""

    # Append last sentence if not finished
    if likely_incomplete_sentence:
        better_sentences.append(curr.strip(" "))

    return better_sentences


def no_period_symbol(sentence):
    return not (sentence.endswith(":") or sentence.endswith(".") or sentence.endswith("?") or sentence.endswith("!")
                or sentence.endswith("\"") or sentence.endswith("'"))


def year_like_sentence_end(sentence):
    regex.search("[0-9]\.$", sentence)
