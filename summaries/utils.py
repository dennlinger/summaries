"""
Collection of helper functions.
"""

from functools import lru_cache
from typing import Tuple, List

import spacy
from spacy.language import Doc
from spacy.tokens import Span
from rouge_score.rouge_scorer import _create_ngrams, _score_ngrams

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


def find_closest_reference_matches(summary_doc: Doc, reference_doc: Doc) -> List[float]:
    """
    Aggregate function trying to first find an exact match in a reference document, and otherwise
    resorting to ROUGE-2 maximization for finding a related sentence.
    """

    relative_positions = []

    reference_sentences = [sentence.text for sentence in reference_doc.sents]
    for summary_sentence in summary_doc.sents:
        # Check for exact matches first (extractive summary)
        if summary_sentence.text in reference_sentences:
            # Note that the actual position can only range between an open interval of [0, len(reference_sentences))
            relative_positions.append(reference_sentences.index(summary_sentence.text) / len(reference_sentences))
        else:
            relative_positions.append(max_rouge_2_match(summary_sentence, reference_doc))

    return relative_positions


# TODO: Should not return fractions here, but rather actual text (or optionally, position!)
#  Best solution would be to write a separate wrapper for the fractional results, and call a general function?
def max_rouge_2_match(target_sentence: Span, source_text: Doc) -> float:
    """
    Returns the relative position of the closest reference based on maximized ROUGE-2 recall of a single sentence.
    Uses the spacy lemmatizer to increase chance of overlaps.
    Since we're limiting the length to sentences, recall is a decent approximation for overall relation.
    """
    if len(target_sentence) <= 3:
        raise ValueError(f"Sentence splitting likely went wrong! Sentence: {target_sentence.text}")

    # Only need to compute the ngrams of the summary sentence once
    target_ngrams = _create_ngrams([token.lemma_ for token in target_sentence], n=2)

    max_score = -1
    max_index = -1

    # Compute ROUGE-2 recall scores for each sentence in the source document to find the most relevant one.
    for idx, sentence in enumerate(source_text.sents):
        source_ngrams = _create_ngrams([token.lemma_ for token in sentence], n=2)

        score = _score_ngrams(target_ngrams=target_ngrams, prediction_ngrams=source_ngrams)

        if score.recall > max_score:
            max_score = score.recall
            max_index = idx

    if max_score < 0 or max_index < 0:
        raise ValueError("No sentence score has been computed!")
    else:
        return max_index / len(list(source_text.sents))

