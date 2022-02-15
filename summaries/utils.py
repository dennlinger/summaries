"""
Collection of helper functions.
"""

from collections import namedtuple, Counter
from functools import lru_cache
from operator import attrgetter
from typing import Tuple, List

from rouge_score.rouge_scorer import _create_ngrams, _score_ngrams
from spacy.language import Doc
from spacy.tokens import Span
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


class RelevantSentence(namedtuple("RelevantSentence", ["sentence", "recall", "relative_position"])):
    """
    Wrapper class for named tuple of results, including recall scores and relative position
    """


def print_relevant_sentences(relevant_sentences: List[RelevantSentence], ordered: bool = False) -> None:
    """
    Convenience wrapper to print extracted sentences.
    :param relevant_sentences: List of extracted RelevantSentence objects
    :param ordered: If true, sort the relevant sentences by their relative position first.
    :return: None
    """
    if ordered:
        relevant_sentences = sorted(relevant_sentences, key=attrgetter("relative_position"))
    for sent in relevant_sentences:
        print(sent.sentence)


def find_closest_reference_matches(summary_doc: Doc, reference_doc: Doc) -> List[float]:
    """
    Aggregate function trying to first find an exact match in a reference document, and otherwise
    resorting to ROUGE-2 maximization for finding a related sentence.
    """

    # Note that the actual position is an *open interval* of [0, 1), because we divide by len(reference_sentences)
    relative_positions = []

    reference_sentences = [sentence.text for sentence in reference_doc.sents]
    reference_ngrams = [_create_ngrams([token.lemma_ for token in sentence], n=2) for sentence in reference_doc.sents]
    for summary_sentence in summary_doc.sents:
        # Check for exact matches first (extractive summary)
        if summary_sentence.text in reference_sentences:
            relative_positions.append(reference_sentences.index(summary_sentence.text) / len(reference_sentences))
        else:
            relative_positions.append(
                max_rouge_2_match(summary_sentence, reference_sentences, reference_ngrams).relative_position
            )

    return relative_positions


def max_rouge_2_match(target_sentence: Span, source_sentences: List[str], source_text_ngrams: List[Counter]) -> \
        RelevantSentence:
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
    for idx, source_ngrams in enumerate(source_text_ngrams):
        score = _score_ngrams(target_ngrams=target_ngrams, prediction_ngrams=source_ngrams)

        if score.recall > max_score:
            max_score = score.recall
            max_index = idx

    if max_score < 0 or max_index < 0:
        raise ValueError("No sentence score has been computed!")
    else:
        return RelevantSentence(source_sentences[max_index], max_score, max_index / len(source_sentences))
