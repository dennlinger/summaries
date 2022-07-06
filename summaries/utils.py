"""
Collection of helper functions.
"""

from collections import namedtuple, Counter
from functools import lru_cache
from operator import attrgetter
from typing import Tuple, List, Union, Optional

from rouge_score.rouge_scorer import _create_ngrams, _score_ngrams
from spacy.language import Doc, Language
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
def get_nlp_model(size: str, disable: Tuple[str] = ("ner",), lang: str = "de"):
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


class RelevantSentence(namedtuple("RelevantSentence", ["sentence", "metric", "relative_position"])):
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


def find_closest_reference_matches(summary_doc: Union[Doc, List[str]],
                                   reference_doc: Union[Doc, List[str]],
                                   n: int = 2,
                                   processor: Optional[Language] = None) \
        -> List[float]:
    """
    Aggregate function trying to first find an exact match in a reference document, and otherwise
    resorting to ROUGE-N maximization for finding a related sentence. Will either process a list of sentences,
    or a processed spaCy document.
    """

    # Note that the actual position is an *open interval* of [0, 1), because we divide by len(reference_sentences)
    relative_positions = []

    # Determine processing steps based on input
    if isinstance(summary_doc, Doc):
        reference_sentences = [sentence.text for sentence in reference_doc.sents]
        reference_ngrams = [_create_ngrams([token.lemma_ for token in sentence], n=n)
                            for sentence in reference_doc.sents]
        summary_sentence_iterator = summary_doc.sents
    elif isinstance(summary_doc, list):
        if processor is None:
            raise ValueError("Unspecified processor! Probably forgot to pass a loaded spaCy model to tokenize!")
        reference_sentences = reference_doc
        reference_ngrams = [_create_ngrams([token.lemma_ for token in processor(sentence)], n=n)
                            for sentence in reference_doc]
        summary_sentence_iterator = [processor(sentence) for sentence in summary_doc]
    else:
        raise ValueError("Unrecognized type of summary document object")

    for summary_sentence in summary_sentence_iterator:
        # Check for exact matches first (extractive summary)
        if summary_sentence.text in reference_sentences:
            safe_divisor = max(len(reference_sentences) - 1, 1)
            relative_position = reference_sentences.index(summary_sentence.text) / safe_divisor
            relative_positions.append(relative_position)
        # Otherwise, determine an approximate match with the highest ROUGE-2 overlap.
        else:
            relative_positions.append(
                max_rouge_n_match(summary_sentence, reference_sentences, reference_ngrams, n, "fmeasure").relative_position
            )

    return relative_positions


def max_rouge_n_match(target_sentence: Union[Span, Doc],
                      source_sentences: List[str],
                      source_text_ngrams: List[Counter],
                      n: int = 2,
                      optimization_attribute: str = "recall") \
        -> RelevantSentence:
    """
    Returns the relative position of the closest reference based on maximized ROUGE-2 measure of a single sentence.
    Uses the spacy lemmatizer to increase chance of overlaps.
    Since we're limiting the length to sentences, recall is a decent approximation for overall relation,
    but other attributes can be set, too.
    """

    # Only need to compute the ngrams of the summary sentence once
    target_ngrams = _create_ngrams([token.lemma_ for token in target_sentence], n=n)

    max_score = -1
    max_index = -1

    # Compute ROUGE-2 recall scores for each sentence in the source document to find the most relevant one.
    for idx, source_ngrams in enumerate(source_text_ngrams):
        score = _score_ngrams(target_ngrams=target_ngrams, prediction_ngrams=source_ngrams)

        if getattr(score, optimization_attribute) > max_score:
            max_score = getattr(score, optimization_attribute)
            max_index = idx

    if max_score < 0 or max_index < 0:
        raise ValueError("No sentence score has been computed!")
    else:
        # Avoid ZeroDivisionError by offsetting by one

        relative_position = max_index / safe_divisor(source_sentences)
        return RelevantSentence(source_sentences[max_index], max_score, relative_position)


def safe_divisor(sentences):
    """
    Provides an offset divisor that avoids ZeroDivisonErrors
    """
    return max(len(sentences) - 1, 1)
