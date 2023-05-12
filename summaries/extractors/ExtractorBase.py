"""
Base class for template of an extractor class.
"""
from ..utils import interpret_lang_code
from ..document import Segment


class Extractor:

    def __init__(self):
        pass

    def rank(self, segments: list[Segment]) -> None:
        """
        Returns a ranking score for a list of individual segments.
        :param segments: Individual text segments.
        :return: List of scores for each individual segment.
        """
        raise NotImplementedError("Ranking function must be implemented by derivation class!")

    def filter(self, segments: list[Segment], relevance_threshold: float = 0.0) -> list[Segment]:
        """
        Similar to .rank(), but instead filters out irrelevant passages directly and returns a set of Segments.
        :param segments: Individual text segments.
        :param relevance_threshold: Threshold value that is required for documents to be included.
        :return:
        """