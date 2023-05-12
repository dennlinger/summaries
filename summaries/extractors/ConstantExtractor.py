"""
Extractor type that assigns a constant score to each segment for testing purposes.
"""

from ..document import Segment
from .ExtractorBase import Extractor


class ConstantExtractor(Extractor):

    def __init__(self):
        super(ConstantExtractor, self).__init__()

    def rank(self, segments: list[Segment], constant_score: float = 1.0) -> None:
        """
        Returns a ranking score for a list of individual segments.
        :param segments: Individual text segments.
        :param constant_score
        :return: List of scores for each individual segment.
        """
        for segment in segments:
            segment.relevance_vector.append(constant_score)

    def filter(self, segments: list[Segment], relevance_threshold: float = 0.0) -> list[Segment]:
        """
        Similar to .rank(), but instead filters out irrelevant passages directly and returns a set of Segments.
        :param segments: Individual text segments.
        :param relevance_threshold: Threshold value that is required for documents to be included.
        :return:
        """
        raise NotImplementedError("Filtering with constant values does not make sense!")