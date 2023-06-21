"""
Extractor type that allows for filtering based on the leading segments in each document.
Can be either utilized with a fixed cutoff count k,
or alternatively a decaying relevance.
"""
import warnings
from typing import Optional, Callable

from ..document import Segment
from .ExtractorBase import Extractor


class LeadExtractor(Extractor):
    fixed_cutoff: Optional[int]
    decayed_relevance_func: Optional[Callable]

    def __init__(self, fixed_cutoff: Optional[int] = None, decayed_relevance_func: Optional[Callable] = None):
        self.fixed_cutoff = fixed_cutoff
        self.decayed_relevance_func = decayed_relevance_func

        if fixed_cutoff is None and decayed_relevance_func is None:
            warnings.warn("Neither cutoff nor decaying relevance function specified. "
                          "Will default to an inverse scaling function.")
            self.decayed_relevance_func = default_inverse_scaling
        elif fixed_cutoff is not None and decayed_relevance_func is not None:
            raise ValueError("Cannot handle simultaneously specified fixed cutoff and decaying relevance.")

        super(LeadExtractor, self).__init__()

    def rank(self, segments: list[Segment]) -> None:
        """
        Returns a ranking score for a list of individual segments.
        :param segments: Individual text segments.
        :return: List of scores for each individual segment.
        """
        for segment in segments:
            # If we have a specified fixed cutoff position, ignore other relevance scores
            if self.fixed_cutoff:
                if segment.segment_id < self.fixed_cutoff:
                    segment.relevance_vector.append(1.0)
                else:
                    segment.relevance_vector.append(0.0)
            # For unspecified cutoffs, append a relevance value according to our function
            elif self.decayed_relevance_func:
                segment.relevance_vector.append(self.decayed_relevance_func(segment.segment_id))

    def filter(self, segments: list[Segment], relevance_threshold: float = 0.0) -> list[Segment]:
        """
        Similar to .rank(), but instead filters out irrelevant passages directly and returns a set of Segments.
        :param segments: Individual text segments.
        :param relevance_threshold: Threshold value that is required for documents to be included.
        :return:
        """

        result = []
        # Filter the segments based on whichever criterion is specified.
        for segment in segments:
            if self.fixed_cutoff:
                if relevance_threshold != 0.0:
                    warnings.warn("relevance_threshold specified, but unused due to fixed cutoff.")
                if segment.segment_id < self.fixed_cutoff:
                    result.append(segment)
            elif self.decayed_relevance_func:
                if self.decayed_relevance_func(segment.segment_id) > relevance_threshold:
                    result.append(segment)

        return result


def default_inverse_scaling(position: int) -> float:
    return 1 / (position + 1)
