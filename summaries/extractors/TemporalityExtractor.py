"""
Filters documents by relevance based on temporal mentions available for documents.
"""

from ..document import Segment
from .ExtractorBase import Extractor


class TemporalityExtractor(Extractor):
    def __init__(self):
        pass

    def rank(self, segments: list[Segment]) -> None:
        raise NotImplementedError()

    def filter(self, segments: list[Segment], relevance_threshold: float = 0.0) -> list[Segment]:
        raise NotImplementedError