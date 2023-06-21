"""
Probably the most versatile extractor module, which allows to rank segments based on query relevance.
"""

from ..document import Segment
from .ExtractorBase import Extractor


class QueryRelevanceExtractor(Extractor):

    def __init__(self):
        pass

    def rank(self, segments: list[Segment]) -> None:
        pass

    def filter(self, segments: list[Segment], relevance_threshold: float = 0.0) -> list[Segment]:
        pass