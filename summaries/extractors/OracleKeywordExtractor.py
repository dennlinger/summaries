"""
Oracle Extractor allows users to specify a list of given keywords that will be returned.
Helpful to supply previously known keywords, and helpful for verifying retrieval-specific components given a
deterministic behavior of extractors.
"""

from typing import List

from .ExtractorBase import Extractor


class OracleKeywordExtractor(Extractor):
    given_keywords: List[str]
    num_topics: int

    def __init__(self, num_topics: int, lang: str, given_keywords: List[str]):
        super(OracleKeywordExtractor, self).__init__(lang=lang)

        self.num_topics = num_topics
        self.given_keywords = given_keywords

        if num_topics > len(self.given_keywords):
            raise ValueError("Number of topics to be returned greater than available keywords!")

    def filter(self):
        raise NotImplementedError("No filter function available for OracleKeywordExtractor yet.")
