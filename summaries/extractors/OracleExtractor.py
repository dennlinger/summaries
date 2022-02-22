"""
Oracle Extractor allows users to specify a list of given keywords that will be returned.
Helpful to supply previously known keywords, and helpful for verifying retrieval-specific components given a
deterministic behavior of extractors.
"""

from typing import List

from .ExtractorBase import Extractor


class OracleExtractor(Extractor):
    given_keywords: List[str]

    def __init__(self, num_topics: int, lang: str, given_keywords: List[str]):
        super(OracleExtractor, self).__init__(num_topics=num_topics, lang=lang)

        self.given_keywords = given_keywords

        if num_topics > len(self.given_keywords):
            raise ValueError("Number of topics to be returned greater than available keywords!")

    def extract_keywords(self, text: str) -> List[str]:
        return self.given_keywords[:self.num_topics]
