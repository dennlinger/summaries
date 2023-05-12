"""
Simple extractive baseline using YAKE: https://github.com/LIAAD/yake
Notably, YAKE is suited well for several languages, but only supports single document texts.
"""
from typing import List

from yake import KeywordExtractor

from .ExtractorBase import Extractor


class YakeKeywordExtractor(Extractor):
    max_ngram_size: int
    num_topics: int
    yake_extractor: KeywordExtractor

    def __init__(self, num_topics: int, max_ngram_size: int = 3, lang: str = "de"):
        super(YakeKeywordExtractor, self).__init__(lang=lang)

        self.num_topics = num_topics
        self.max_ngram_size = max_ngram_size

        self.yake_extractor = KeywordExtractor(self.lang, self.max_ngram_size)

    def extract_keywords(self, text: str) -> List[str]:
        keywords_with_scores = self.yake_extractor.extract_keywords(text)[:self.num_topics]
        return [keyword for keyword, _ in keywords_with_scores]
