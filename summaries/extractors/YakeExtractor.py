"""
Simple extractive baseline using YAKE: https://github.com/LIAAD/yake
Notably, YAKE is suited well for several languages, but only supports single document texts.
"""

from yake import KeywordExtractor

from .ExtractorBase import Extractor


class YakeExtractor(Extractor):
    max_ngram_size: int
    yake_extractor: KeywordExtractor

    def __init__(self, num_topics, max_ngram_size=3, lang="de"):
        super(YakeExtractor, self).__init__(num_topics=num_topics, lang=lang)
        self.max_ngram_size = max_ngram_size

        self.yake_extractor = KeywordExtractor(self.lang, self.max_ngram_size)

    def extract_keywords(self, text):
        return self.yake_extractor.extract_keywords(text)[:self.num_topics]
