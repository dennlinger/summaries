"""
Base class for template of an extractor class.
"""

from typing import List

from ..utils import interpret_lang_code


class Extractor:
    topics: List[str]
    num_topics: int
    lang: str

    def __init__(self, num_topics, lang):
        self.topics = []
        self.num_topics = num_topics
        self.lang = interpret_lang_code(lang)

    def extract_keywords(self, text):
        raise NotImplementedError("Keyword extraction not implemented!")
