"""
Base class for template of an extractor class.
"""
from ..utils import interpret_lang_code


class Extractor:
    topics: list[str]
    num_topics: int
    lang: str

    def __init__(self, num_topics: int, lang: str):
        self.topics = []
        self.num_topics = num_topics
        self.lang = interpret_lang_code(lang)

    def extract_keywords(self, text: str) -> list[str]:
        raise NotImplementedError("Keyword extraction not implemented!")
