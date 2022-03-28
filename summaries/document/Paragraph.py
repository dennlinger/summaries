"""
Representation of several (distinctly separate) sentences.
"""

from typing import List
from itertools import count


from .Sentence import Sentence


class Paragraph:
    paragraph_id: int
    sentences: List[Sentence]

    # See: https://stackoverflow.com/questions/8628123/counting-instances-of-a-class
    _ids = count(0)

    def __init__(self, sentences: List[Sentence]):
        self.sentences = sentences
        self.paragraph_id = next(self._ids)
