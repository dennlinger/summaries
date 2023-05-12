"""
Base class for all generative models
"""

from ..document import Segment


class Generator:

    def __init__(self):
        self.model = None

    def generate(self, segments: list[Segment], ex_ante_aspects: dict, ex_post_aspects: dict) -> str:
        raise NotImplementedError("Generator base not implemented!")