"""
Base class for all generative models
"""


class Generator:

    def __init__(self):
        self.model = None

    def generate(self, text: str) -> str:
        raise NotImplementedError("Generator base not implemented!")