"""
Base class for all generative models
"""


class Generator:

    def __init__(self):
        self.model = None

    def generate(self, text) -> str:
        raise NotImplementedError("Generator base not implemented!")