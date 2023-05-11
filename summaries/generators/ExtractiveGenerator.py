"""
For use cases where the abstractivity of content is not required, we may instead consider an "extractive generator".
This module instead utilizes the ex-ante preferences as a ranking, and generative a less aspect-focused summary.
"""

from .GeneratorBase import Generator


class ExtractiveGenerator(Generator):

    def __init__(self):
        super(ExtractiveGenerator, self).__init__()
