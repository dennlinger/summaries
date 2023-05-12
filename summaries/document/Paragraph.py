"""
Representation of several (distinctly separate) sentences.
"""

from itertools import count


from . import Segment


class Paragraph:
    paragraph_id: int
    sentences: list[Segment]

    # See: https://stackoverflow.com/questions/8628123/counting-instances-of-a-class
    _ids = count(0)

    def __init__(self, sentences: list[Segment], determine_temporal_tags: bool = False):
        self.paragraph_id = next(self._ids)
        self.sentences = sentences

        if determine_temporal_tags:
            # TODO: Call Heideltime here
            pass

        # TODO: Determine whether there is a smarter design for this.
        # Propagate paragraph id to sentences, for easier retrieval later.
        for sentence in sentences:
            sentence.paragraph_id = self.paragraph_id
