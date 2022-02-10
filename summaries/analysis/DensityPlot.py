"""
A visual analysis tool to understand the distribution of source positionings for extractive and abstractive
summarization systems (and data sources).
"""

from typing import List

import matplotlib.pyplot as plt
from spacy.language import Language

from ..utils import get_nlp_model


class DensityPlot:
    processor: Language

    def __init__(self):
        self.processor = get_nlp_model(size="sm", disable=tuple("ner"), lang="de")

    def plot(self, references: List[str], summaries: List[str]) -> None:
        """
        Generates density plots based on normalized position of reference sentences.
        :param references: List of reference texts
        :param summaries: List of summaries; can be either gold or system summaries
        :return:
        """

        reference_positions = []
        for reference, summary in zip(references, summaries):
            reference_sentences = [sentence.text for sentence in self.processor(reference).sents]
            summary_sentences = [sentence.text for sentence in self.processor(summary).sents]

            for summary_sentence in summary_sentences:
                reference_positions.append(self.find_closest_reference_match(summary_sentence, reference_sentences))

        self.generate_plot(reference_positions)

    @staticmethod
    def find_closest_reference_match(summary_sentence: str, reference_sentences: List[str]) -> float:
        # Check for exact matches first
        if summary_sentence in reference_sentences:
            return reference_sentences.index(summary_sentence) / len(reference_sentences)

        return 0 / len(reference_sentences)

    @staticmethod
    def generate_plot(positions: List[float]):
        plt.hist(positions, bins=50, range=[0.0, 1.0])
