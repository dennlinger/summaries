"""
A visual analysis tool to understand the distribution of source positionings for extractive and abstractive
summarization systems (and data sources).
"""

from typing import List, Optional

from seaborn import histplot
import matplotlib.pyplot as plt
from spacy.language import Language

from ..utils import get_nlp_model, find_closest_reference_matches


class DensityPlot:
    processor: Language
    max_num_bins: int

    def __init__(self, max_num_bins: int = 100, processor: Optional[Language] = None):
        self.max_num_bins = max_num_bins
        # Load default language model if none was specified
        if processor:
            self.processor = processor
        else:
            self.processor = get_nlp_model(size="sm", disable=tuple("ner"), lang="de")

    def plot(self, references: List[List[str]], summaries: List[List[str]]) -> None:
        """
        Generates density plots based on normalized position of reference sentences.
        :param references: List of reference texts, split into sentences.
        :param summaries: List of summaries; can be either gold or system summaries. Split into sentences.
        :return:
        """

        reference_positions = []
        max_article_length = 0
        for reference_doc, summary_doc in zip(references, summaries):

            # Need maximum length to determine lower bound of bins in histogram
            if len(list(reference_doc)) > max_article_length:
                max_article_length = len(list(reference_doc))

            # Compute likely source-target alignments
            reference_positions.extend(find_closest_reference_matches(summary_doc, reference_doc, self.processor))

        self.generate_plot(reference_positions, min(self.max_num_bins, max_article_length))

    @staticmethod
    def generate_plot(positions: List[float], bins):
        # Simple hack to only plot a KDE if there are "sufficient" samples available.
        if len(positions) > 10:
            plot_kde = True
        else:
            plot_kde = False
        histplot(positions, bins=bins, stat="probability", kde=plot_kde, binrange=(0, 1))
        plt.show()
