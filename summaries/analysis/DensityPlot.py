"""
A visual analysis tool to understand the distribution of source positionings for extractive and abstractive
summarization systems (and data sources).
"""

from typing import List, Optional, Union

from seaborn import histplot
import matplotlib.pyplot as plt
from spacy.language import Language
from tqdm import tqdm

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
            # TODO: Implement automated processing language
            self.processor = get_nlp_model(size="sm", disable=tuple("ner"), lang="de")

    def plot(self, references: List[List[str]], summaries: List[List[str]], out_fn: Optional[str] = None) -> None:
        """
        Generates density plots based on normalized position of reference sentences.
        :param references: List of reference texts, split into sentences.
        :param summaries: List of summaries; can be either gold or system summaries. Split into sentences.
        :param out_fn: File name where to store the plot.
        :return:
        """

        reference_positions = []
        max_article_length = 0
        for reference_doc, summary_doc in tqdm(zip(references, summaries)):

            # Need maximum length to determine lower bound of bins in histogram
            if len(list(reference_doc)) > max_article_length:
                max_article_length = len(list(reference_doc))

            # Compute likely source-target alignments
            reference_positions.extend(find_closest_reference_matches(summary_doc, reference_doc, 2, self.processor))

        self.generate_plot(reference_positions, min(self.max_num_bins, max_article_length), out_fn)

    @staticmethod
    def generate_plot(positions: List[float], bins: int, out_fn: Union[None, str]):
        plt.figure()
        # Simple hack to only plot a KDE if there are "sufficient" samples available.
        if len(positions) > 10:
            plot_kde = True
        else:
            plot_kde = False
        ax = histplot(positions, bins=bins, stat="probability", kde=plot_kde, binrange=(0, 1))
        ax.set(xlim=(0, 1))
        ax.set(ylim=(0, 0.25))
        if out_fn:
            plt.savefig(out_fn, dpi=100)
        plt.show()

