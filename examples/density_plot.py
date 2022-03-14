"""
Simple example to visualize a density plot
"""
from summaries.analysis import DensityPlot
from summaries.utils import get_nlp_model

from generate_summary import load_text_file_in_sentences


if __name__ == '__main__':

    example_reference = load_text_file_in_sentences("Aachen_Wiki.txt")
    example_summary = load_text_file_in_sentences("Aachen_Klexikon.txt")
    # Test with single sentence (should be at position 0.89
    # extracted_sentence = "Aufwärts Aachen spielt in der ersten Schachbundesliga."

    plot = DensityPlot()
    plot.plot([example_reference], [example_summary])
