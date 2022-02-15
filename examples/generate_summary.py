"""
Minimal script to generate a summary over a single short article.
"""

from summaries import AspectSummarizer

from extract_relevant_sentence_list import load_text_file


if __name__ == '__main__':

    source_text = load_text_file("Aachen_Wiki.txt")

    summ = AspectSummarizer("yake", "frequency")

    print(summ.summarize(source_text))

