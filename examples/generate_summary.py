"""
Minimal script to generate a summary over a single short article.
"""

from summaries import AspectSummarizer
from summaries.extractors import OracleExtractor

from extract_relevant_sentence_list import load_text_file


if __name__ == '__main__':

    source_text = load_text_file("Aachen_Wiki_short.txt")
    oracle_keywords = ["Aachen", "Nordrhein-Westfalen", "Einwohner", "Aquae Granni", "Aachener Dom", "Universität",
                       "UNESCO", "Oche"]
    extractor = OracleExtractor(len(oracle_keywords), lang="de", given_keywords=oracle_keywords)
    summ = AspectSummarizer(extractor, "dpr")

    print(summ.summarize(source_text))

