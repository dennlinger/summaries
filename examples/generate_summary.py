"""
Minimal script to generate a summary over a single short article.
"""
from typing import List

from summaries import AspectSummarizer
from summaries.extractors import OracleExtractor


def load_text_file_in_sentences(fn: str) -> List[str]:
    with open(fn, "r") as f:
        text = f.readlines()

    # Remove empty lines and headings
    text = [line.strip("\n ") for line in text if line.strip("\n ") and not line.startswith("=")]
    return text


if __name__ == '__main__':

    source_text = load_text_file_in_sentences("Aachen_Wiki_short.txt")
    oracle_keywords = ["Aachen", "Nordrhein-Westfalen", "Einwohner", "Aquae Granni", "Aachener Dom", "Universität",
                       "UNESCO", "Oche"]
    extractor = OracleExtractor(len(oracle_keywords), lang="de", given_keywords=oracle_keywords)
    summ = AspectSummarizer(extractor, "dpr")

    print(summ.summarize(source_text))

