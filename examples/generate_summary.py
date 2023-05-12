"""
Minimal script to generate a summary over a single short article.
"""
from typing import List

from summaries import AspectSummarizer


def load_text_file(fn: str) -> str:
    with open(fn, "r") as f:
        text = f.readlines()

    # Remove empty lines and headings
    text = [line.strip("\n ") for line in text if line.strip("\n ") and not line.startswith("=")]
    return "\n".join(text)


if __name__ == '__main__':

    source_text = load_text_file("Aachen_Wiki_short.txt")
    summ = AspectSummarizer(segment_level="sentence")

    ex_ante_aspects = {
        "lead": [5]
    }
    ex_post_aspects = None

    print(summ.summarize(source_text, ex_ante_aspects, segment_limit_intermediate_representation=5))


