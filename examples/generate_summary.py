"""
Minimal script to generate a summary over a single short article.
"""

from summaries import AspectSummarizer


if __name__ == '__main__':
    with open("Aachen_Wiki.txt") as f:
        source_text = f.readlines()

    source_text = [line.strip("\n ") for line in source_text if line.strip("\n ") and not line.startswith("=")]
    source_text = " ".join(source_text)

    summ = AspectSummarizer("yake", "frequency")

    print(summ.summarize(source_text))

