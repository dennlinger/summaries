"""
Minimal script to generate a summary over a single short article.
"""

from summaries import AspectSummarizer


if __name__ == '__main__':
    with open("Aachen.txt") as f:
        source_text = "".join(f.readlines())

    summ = AspectSummarizer("yake", "frequency")

    print(summ.summarize(source_text))

