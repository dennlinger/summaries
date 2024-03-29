"""
Similar to examples/density_plot.py, but computes it on the full MLSum (German) dataset.
"""

from datasets import load_dataset

from tqdm import tqdm
from summaries.analysis import Stats
from summaries.better_split import better_sentence_split

if __name__ == '__main__':
    dataset = load_dataset("mlsum", "de")

    stats = Stats(lang="de")

    for partition_name in ["train", "validation", "test"]:
        partition = dataset[partition_name]

        reference_texts = []
        summary_texts = []

        # first_n = 5

        for idx, sample in enumerate(tqdm(partition)):
            reference_sentences = better_sentence_split(sample["text"])
            summary_sentences = better_sentence_split(sample["summary"])
            reference_texts.append(reference_sentences)
            summary_texts.append(summary_sentences)

            # if idx >= first_n:
            #     break

        # FIXME: Moved to new tool!
        stats.density_plot(reference_texts, summary_texts, out_fn=f"density_{partition_name}.png")

