"""
Similar to examples/density_plot.py, but computes it on the full MLSum (German) dataset.
"""

from datasets import load_dataset

from tqdm import tqdm
from summaries.analysis import DensityPlot
from summaries.better_split import better_sentence_split

if __name__ == '__main__':
    dataset = load_dataset("mlsum", "de")

    dp = DensityPlot(max_num_bins=100)

    for partition_name in ["train", "validation", "test"]:
        partition = dataset[partition_name]

        reference_texts = []
        summary_texts = []

        for sample in tqdm(partition):
            reference_sentences = better_sentence_split(sample["text"])
            summary_sentences = better_sentence_split(sample["summary"])
            reference_texts.append(reference_sentences)
            summary_texts.append(summary_sentences)

        # first_n = 5
        # reference_texts = reference_texts[:first_n]
        # summary_texts = summary_texts[:first_n]

        dp.plot(reference_texts, summary_texts, out_fn=f"density_{partition_name}.png")

