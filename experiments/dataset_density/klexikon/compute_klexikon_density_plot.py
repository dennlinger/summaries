"""
Similar to examples/density_plot.py, but computes it on the full dataset.
"""

from datasets import load_dataset
from tqdm import tqdm

from summaries.analysis import DensityPlot

if __name__ == '__main__':
    dataset = load_dataset("dennlinger/klexikon")

    dp = DensityPlot(max_num_bins=100)

    for partition_name in ["train", "validation", "test"]:
        partition = dataset[partition_name]

        reference_texts = []
        summary_texts = []

        for sample in partition:
            reference_texts.append(sample["wiki_sentences"])
            summary_texts.append(sample["klexikon_sentences"])

        # first_n = 5
        # reference_texts = reference_texts[:first_n]
        # summary_texts = summary_texts[:first_n]

        dp.plot(reference_texts, summary_texts, out_fn=f"density_{partition_name}.png")

