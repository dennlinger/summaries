"""
Generates two-sided violin plot of the MLSUM and Massivesumm datasets before and after dataset filtering.
This script generates the plots for reference texts.
"""
import json
import os

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from datasets import load_dataset

from summaries.analysis import Analyzer
from summaries.preprocessing import Cleaner


if __name__ == '__main__':

    # Set correct font size
    matplotlib.rc('xtick', labelsize=18)
    matplotlib.rc('ytick', labelsize=18)
    # set LaTeX font
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'

    #MassiveSumm
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "massivesumm", "deu.all.jsonl")

    train = []
    # The data is in JSONL format
    with open(data_path, "r") as f:
        for line in f.readlines():
            sample = json.loads(line.strip("\n "))
            # Technically, this shouldn't filter out any samples, but just to make sure...
            if sample["language"] == "deu":
                train.append(sample)

    print(f"{len(train)} samples before filtering.")

    analyzer = Analyzer(lemmatize=True, lang="de")
    # Analysis with minimal length requirements set
    cleaner = Cleaner(analyzer, deduplication_method="test_first",
                      length_metric="char", min_length_summary=20, min_length_reference=50,
                      min_compression_ratio=1.25,
                      extractiveness="fully")

    clean_massivesumm = cleaner.clean_dataset("summary", "text", train, enable_tqdm=True)

    dirty_lengths_massive = [len(sample["text"]) for sample in train]
    clean_lengths_massive = [len(sample["text"]) for sample in clean_massivesumm["train"]]

    print(f"{len(dirty_lengths_massive)} vs {len(clean_lengths_massive)} after filtering.")
    del clean_massivesumm

    # MLSUM loading
    dataset = load_dataset("mlsum", "de")
    clean_mlsum = cleaner.clean_dataset("summary", "text",
                                        dataset["train"], dataset["validation"], dataset["test"],
                                        enable_tqdm=True)

    dirty_lengths_mlsum = [len(sample["text"]) for sample in dataset["train"]]
    clean_lengths_mlsum = [len(sample["text"]) for sample in clean_mlsum["train"]]

    dirty_type_massive = ["unfiltered" for _ in range(len(dirty_lengths_massive))]
    clean_type_massive = ["filtered" for _ in range(len(clean_lengths_massive))]
    dirty_type_mlsum = ["unfiltered" for _ in range(len(dirty_lengths_mlsum))]
    clean_type_mlsum = ["filtered" for _ in range(len(clean_lengths_mlsum))]

    df = pd.DataFrame()
    df["Sample Lengths"] = dirty_lengths_massive + clean_lengths_massive + dirty_lengths_mlsum + clean_lengths_mlsum
    df["Filtering"] = dirty_type_massive + clean_type_massive + dirty_type_mlsum + clean_type_mlsum
    df["Dataset"] = ["MassiveSumm" for _ in range(len(clean_lengths_massive) + len(dirty_lengths_massive))] + \
                    ["MLSUM" for _ in range(len(clean_lengths_mlsum) + len(dirty_lengths_mlsum))]

    # Given that we have a hard time plotting longer samples in context, filter the data beforehand.
    df = df[df["Sample Lengths"] <= 10000]
    ax = sns.violinplot(data=df, x="Dataset", y="Sample Lengths", hue="Filtering", split=True, gridsize=2500,
                        width=1.0, showmeans=True, scale="count", scale_hue=False,
                        cut=0, inner="quartile", palette="colorblind") #, dodge=0.6, )
    plt.ylim([0, 9000])
    ax.set(xlabel=None)
    ax.set(ylabel=None)
    plt.legend(loc="upper center", prop={'size': 18})
    plt.savefig("violins_reference.png", dpi=400)
    # plt.show()
