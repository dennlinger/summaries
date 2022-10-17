"""
Generates two-sided violin plot of the MLSUM and Massivesumm datasets before and after dataset filtering.
This script generates the plots for summary texts.
"""
import json

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
    data_path = "/home/dennis/massivesumm/deu.all.jsonl"

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

    dirty_lengths_massive = [len(sample["summary"]) for sample in train]
    clean_lengths_massive = [len(sample["summary"]) for sample in clean_massivesumm["train"]]

    del clean_massivesumm

    # MLSUM loading
    dataset = load_dataset("mlsum", "de")
    clean_mlsum = cleaner.clean_dataset("summary", "text",
                                        dataset["train"], dataset["validation"], dataset["test"],
                                        enable_tqdm=True)

    dirty_lengths_mlsum = [len(sample["summary"]) for sample in dataset["train"]]
    clean_lengths_mlsum = [len(sample["summary"]) for sample in clean_mlsum["train"]]

    dirty_type_massive = ["unfiltered" for _ in range(len(dirty_lengths_massive))]
    clean_type_massive = ["filtered" for _ in range(len(clean_lengths_massive))]
    dirty_type_mlsum = ["unfiltered" for _ in range(len(dirty_lengths_mlsum))]
    clean_type_mlsum = ["filtered" for _ in range(len(clean_lengths_mlsum))]

    df = pd.DataFrame()
    df["Sample Lengths"] = dirty_lengths_massive + clean_lengths_massive + dirty_lengths_mlsum + clean_lengths_mlsum
    df["Summary Text"] = dirty_type_massive + clean_type_massive + dirty_type_mlsum + clean_type_mlsum
    df["Dataset"] = ["MassiveSumm" for _ in range(len(clean_lengths_massive) + len(dirty_lengths_massive))] + \
                    ["MLSUM" for _ in range(len(clean_lengths_mlsum) + len(dirty_lengths_mlsum))]

    ax = sns.violinplot(data=df, x="Dataset", y="Sample Lengths", hue="Summary Text", split=True, gridsize=1200,
                        showmeans=True, scale="count", cut=0, inner="quartile", dodge=0.6, palette="colorblind")
    plt.ylim([0, 450])
    ax.set(xlabel=None)
    ax.set(ylabel=None)
    plt.legend(loc="upper center", prop={'size': 18})
    plt.savefig("violins_summary.png", dpi=400)
    plt.show()
