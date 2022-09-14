"""
Script to generate baseline lead_3 summaries for all datasets.
"""
import json

from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn

from summaries.utils import get_nlp_model
from summaries.baselines import lead_3
from utils import get_dataset, get_rouge_scores


if __name__ == '__main__':
    eval_rouge_scores = True
    fast = False

    nlp = get_nlp_model("sm", lang="de")

    # MLSUM
    name = "mlsum"
    reference_column = "text"
    summary_column = "summary"

    plot_scores = {
        "filtered": {"rouge1": [], "rouge2": [], "rougeL": []},
        "unfiltered": {"rouge1": [], "rouge2": [], "rougeL": []}
    }

    for do_filter in [False, True]:
        if do_filter:
            filtered = "filtered"
        else:
            filtered = "unfiltered"

        data = get_dataset(name, filtered=do_filter)

        for split in ["validation", "test"]:
            print(f"Computing {filtered} {split} split...")
            samples = data[split]
            # Extract reference texts only.
            reference_texts = [sample[reference_column] for sample in samples]
            summary_texts = [sample[summary_column] for sample in samples]

            generated_summaries = []
            for doc in tqdm(nlp.pipe(reference_texts, n_process=8)):
                generated_summaries.append(lead_3([sent.text for sent in doc.sents]))

            with open(f"{name}_{split}_{filtered}_lead3.json", "w") as f:
                json.dump(generated_summaries, f, ensure_ascii=False, indent=2)

            if eval_rouge_scores:
                aggregator = get_rouge_scores(summary_texts, generated_summaries)

                # Also add the F1 scores for later plotting
                for metric, scores in aggregator._scores.items():
                    plot_scores[filtered][metric].extend([score.fmeasure for score in scores])

    if name == "mlsum":
        for metric in ["rouge1", "rouge2", "rougeL"]:
            fig, ax = plt.subplots()
            for filtered in ["unfiltered", "filtered"]:
                seaborn.histplot(plot_scores[filtered][metric], bins=20, stat="count",
                                 kde=False, binrange=(0, 1), ax=ax)
            ax.set(xlim=(0, 1))
            # TODO: How to automatically detect whether the maximum is larger?
            ax.set(ylim=(0, 1500))

            plt.savefig(f"mlsum_{metric}.png", dpi=300)
            plt.show()