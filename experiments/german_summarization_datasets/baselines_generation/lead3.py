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
    nlp.max_length = 4_000_000
    # for name in ["mlsum", "klexikon", "legalsum", "eurlexsum"]:
    for name in ["mlsum"]:
        if name == "mlsum":
            reference_column = "text"
            summary_column = "summary"
        elif name == "klexikon":
            reference_column = "wiki_text"
            summary_column = "klexikon_text"
        elif name == "legalsum":
            reference_column = "reference"
            summary_column = "summary"
        elif name == "eurlexsum":
            reference_column = "reference_text"
            summary_column = "summary_text"
        else:
            raise ValueError("Not configured yet.")

        plot_scores = {
            "filtered": {"rouge1": [], "rouge2": [], "rougeL": []},
            "unfiltered": {"rouge1": [], "rouge2": [], "rougeL": []}
        }

        # for do_filter in [False, True]:
        for do_filter in [True]:
            if do_filter:
                filtered = "filtered"
            else:
                filtered = "unfiltered"

            data = get_dataset(name, filtered=do_filter)

            for split in ["validation", "test"]:
                print(f"Computing {filtered} {split} split...")
                samples = data[split]

                reference_texts = [sample[reference_column] for sample in samples]
                summary_texts = [sample[summary_column].replace("\n", " ") for sample in samples]

                generated_summaries = []
                print(f"Generating spacy docs for each summary...")
                for reference in tqdm(reference_texts):
                    doc = nlp(reference)
                    sentences = [sent.text.strip("\n ") for sent in doc.sents if sent.text.strip("\n ") != ""]

                    generated_summaries.append(lead_3(sentences))

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
