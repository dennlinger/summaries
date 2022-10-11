"""
Compute lead-k summaries, based on the length of the reference text divided by the average compression ratio.
"""
import json

from tqdm import tqdm
import numpy as np

from summaries.utils import get_nlp_model
from summaries.baselines import lead_k
from summaries import Analyzer
from utils import get_dataset, get_rouge_scores


if __name__ == '__main__':
    eval_rouge_scores = True
    fast = False

    nlp = get_nlp_model("sm", lang="de")
    nlp.max_length = 4_000_000
    analyzer = Analyzer(lemmatize=True, lang="de")

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
                # Extract reference texts only.
                reference_texts = [sample[reference_column] for sample in samples]
                summary_texts = [sample[summary_column].replace("\n", " ") for sample in samples]

                # Compute the compression ratios based on this
                ratios = [analyzer.compression_ratio(summary, reference, "char")
                          for summary, reference in zip(summary_texts, reference_texts)]
                average_ratio = np.mean(ratios)

                generated_summaries = []
                for reference in tqdm(reference_texts):
                    doc = nlp(reference)
                    sentences = [sent.text.strip("\n ") for sent in doc.sents if sent.text.strip("\n ") != ""]

                    # Approximate the target length based on average compression
                    target_length = round(len(sentences) / average_ratio)
                    generated_summaries.append(lead_k(sentences, k=target_length))

                with open(f"{name}_{split}_{filtered}_leadk.json", "w") as f:
                    json.dump(generated_summaries, f, ensure_ascii=False, indent=2)

                if eval_rouge_scores:
                    get_rouge_scores(summary_texts, generated_summaries)
