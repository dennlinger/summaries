"""
Compute LexRank summaries, based on the length of the reference text divided by the average compression ratio.
"""
import json

from tqdm import tqdm
import numpy as np

from summaries.utils import get_nlp_model
from summaries.baselines import lexrank_st
from summaries import Analyzer
from utils import get_dataset, get_rouge_scores


if __name__ == '__main__':
    eval_rouge_scores = True
    fast = False

    nlp = get_nlp_model("sm", lang="de")
    analyzer = Analyzer(lemmatize=True, lang="de")

    # MLSUM
    name = "mlsum"
    reference_column = "text"
    summary_column = "summary"

    # for do_filter in [False, True]:
    for do_filter in [True]:
        if do_filter:
            filtered = "filtered"
        else:
            filtered = "unfiltered"

        data = get_dataset(name, filtered=do_filter)

        # for split in ["validation", "test"]:
        for split in ["test"]:
            print(f"Computing {filtered} {split} split...")
            samples = data[split]
            # Extract reference texts only.
            reference_texts = [sample[reference_column] for sample in samples]
            summary_texts = [sample[summary_column] for sample in samples]

            # Compute the compression ratios based on this
            ratios = [analyzer.compression_ratio(summary, reference, "char")
                      for summary, reference in zip(summary_texts, reference_texts)]
            average_ratio = np.mean(ratios)

            generated_summaries = []
            for doc in tqdm(nlp.pipe(reference_texts, n_process=8)):
                sentences = [sent.text for sent in doc.sents]
                # Approximate the target length based on average compression. Min length is one sentence.
                target_length = min(round(len(sentences) / average_ratio), 1)
                generated_summaries.append(lexrank_st(sentences,
                                                      st_model="paraphrase-multilingual-mpnet-base-v2",
                                                      device="cuda:1",
                                                      num_sentences=target_length))

            with open(f"{name}_{split}_{filtered}_lexrank.json", "w") as f:
                json.dump(generated_summaries, f, ensure_ascii=False, indent=2)

            if eval_rouge_scores:
                get_rouge_scores(summary_texts, generated_summaries)