"""
Split sentences for MLSUM. This is the unfiltered version.
"""
import json

from datasets import load_dataset
from tqdm import tqdm

from summaries.utils import get_nlp_model

if __name__ == '__main__':
    nlp = get_nlp_model("sm", lang="de")

    dataset = load_dataset("mlsum", "de")

    # for split in ["train", "validation", "test"]:
    for split in ["test"]:
        ref_texts = []
        summ_texts = []
        for sample in dataset[split]:
            ref_texts.append(sample["text"])
            summ_texts.append(sample["summary"])

        ref_sentences = list(tqdm(nlp.pipe(ref_texts, n_process=16)))
        ref_sentences = [[sent.text for sent in doc.sents] for doc in ref_sentences]
        with open("reference_sentences.json", "w") as f:
            json.dump(ref_sentences, f, ensure_ascii=False, indent=2)

        summ_sentences = list(tqdm(nlp.pipe(summ_texts, n_process=16)))
        summ_sentences = [[sent.text for sent in doc.sents] for doc in summ_sentences]
        with open("summary_sentences.json", "w") as f:
            json.dump(summ_sentences, f, ensure_ascii=False, indent=2)

