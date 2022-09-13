"""
Split sentences for MLSUM. This is the unfiltered version.
"""
import json

from datasets import load_dataset
from tqdm import tqdm

from summaries.utils import get_nlp_model


def sentencize(model, texts, fn):
    sentencized_texts = []
    for doc in model.pipe(texts, n_process=8):
        sentencized_texts.append([sent.text for sent in doc.sents])

    with open(fn, "w") as f:
        json.dump(sentencized_texts, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    nlp = get_nlp_model("sm", lang="de")

    dataset = load_dataset("mlsum", "de")

    for split in ["train", "validation", "test"]:
        ref_texts = []
        summ_texts = []
        for sample in dataset[split]:
            ref_texts.append(sample["text"])
            summ_texts.append(sample["summary"])

        sentencize(nlp, ref_texts, f"{split}_reference_sentences.json")
        sentencize(nlp, summ_texts, f"{split}_summary_sentences.json")
