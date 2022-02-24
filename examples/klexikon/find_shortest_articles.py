"""
For our idea of overfitting networks, find the shortest possible articles.
"""

from datasets import load_dataset
import numpy as np
from transformers import MT5Tokenizer


def get_clean_and_combined_text(sentences):
    clean_texts = [line for line in sentences
                   if line.strip("\n ") and not line.startswith("=")]
    concatenated_text = " ".join(clean_texts)
    return concatenated_text


if __name__ == '__main__':

    dataset = load_dataset("dennlinger/klexikon")
    lengths = []
    for sample in dataset["train"]:
        lengths.append(len(sample["wiki_sentences"]))

    sorted_indices = np.argsort(lengths)

    tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")

    # These are the shortest 10 indices:
    # array([ 260, 1301, 2088,  665, 1572,  436, 1887, 1422, 1506,  474])
    for idx in sorted_indices[:10]:
        sample = dataset["train"][int(idx)]
        print(f"Length of shortest input in sentences: {len(sample['wiki_sentences'])}")
        print(f"Length of shortest input articles in word pieces:")

        print(len(tokenizer(get_clean_and_combined_text(sample["wiki_sentences"]))["input_ids"]))
        print(f"Length of associated summary:")
        print(len(tokenizer(get_clean_and_combined_text(sample["klexikon_sentences"]))["input_ids"]))


