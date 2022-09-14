"""
Script to generate baseline lead_3 summaries for all datasets.
"""
import json

from tqdm import tqdm

from summaries.utils import get_nlp_model
from summaries.baselines import lead_3
from utils import get_dataset


if __name__ == '__main__':
    nlp = get_nlp_model("sm", lang="de")

    # MLSUM
    name = "mlsum"
    reference_column = "text"

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

            generated_summaries = []
            for doc in tqdm(nlp.pipe(reference_texts, n_process=8)):
                generated_summaries.append(lead_3([sent.text for sent in doc.sents]))

            with open(f"{name}_{split}_{filtered}_lead3.json", "w") as f:
                json.dump(generated_summaries, f, ensure_ascii=False, indent=2)
