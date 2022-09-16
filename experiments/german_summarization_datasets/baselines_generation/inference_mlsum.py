"""
Inference of existing DL models with the MLSUM dataset
"""
import json

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

from utils import get_dataset, get_rouge_scores


def get_summarizer_pipeline(model_name: str, batch_size: int = 16):
    # Chosen cutoff by ourselves
    if "ml6team" in model_name:
        model_max_length = 768
    # In model card
    elif "T-Systems-onsite" in model_name:
        model_max_length = 800
    else:
        model_max_length = 512
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=model_max_length)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = 1 if torch.cuda.is_available() else -1
    pipe = pipeline("summarization", model=model, tokenizer=tokenizer, device=device, batch_size=batch_size)

    return pipe


if __name__ == '__main__':
    model_names = [
        # "mrm8488/bert2bert_shared-german-finetuned-summarization",
        "Shahm/t5-small-german",
        "Einmalumdiewelt/T5-Base_GNAD",
        "ml6team/mt5-small-german-finetune-mlsum",
        "T-Systems-onsite/mt5-small-sum-de-en-v2",
    ]

    # MLSUM
    name = "mlsum"
    reference_column = "text"
    summary_column = "summary"

    eval_rouge_scores = True

    for model_name in model_names:
        pipe = get_summarizer_pipeline(model_name, batch_size=16)

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

                generated_summaries = [generation["summary_text"] for generation in
                                       pipe(reference_texts, min_length=10, no_repeat_ngram_size=3, truncation=True)]

                with open(f"{name}_{split}_{filtered}_{model_name.replace('/', '__')}.json", "w") as f:
                    json.dump(generated_summaries, f, ensure_ascii=False, indent=2)

                if eval_rouge_scores:
                    aggregator = get_rouge_scores(summary_texts, generated_summaries)
