"""
Training loop for working with the merged dataset.
This simplifies the training to basically determining hyperparameters.
"""

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset


def get_args(debug: bool = False) -> Seq2SeqTrainingArguments:
    if debug:
        args = Seq2SeqTrainingArguments(
            output_dir="./German-MultiSumm",
            overwrite_output_dir=True,
            do_train=True,
            do_eval=True,
            do_predict=False,
            evaluation_strategy="steps",
            eval_steps=50,
            auto_find_batch_size=True,
            # per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=4,
            # eval_accumulation_steps=20,
            # eval_delay=0.5,
            learning_rate=5e-5,
            num_train_epochs=1000,
            lr_scheduler_type="constant",
            warmup_steps=50,  # roughly equivalent to 1/6 of the first epoch
            logging_strategy="steps",
            logging_steps=50,
            save_strategy="no",
            seed=768,
            data_seed=512,
            # bf16=True,  # experimental feature, doesn't work on our Titan RTX GPUs!
            # bf16_full_eval=True,
            optim="adamw_torch",
            gradient_checkpointing=False,  # Experiment with memory saves?
        )
    else:
        args = Seq2SeqTrainingArguments(
            output_dir="./German-MultiSumm",
            overwrite_output_dir=True,
            do_train=True,
            do_eval=True,
            do_predict=False,
            evaluation_strategy="epoch",
            auto_find_batch_size=True,
            # per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=2,
            # eval_accumulation_steps=20,
            # eval_delay=0.5,
            learning_rate=5e-5,
            num_train_epochs=7,
            lr_scheduler_type="linear",
            warmup_steps=5000,  # roughly equivalent to 1/6 of the first epoch
            logging_strategy="steps",
            logging_steps=250,
            save_strategy="epoch",
            seed=768,
            data_seed=512,
            # bf16=True,  # experimental feature, doesn't work on our Titan RTX GPUs!
            # bf16_full_eval=True,
            run_name="Mega German Summarization",
            optim="adamw_torch",
            gradient_checkpointing=False,  # Experiment with memory saves?
        )

    return args


if __name__ == '__main__':
    debug = True

    if debug:
        model_name = "benjamin/gerpt2"
        max_length = 256
        summary_max_length = 128
    else:
        model_name = "benjamin/gerpt2"
        max_length = 768
        summary_max_length = 512

    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=max_length, use_fast=False)
    # Enable this for T5-based models
    # model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    # Enable this for GPT-based models
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Adapted from Jing's practical code
    def tokenize_function(samples):
        model_inputs = tokenizer(
            samples["text"],
            text_target=samples["target"],
            padding="longest",
            truncation=True,
            max_length=max_length)

        return model_inputs

    data_files = {
        "train": "german_summarization.jsonl",
        "validation": "german_summarization_validation.jsonl"
    }
    if debug:
        data_files["train"] = "german_summarization_validation.jsonl"

    dataset = load_dataset("json", data_files=data_files)
    if debug:
        dataset["train"] = dataset["train"].select(range(5))
        dataset["validation"] = dataset["validation"].select(range(5))

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    args = get_args(debug=debug)

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        # optimizers=None,
        # preprocess_logits_for_metrics=None
    )

    trainer.train()
