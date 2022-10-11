"""
Quick debugging script to verify which versions of a model work on our GPUs.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("benjamin/gerpt2-large", model_max_length=1024)
    print(tokenizer.model_max_length)

    data = load_dataset("dennlinger/klexikon")
    text = " ".join(sentence.strip("\n ") for sentence in data["train"][0]["wiki_sentences"])
    sample = tokenizer(text, return_tensors="pt", max_length=1024, padding="longest", truncation=True)
    print(sample["input_ids"].shape)

    # model = AutoModelForCausalLM.from_pretrained("benjamin/gerpt2-large")
    # model.to("cuda:1")
    # sample.to("cuda:1")
    #
    # result = model(**sample).to("cpu")
