"""
Quick debugging script to verify which versions of a model work on our GPUs.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("benjamin/gerpt2-large")
    model = AutoModelForCausalLM.from_pretrained("benjamin/gerpt2-large")

    data = load_dataset("dennlinger/klexikon")

    sample = tokenizer(data["train"][0], return_tensors="pt")

    model.to("cuda:1")
    sample.to("cuda:1")

    result = model(**sample).to("cpu")
