"""
Training loop for working with the merged dataset.
This simplifies the training to basically determining hyperparameters.
"""

from transformers import Seq2SeqTrainer, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

if __name__ == '__main__':
    pass