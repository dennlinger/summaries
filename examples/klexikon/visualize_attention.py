"""
Main script to visualize attention on Klexikon articles with mT5.
"""

from tqdm import tqdm
from datasets import load_dataset
from transformers import MT5TokenizerFast, MT5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments


def prepare_text_input(sentences):
    clean_texts = [line.strip("\n ") for line in sentences
                   if line.strip("\n ") and not line.startswith("=")]
    # Add sentence separators
    concatenated_text = " <extra_id_1> ".join(clean_texts)
    return concatenated_text


if __name__ == '__main__':
    shortest_article_ids = [260, 1301, 2088, 665, 1572, 436, 1887, 1422, 1506, 474]

    epochs = 3

    dataset = load_dataset("dennlinger/klexikon")
    tokenizer = MT5TokenizerFast.from_pretrained("google/mt5-small")
    model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")

    for idx in tqdm(shortest_article_ids):
        sample = dataset["train"][idx]

        # Prepare with sensible border tokens. Decoder needs to start with <pad>
        wiki_text = f"<extra_id_0> {prepare_text_input(sample['wiki_sentences'])}"
        klexikon_text = f"<pad> {prepare_text_input(sample['klexikon_sentences'])}"

        model_inputs = tokenizer(wiki_text, return_tensors="pt")
        decoder_inputs = tokenizer(klexikon_text, return_tensors="pt")
        model_inputs["decoder_input_ids"] = decoder_inputs["input_ids"]

        for _ in range(epochs):

            model.zero_grad()

            result = model(input_ids=model_inputs["input_ids"], attention_mask=model_inputs["attention_mask"],
                           decoder_input_ids=decoder_inputs["input_ids"], output_attentions=True,
                           label=decoder_inputs["input_ids"])

            loss = result["loss"]
            loss.backward()


