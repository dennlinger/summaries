"""
Main script to visualize attention on Klexikon articles with mT5.
"""
import os

import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import MT5TokenizerFast, MT5ForConditionalGeneration, AdamW


def prepare_text_input(sentences, max_sentences=15):
    clean_texts = [line.strip("\n ") for line in sentences
                   if line.strip("\n ") and not line.startswith("=")]
    # Add sentence separators
    clean_texts = clean_texts[:max_sentences]
    concatenated_text = " <extra_id_1> ".join(clean_texts)
    return concatenated_text


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    shortest_article_ids = [260, 1301, 2088, 665, 1572, 436, 1887, 1422, 1506, 474]

    epochs = 300  # 250 seems to already work decently well, maybe even less.
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    dataset = load_dataset("dennlinger/klexikon")
    tokenizer = MT5TokenizerFast.from_pretrained("google/mt5-small")

    for idx in tqdm(shortest_article_ids):
        # Reload model for each sample to re-start from checkpoint
        model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small").to(device)

        sample = dataset["train"][idx]

        # Prepare with sensible border tokens. Decoder needs to start with <pad>
        wiki_text = f"<extra_id_0> <extra_id_1> {prepare_text_input(sample['wiki_sentences'], max_sentences=20)}"
        klexikon_text = f"<pad> <extra_id_1> {prepare_text_input(sample['klexikon_sentences'], max_sentences=10)}"

        model_inputs = tokenizer(wiki_text, return_tensors="pt")
        decoder_inputs = tokenizer(klexikon_text, return_tensors="pt")
        model_inputs["decoder_input_ids"] = decoder_inputs["input_ids"]

        model_inputs.to(device)

        for _ in tqdm(range(epochs)):
            optimizer = AdamW(model.parameters(), lr=3e-4)
            model.train()

            result = model(input_ids=model_inputs["input_ids"], attention_mask=model_inputs["attention_mask"],
                           decoder_input_ids=model_inputs["decoder_input_ids"], output_attentions=True,
                           labels=model_inputs["decoder_input_ids"])

            loss = result.loss
            # print(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        model.save_pretrained(f"./overfit_models/{idx}")

        """
        Moved to separate file for processing
        """
        # result = model(input_ids=model_inputs["input_ids"], attention_mask=model_inputs["attention_mask"],
        #                decoder_input_ids=model_inputs["decoder_input_ids"], output_attentions=True,
        #                labels=model_inputs["decoder_input_ids"])
        #
        # predicted_ids = torch.argmax(result.logits.detach().to("cpu"), dim=-1)
        # print(tokenizer.decode(predicted_ids[0]))
        #
        # model_view(cross_attention=result.cross_attentions,
        #            encoder_tokens=tokenizer.convert_ids_to_tokens(model_inputs["input_ids"][0]),
        #            decoder_tokens=tokenizer.convert_ids_to_tokens(model_inputs["decoder_input_ids"][0]))