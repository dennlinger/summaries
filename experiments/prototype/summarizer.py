import spacy
import re
import json
import pprint
from collections import Counter
import torch

from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import numpy as np
from tqdm import tqdm
import cohere

from env import CO_API_KEY


RELEVANT_TYPES = {"ORG", "PERSON", "LOC", "GPE", "LANGUAGE"}



def preprocess():
    nlp = spacy.load('en_core_web_sm')
    tok = AutoTokenizer.from_pretrained("CohereForAI/c4ai-command-r-plus")
    tokens = 0

    for chapter in range(2, 8):
        with open(f"../../../Documents/Thesis_{chapter}.txt", "r") as f:
            text = f.read()

        # Basic processing
        text = re.sub(r'\n([A-Za-z])', '\g<1>', text)
        text = re.sub(r'[ \t\r\x0b\x0c]', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {3,}', '  ', text)

        doc = nlp(text)
        print(f"{len(list(doc.sents))} sentences in Chapter {chapter}.")

        tokenized = tok(text)
        toks = len(tokenized['input_ids'])
        tokens += toks
        sentences = [sent.text for sent in doc.sents if len(sent.text) > 25]
        print(f"{toks} tokens in Chapter {chapter}.")
        with open(f"./Thesis_{chapter}.json", "w") as f:
            json.dump(sentences, f, indent=2, ensure_ascii=False)

    print(f"{tokens} total tokens in Thesis.")


def extract_lead_entities(nlp, text, k=10, per_item=3):
    """
    Extracts the first k occurring entities.
    """

    # Basic processing
    # text = re.sub(r'\n([A-Za-z])', ' \g<1>', text)
    # text = re.sub(r'[ \t\r\x0b\x0c]', ' ', text)
    # text = re.sub(r'\n{3,}', '\n\n', text)
    # text = re.sub(r' {3,}', '  ', text)

    doc = nlp(text)

    # for ent in doc.ents:
    #     if "klexikon" in ent.text.lower() or "eur-lex-sum" in ent.text.lower():
    #         print(ent.text, ent.label_)
    relevant_ents = [ent.text for ent in doc.ents if ent.label_ in RELEVANT_TYPES and len(ent.text) >= 3]

    relevant_ents = [el[0] for el in Counter(relevant_ents).most_common(k)]

    contexts = {}

    # Get the first n sentences for each of the top-occurring entities,
    # where said entity is present.
    for top_ent in relevant_ents:
        counter = 0
        contexts[top_ent] = []
        for ent in doc.ents:
            if ent.text == top_ent:
                contexts[top_ent].append(ent.sent.text)
                counter += 1
            if counter > per_item:
                break

    return contexts


def query_relevance(exid, sentences, aspect, model, k=10):
    with open(f"./TV/{exid}.npy", "rb") as f:
        embeddings = np.load(f, allow_pickle=True)

    query_dict = {}

    ASPECT_QUERIES = [
        aspect,
        f"{aspect} of the TV show"
    ]
    # ASPECT_QUERIES.extend(entities)

    for query in ASPECT_QUERIES:
        query_dict[query] = []
        query_embed = model.encode(query)
        cos_scores = util.cos_sim(query_embed, embeddings)[0]
        top_results = torch.topk(cos_scores, k=k)
        # print(top_results)

        # print("Query:", query)
        # print("\nTop 10 most similar sentences in corpus:")

        for score, idx in zip(top_results[0], top_results[1]):
            # print(sentences[idx], "(Score: {:.4f})".format(score))
            query_dict[query].append(sentences[idx])
    return query_dict


if __name__ == '__main__':
    # preprocess()
    nlp = spacy.load('en_core_web_md')

    model = SentenceTransformer("Alibaba-NLP/gte-large-en-v1.5",
                                device="cpu",
                                trust_remote_code=True)

    with open("TV_filtered.json", "r") as f:
        data = json.load(f)

    for exid, sample in tqdm(data.items()):

        data[exid]["generation"] = {}
        for aspect, gold in tqdm(sample["targets"]):


            prompt = f"""
You are instructed to write a subsection of a Wikipedia article.
You may infer the page title of the requested article from the provided context.
Specifically, you are tasked with the creation of a subsection with in this article, titled "{aspect}", and the following segments provide context for this subsection.
The data is provided as a JSON object with the following structure:

{{
    "<query string>": [
        "relevant segment",
        ...
    ],
    "<entity name>": [
        "relevant segment",
        ...
    ]
...
}}

Ensure the generated text is written in the style of a Wikipedia article.
"""

            prompt_dict = {}

            query_dict = query_relevance(exid, sample["inputs"], aspect, model, k=15)
            prompt += json.dumps(query_dict, indent=2)

            # Get the relevant entity stuff
            input_text = "\n".join(sample["inputs"])
            ent_sents = extract_lead_entities(nlp, input_text, k=3, per_item=3)
            # pprint.pp(ent_sents)
            prompt_dict.update(ent_sents)

            prompt += json.dumps(prompt_dict, indent=2)
            prompt += f"\n\nWrite no more than two paragraphs.\nWikipedia Article Subsection for the aspect \"{aspect}\":"

            # Pass this to the Cohere client
            co = cohere.Client(api_key=CO_API_KEY)
            chat = co.chat(
                message=prompt,
                model="command-r-plus"
            )
            print(chat.text)
            data[exid]["generation"][aspect] = chat.text

    with open("TV_with_responses.json", "w") as f:
        json.dump(data, f, indent=2)
