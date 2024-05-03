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

THESIS_QUERIES = [
    "Limitations of summarization systems",
    "EUR-Lex-Sum dataset",
    "Klexikon dataset",
    "SRLScore",
    "Aspect-based summarization model"
    "Ex-ante retrieval",
    "Ex-post rewriting"
]



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


def extract_lead_entities(nlp, fn, k=10, per_item=3):
    """
    Extracts the first k occurring entities.
    """
    with open(fn, "r") as f:
        text = f.read()

    # Basic processing
    text = re.sub(r'\n([A-Za-z])', ' \g<1>', text)
    text = re.sub(r'[ \t\r\x0b\x0c]', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {3,}', '  ', text)

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

def query_relevance():
    sentences = []
    embeddings = np.ndarray(shape=(0, 1024), dtype=np.float32)

    for chapter in tqdm(range(2, 8)):
        with open(f"./Thesis/Thesis_{chapter}.json", "r") as f:
            data = json.load(f)
            sentences.extend(data)

        with open(f"./Thesis/Thesis_{chapter}.npy", "rb") as f:
            new_embs = np.load(f, allow_pickle=True)
            embeddings = np.concatenate((embeddings, new_embs), axis=0)

    model = SentenceTransformer("Alibaba-NLP/gte-large-en-v1.5",
                                device="cpu",
                                trust_remote_code=True)

    query_dict = {}
    for query in tqdm(THESIS_QUERIES):
        query_dict[query] = []
        query_embed = model.encode(query)
        cos_scores = util.cos_sim(query_embed, embeddings)[0]
        top_results = torch.topk(cos_scores, k=10)
        print(top_results)

        # print("Query:", query)
        # print("\nTop 10 most similar sentences in corpus:")

        for score, idx in zip(top_results[0], top_results[1]):
            # print(sentences[idx], "(Score: {:.4f})".format(score))
            query_dict[query].append(sentences[idx])
    return query_dict


if __name__ == '__main__':
    # preprocess()
    nlp = spacy.load('en_core_web_md')

    # Extracts 9 sentences per chapter, three for each of the
    # top-3 most frequent entities.
    prompt = """
Consider the following context extracted from a PhD thesis in the field of Computer Science.
Excerpts of the thesis are provided in a JSON object containing excerpts for multiple chapters.
The contents are related to either a entity or a context query.
The structure of the JSON is as follows:
{
    "<query string>": [
        "relevant segment",
        ...
    ],
    "<entity name>": [
        "relevant segment",
        ...
    ]
...
}

Rewrite the most appropriate excerpts into a cohesive summary.
Maintain a scientific tone and focus on individual contributions.
Ensure the summary is longer than 250 words, but keep your response shorter than 400 words.
"""

    prompt_dict = {}

    query_dict = query_relevance()
    prompt += json.dumps(query_dict, indent=2)#

    # Get the relevant entity stuff
    for chapter in tqdm(range(2, 8)):
        fn = f"../../../Documents/Thesis_{chapter}.txt"
        ent_sents = extract_lead_entities(nlp, fn, k=3, per_item=3)
        # pprint.pp(ent_sents)
        prompt_dict.update(ent_sents)

    prompt += json.dumps(prompt_dict, indent=2)
    prompt += "\nSummary:"

    # Pass this to the Cohere client
    co = cohere.Client(api_key=CO_API_KEY)
    chat = co.chat(
        message=prompt,
        model="command-r-plus"
    )
    print(chat.text)
