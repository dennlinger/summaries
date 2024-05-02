import spacy
import re
from transformers import AutoTokenizer

if __name__ == '__main__':
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
        print(f"{toks} tokens in Chapter {chapter}.")

    print(f"{tokens} total tokens in Thesis.")
