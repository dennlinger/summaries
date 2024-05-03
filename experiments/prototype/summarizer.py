"""
We generate the embeddings of the model on Colab, because it is faster.
"""

import json

from sentence_transformers import SentenceTransformer


if __name__ == '__main__':
    fn = "./TV.json"

    with open(fn) as f:
        data = json.load(f)

    st = SentenceTransformer("Alibaba-NLP/gte-large-en-v1.5",
                             device="cpu",
                             trust_remote_code=True)