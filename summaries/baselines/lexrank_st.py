"""
Uses a modified variant of LexRank, coupled with sentence-transformers similarity measures.
This has been previously used in some of my other works, and works fairly reliably.
"""

from typing import Optional, List

import numpy as np
from spacy.language import Language
from sentence_transformers.util import cos_sim
from sentence_transformers import SentenceTransformer

from .LexRank import degree_centrality_scores


def lexrank_st_baseline(text: str,
                        processor: Optional[Language] = None,
                        lang: Optional[str] = None,
                        num_sentences: Optional[int] = None) -> str:
    """
    Baseline using a modification of LexRank by replacing the default centrality scores with sentence-transformers
    similarity scores. By default, uses MMR to determine the optimal output length, but also works with pre-specified
    target lengths for the summary (in number of sentences)
    :param reference_text: The input text.
    :param processor: Optional language processor, which will be used to split sentences and determine tokens.
    :param lang: Optional specification of the language, in case no processor is specified.
    :param num_sentences: Optional number of sentences to determine the length of the summary. If none is specified,
        will default to using MMR (maximal marginal relevance) for the length determination.
    :return: A summary text generated with the lexrank-st method.
    """


def compute_lexrank_sentences(model: SentenceTransformer, segments: List, device: str, num_segments: int):
    # Models will automatically run on GPU, even without device specification!
    embeddings = model.encode(segments, convert_to_tensor=True, device=device)

    self_similarities = cos_sim(embeddings, embeddings).cpu().numpy()

    centrality_scores = degree_centrality_scores(self_similarities, threshold=None, increase_power=True)

    # Use argpartition instead of argsort for faster sorting, since we only need k << n sentences.
    # most_central_indices = np.argsort(-centrality_scores)
    central_indices = np.argpartition(centrality_scores, -num_segments)[-num_segments:]

    # TODO: Figure out whether sorting makes sense here? We assume that Wikipedia has some sensible structure.
    #   Otherwise, reversing would be enough to get the job done and get the most similar sentences first.
    # Scores are originally in ascending order
    # list(most_central_indices).reverse()
    return sorted(list(central_indices))