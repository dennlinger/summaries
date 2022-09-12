"""
Uses a modified variant of LexRank, coupled with sentence-transformers similarity measures.
This has been previously used in some of my other works, and works fairly reliably.
"""

import warnings
from typing import Optional, List, Union

import numpy as np
import torch.cuda
import torch
from spacy.language import Language
from sentence_transformers.util import cos_sim
from sentence_transformers import SentenceTransformer

from summaries.utils import get_nlp_model, get_st_model
from .LexRank import degree_centrality_scores


def lexrank_st_baseline(text: Union[List[str], str],
                        processor: Optional[Language] = None,
                        lang: Optional[str] = None,
                        st_model: Optional[Union[str, SentenceTransformer]] = None,
                        num_sentences: Optional[int] = None,
                        max_length: Optional[int] = None,
                        device: Optional[str] = None) -> str:
    """
    Baseline using a modification of LexRank by replacing the default centrality scores with sentence-transformers
    similarity scores. By default, uses MMR to determine the optimal output length, but also works with pre-specified
    target lengths for the summary (in number of sentences)
    :param text: The input text.
    :param processor: Optional language processor, which will be used to split sentences and determine tokens.
    :param lang: Optional specification of the language, in case no processor is specified.
    :param st_model: Sentence Transformer model to use for the similarity computation.
    :param num_sentences: Optional number of sentences to determine the length of the summary. If none is specified,
        will default to using MMR (maximal marginal relevance) for the length determination.
    :param max_length: Optional maximum length of target summary (in characters). Will use this to only add sentences
        until the maximum length is reached. Alternative to `num_sentences`.
    :param device: Pytorch device identifier on which to run the encodings. Will default to first GPU, if available,
        or CPU if not manually specified.
    :return: A summary text generated with the lexrank-st method.
    """
    if num_sentences and max_length:
        raise ValueError("Either specify num_sentences OR max_length for the LexRank summarizer, but not both!")

    if device is None:
        warnings.warn("No device specified. Will default to first GPU, if available, or otherwise CPU.")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Resolve Sentence Transformer model based on input
    if st_model is None:
        st_model = get_st_model("paraphrase-multilingual-mpnet-base-v2", device=device)
    elif isinstance(st_model, str):
        st_model = get_st_model(st_model, device=device)

    # Warn users about issues with parameters
    if isinstance(text, list) and (processor or lang):
        warnings.warn("Processor or language are specified despite not being needed for pre-split sentences.")
    # Resolve sentence input format
    if isinstance(text, str):
        # Resolve language processor
        if processor and lang:
            warnings.warn("Language code specified despite a pre-loaded processor being passed. "
                          "Will use the processor and ignore the language code.")
        elif not processor and not lang:
            raise ValueError("Specify either a pre-determined processor or a language code!")
        elif not processor and lang:
            processor = get_nlp_model("sm", lang=lang)

        text = [sent.text for sent in processor(text)]

    ordered_indices = compute_lexrank_sentences(st_model, text, device=device)

    if num_sentences:
        return " ".join([text[index] for index in ordered_indices[:num_sentences]])
    elif max_length:
        result = ""
        for index in ordered_indices:
            # Break if we hit the max length by adding the current element.
            if len(result) + len(text[index]) > max_length:
                break
            result += f"{text[index]} "
        return result.strip(" ")


def compute_lexrank_sentences(model: SentenceTransformer, segments: List, device: str) -> List[int]:
    # Models will automatically run on GPU, even without device specification!
    embeddings = model.encode(segments, convert_to_tensor=True, device=device)

    self_similarities = cos_sim(embeddings, embeddings).cpu().numpy()

    centrality_scores = degree_centrality_scores(self_similarities, threshold=None, increase_power=True)

    central_indices = list(np.argsort(-centrality_scores))
    # Use argpartition instead of argsort for faster sorting, if we only need k << n sentences.
    # central_indices = np.argpartition(centrality_scores, -num_segments)[-num_segments:]

    return central_indices
