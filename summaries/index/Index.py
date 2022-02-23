"""
Index class with optimized structure for in-memory retrieval.
TODO: Figure out how efficiently such an approach scales to larger source data sets.
"""

from typing import List, Dict, Union

import numpy as np
import torch
from spacy.language import Language
from transformers import Pipeline


class Index:
    term_lookup: Dict[str, List[int]]
    doc_lookup: Dict[int, str]
    encoded_documents: np.array
    tokenized_lookup: Dict[int, List[str]]
    sources: List[str]

    def __init__(self, sources: List[str], processor: Union[Language, Pipeline]):
        self.sources = sources

        self.doc_lookup = {idx: sentence for idx, sentence in enumerate(sources)}

        # Build more traditional inverted index if spacy processor is given
        if isinstance(processor, Language):
            # Generated tokenized version of sentences without punctuation and spaces/newlines
            self.tokenized_lookup = {idx: [token.text for token in processor(sentence)
                                           if not (token.is_punct or token.is_space)]
                                     for idx, sentence in enumerate(sources)}

            # FIXME: Check tokenization method for faster processing
            self.term_lookup = {}
            self.build_term_lookup()
        # for HF models, encode the sentences as vectors
        elif isinstance(processor, Pipeline):
            self.encode_docs(processor)
        else:
            raise ValueError("Unrecognized Index processor!")

    def build_term_lookup(self) -> None:
        """
        Build an inverted index based on the sentence tokens, associated with frequency
        """
        for idx, sentence in self.tokenized_lookup.items():
            for token in sentence:
                occurrences = self.term_lookup.get(token, list())
                occurrences.append(idx)
                self.term_lookup[token] = occurrences

    def encode_docs(self, processor) -> None:
        """
        For DPR-based retrievers, it makes sense to encode the docs in the Index directly.
        """
        # TODO: Allow passing of custom batch sizes or device?
        device = 0 if torch.cuda.is_available() else 1
        batch_size = 32
        # Squeeze necessary because for some reason they have a weird shape.
        self.encoded_documents = np.squeeze(np.array(processor(self.sources, batch_size=batch_size, device=device)),
                                            axis=1)
