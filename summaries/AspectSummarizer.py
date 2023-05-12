"""
Base class file, containing the holistic pipeline.
"""

import warnings
from typing import Union, Optional
from itertools import count

from spacy.language import Language

from .extractors import Extractor, LeadExtractor
from .generators import Generator, OpenAIGenerator, ExtractiveGenerator, VALID_OPENAI_MODEL_NAMES
from .index import Index
from .document import Document, Segment
from .utils import get_nlp_model, interpret_lang_code


class AspectSummarizer:
    # Meta-information regarding the pre-processing of documents.
    segment_level: str
    lang: str
    processor: Language
    # Generation-side part of the model part of the model.
    generator: Generator
    # Storage solution for the intermediate document representation.
    index: Index
    segments = list[Segment]

    def __init__(self,
                 segment_level: str = "sentence",
                 lang: str = "en",
                 generator: Union[str, Generator] = "gpt-3.5-turbo"):
        """
        # TODO Explain init function.
        :param segment_level: Specifies at what granularity an input text is split. Currently supports
            - "document" (document-level distinction, primarily intended for short texts, e.g., social media posts)
            - "paragraph" (paragraph-level split, segmenting at double newlines "\n\n")
            - "sentence" (default value; splits at the sentence-level with the help of spaCy)
        :param lang: Language identifier or code for the language in which texts are. Default is English.
        :param generator: What
        """
        self.segment_level = segment_level
        self.lang = interpret_lang_code(lang)
        # FIXME: Proper loading of model with arguments
        self.processor = get_nlp_model("sm", disable=tuple("ner"), lang=self.lang)

        # TODO: Enable passing of config files with attributes
        self._assign_generator(generator)

        # Initialize empty representation
        self.segments = []

    def _assign_generator(self, generator: Union[str, Generator]) -> None:

        # FIXME: Can cause problems with custom classes overwriting base!
        if isinstance(generator, Generator):
            self.generator = generator
        elif generator in VALID_OPENAI_MODEL_NAMES:
            self.generator = OpenAIGenerator(model_name=generator)
        elif generator == "extractive":
            self.generator = ExtractiveGenerator()
        else:
            raise NotImplementedError("Other Generator models currently not supported!")

    def summarize(self, source_texts: Optional[Union[str, list[str]]] = None,
                  ex_ante_aspects: Optional[dict] = None,
                  ex_post_aspects: Optional[dict] = None,
                  segment_limit_intermediate_representation: Optional[int] = None):
        """
        Generates an aspect-focused summary over a collection of documents, based on user-specified query-parameters.
        :param source_texts: Single input text, or otherwise collection (list) of several input texts.
            TODO: Improve design for repeated queries to the same input document collection
            Leave empty (None) if you want to generate summaries over a previously specified document collection
        :param ex_ante_aspects: Dictionary containing the desired ex-ante (extractive) aspects and the parameters.
            # TODO: Add list of all accepted parameters
            Example:
                ex_ante_aspects = {
                    "length": 500,  # assumed to be in characters
                    "query": ["Barack Obama", "Childhood", "Political Career"],  # three different queries
                    ...
                }
        :param ex_post_aspects: Dictionary containing the desired ex-post (generative) aspects and their parameters.
            # TODO: Add list of all accepted parameters
        :param segment_limit_intermediate_representation: Limits the number of segments passed to the ex-post stage,
            based on the "most relevant" segments from the ex-ante stage. This utilizes a function based on simple
            linear combination, but # TODO: can be adjusted in the init function if necessary.
        :return: An aspect-focused summary generated with the respective specified aspects.
        """
        # Skip processing of documents if querying an existing collection
        if source_texts is not None:
            # Ensure consistent representation between single/multiple texts
            if isinstance(source_texts, str):
                source_texts = [source_texts]

            # Process input documents into segments
            self._segment_documents(source_texts)

        if ex_ante_aspects is None and ex_post_aspects is None:
            warnings.warn("No aspects specified; will generate generic summary.")

        if source_texts is None and len(self.segments) == 0:
            raise ValueError("No internal representation available!"
                             "You might have accidentally overwritten the previously stored documents,"
                             "or forgot to pass segments along with the specified aspects.")

        # Do extractive filtering part.
        for aspect, param_variants in ex_ante_aspects.items():
            for param in param_variants:
                extractor = self._assign_extractor(aspect, param)

                extractor.rank(self.segments)

        # Generate summary
        # TODO: Adjust inputs to the following function call
        if segment_limit_intermediate_representation is None:
            aspect_summary = self.generator.generate(self.segments, ex_ante_aspects, ex_post_aspects)
        # Filter to a smaller intermediate subset if desired
        else:
            passed_segments = self._filter_intermediate_segments(segment_limit_intermediate_representation)
            aspect_summary = self.generator.generate(passed_segments, ex_ante_aspects, ex_post_aspects)

        return aspect_summary

    def clarify_summary(self, clarification_query: str) -> str:
        """
        Based on an intermediate summary representation, clarify parts of a summary with the original documents.
        :param clarification_query: String of a clarification aspect
        :return: Answer clarifying missing information.
        """

        raise NotImplementedError("Currently not implemented!")

    def _segment_documents(self, source_documents: list[str]) -> None:
        """
        Will turn the provided documents into a more structured format,
        including meta-annotation etc.
        :param source_documents:
        :return: No return value, assigns segments internally to class attribute.
        """
        id_counter = count(0)

        for document in source_documents:
            doc = Document(self.processor(document),
                           document_id=next(id_counter),
                           segment_level=self.segment_level)
            for segment in doc:
                self.segments.append(segment)

    # TODO: Determine best output function?
    def _filter_intermediate_segments(self, segment_limit: int):
        """
        Implements a naive filtering method, selecting the most relevant segments up to a certain number.
        Overall relevance is computed by the sum over individual aspect dimensions.
        :return: Subset of segments, with the highest relevance.
        """
        return sorted(self.segments, key=lambda seg: sum(seg.relevance_vector), reverse=True)[:segment_limit]

    @staticmethod
    def _assign_extractor(aspect, param) -> Extractor:
        if aspect == "lead":
            if isinstance(param, int):
                extractor = LeadExtractor(fixed_cutoff=param)
            elif callable(param):
                extractor = LeadExtractor(decayed_relevance_func=param)
            else:
                raise ValueError("Unrecognized parameter for lead-based ex-ante filter.")
        else:
            raise NotImplementedError("Not implemented yet!")

        return extractor