"""
Generates a list of relevant sentences in the source text that best align with a target summary's individual sentences.
For this purpose, for each sentence a maximizing fragment (according to ROUGE-2) is chosen from the input text.
Currently only works for SDS cases (i.e., single reference text).
This method differs from the implementation used in related work, which is primarily based on the method introduced by
Nallapati et al. in their 2016 SummaRuNNer paper. Their method is implemented as the GreedyAligner in this library.
"""
from typing import List

from rouge_score.rouge_scorer import _create_ngrams

from ..utils import max_rouge_n_match, RelevantSentence, valid_optimization_attributes
from .AlignerBase import Aligner


class RougeNAligner(Aligner):

    optimization_attribute: str
    n: int
    lemmatize: bool

    # TODO: Unify the interface for passing a language processor with the Analyzer (and potentially other classes)
    def __init__(self, n: int = 2, optimization_attribute: str = "recall", lemmatize: bool = False, lang: str = "de"):
        """
        Initializes a RougeNAligner
        :param n: N-gram length to consider. Default choice in literature is ROUGE-2, although ROUGE-1 can be used, too.
        :param optimization_attribute: Either one of "precision", "recall", or "fmeasure" to optimize for.
        :param lemmatize: Boolean flag indicating whether or not to use lemmas instead of the raw token texts.
        :param lang: Language code for the underlying tokenizer & optional lemmatizer.
        """
        super(RougeNAligner, self).__init__(lang=lang)

        self.n = n
        if optimization_attribute not in valid_optimization_attributes:
            raise ValueError(f"optimization_attribute must be either of {valid_optimization_attributes}!")
        self.optimization_attribute = optimization_attribute
        self.lemmatize = lemmatize

    def _process_string_inputs(self, summary: str, reference: str) -> List[RelevantSentence]:
        """
        Method that additionally uses spacy to sentencize content before matching.
        """
        relevant_sentences = []
        summary_doc = self.processor(summary)
        reference_doc = self.processor(reference)

        # TODO: Could extend the n-gram creation to actual split words, which would improve for compounds!
        if self.lemmatize:
            reference_ngrams = [_create_ngrams([token.lemma_ for token in sentence], n=self.n)
                                for sentence in reference_doc.sents]
        else:
            reference_ngrams = [_create_ngrams([token.text for token in sentence], n=self.n)
                                for sentence in reference_doc.sents]
        reference_sentences = [sentence.text for sentence in reference_doc.sents]

        for sentence in summary_doc.sents:
            relevant_sentences.append(max_rouge_n_match(sentence,
                                                        reference_sentences,
                                                        reference_ngrams,
                                                        self.n,
                                                        self.optimization_attribute,
                                                        self.lemmatize))

        return relevant_sentences

    def _process_sentencized_inputs(self, summary: List[str], reference: List[str]) -> List[RelevantSentence]:
        """
        Method to process already sentencized inputs. Despite the additionally processed text,
        this method is slightly slower, since obtaining lemmas still requires processing with spacy.
        """
        relevant_sentences = []
        summary_doc = [self.processor(sentence) for sentence in summary]
        reference_doc = [self.processor(sentence) for sentence in reference]

        if self.lemmatize:
            reference_ngrams = [_create_ngrams([token.lemma_ for token in sentence], n=2)
                                for sentence in reference_doc]
        else:
            reference_ngrams = [_create_ngrams([token.text for token in sentence], n=2)
                                for sentence in reference_doc]

        # This simplifies for already split sentences
        reference_sentences = reference

        for sentence in summary_doc:
            relevant_sentences.append(max_rouge_n_match(sentence,
                                                        reference_sentences,
                                                        reference_ngrams,
                                                        self.n,
                                                        self.optimization_attribute,
                                                        self.lemmatize))

        return relevant_sentences


