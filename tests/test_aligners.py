import unittest

from summaries.aligners import SentenceRougeNAligner
from summaries.utils import RelevantSentence


class TestRougeNAligner(unittest.TestCase):

    def test_init(self):
        aligner = SentenceRougeNAligner(n=2, optimization_attribute="fmeasure")

    def test_extract_source_sentences(self):
        aligner = SentenceRougeNAligner(n=2, optimization_attribute="fmeasure")

        # First test by passing sentencized inputs
        gold = ["This is a test.", "This is another, worse test."]
        system = ["This is a test."]

        result = aligner.extract_source_sentences(system, gold)
        expected_result = [RelevantSentence(gold[0], 1.0, 0.0)]
        self.assertEqual(expected_result, result)