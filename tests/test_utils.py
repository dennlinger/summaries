import unittest

import spacy

from summaries.utils import find_closest_reference_matches, max_rouge_2_match


class TestDensityPlot(unittest.TestCase):
    def setUp(self):
        self.processor = spacy.load("de_core_news_sm", disable=("ner",))

    def test_find_closest_reference_match(self):

        extracted_doc = self.processor("Das ist ein Test.")
        reference_doc = self.processor("Ein längerer Inhalt. Er besteht aus mehreren Tests. Das ist ein Test.")

        self.assertEqual([2/3], find_closest_reference_matches(extracted_doc, reference_doc))

    def test_max_rouge_2_match(self):

        extracted_sentence = list(self.processor("Er besteht aus mehreren merkwürdigen Tests").sents)[0]
        reference_doc = self.processor("Ein längerer Inhalt. Er besteht aus mehreren Tests. Das ist ein Test.")

        self.assertEqual(1/3, max_rouge_2_match(extracted_sentence, reference_doc))


if __name__ == '__main__':
    unittest.main()
