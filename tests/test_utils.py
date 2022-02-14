import unittest

import spacy

from summaries.analysis import DensityPlot


class TestDensityPlot(unittest.TestCase):
    def setUp(self):
        self.plot = DensityPlot()

    def test_find_closest_reference_match(self):

        extracted_doc = self.plot.processor("Das ist ein Test.")
        reference_doc = self.plot.processor("Ein längerer Inhalt. Er besteht aus mehreren Tests. Das ist ein Test.")

        self.assertEqual([2/3], self.plot.find_closest_reference_match(extracted_doc, reference_doc))

    def test_max_rouge_2_match(self):

        extracted_sentence = list(self.plot.processor("Er besteht aus mehreren merkwürdigen Tests").sents)[0]
        reference_doc = self.plot.processor("Ein längerer Inhalt. Er besteht aus mehreren Tests. Das ist ein Test.")

        self.assertEqual(1/3, self.plot.max_rouge_2_match(extracted_sentence, reference_doc))


if __name__ == '__main__':
    unittest.main()
