import unittest

from summaries.baselines import lead3_baseline, leadk_baseline, lexrank_st_baseline


class TestBaselines(unittest.TestCase):

    def test_lead3(self):
        input_text = ["This is a reference text.",
                      "It consists of multiple sentences.",
                      "This is the third sentence.",
                      "This sentence should no longer be in the summary."]

        expected_result = ["This is a reference text.",
                           "It consists of multiple sentences.",
                           "This is the third sentence."]

        # Base case
        self.assertEqual(lead3_baseline(input_text), " ".join(expected_result))
        # Assert it works with shorter inputs
        self.assertEqual(lead3_baseline(input_text[:2]), " ".join(expected_result[:2]))
        # Empty input should raise a ValueError
        self.assertRaises(ValueError, lead3_baseline, [])

    def test_leadk_baseline(self):
        input_text = ["This is a reference text.",
                      "It consists of multiple sentences.",
                      "This is the third sentence.",
                      "This is a fourth sentence.",
                      "This sentence should no longer be in the summary."]

        expected_result = ["This is a reference text.",
                           "It consists of multiple sentences.",
                           "This is the third sentence.",
                           "This is a fourth sentence."]

        self.assertEqual(leadk_baseline(input_text, k=4), " ".join(expected_result[:4]))
        self.assertEqual(leadk_baseline(input_text, k=2), " ".join(expected_result[:2]))

    def test_leadk_baseline_with_processor(self):
        input_text = ["This is a reference text.",
                      "It consists of multiple sentences.",
                      "This is the third sentence.",
                      "This is a fourth sentence.",
                      "This sentence should no longer be in the summary."]

        expected_result = ["This is a reference text.",
                           "It consists of multiple sentences.",
                           "This is the third sentence.",
                           "This is a fourth sentence."]
        self.assertEqual(leadk_baseline(" ".join(input_text), k=3, lang="de"), " ".join(expected_result[:3]))

    def test_lexrank_st_baseline(self):

        input_text = ["This is a reference text.",
                      "It consists of multiple sentences.",
                      "This is the third sentence.",
                      "This is a fourth sentence.",
                      "This sentence should no longer be in the summary."]

        expected_result = ["It consists of multiple sentences.",
                           "This is the third sentence.",
                           "This is a fourth sentence."]

        self.assertEqual(lexrank_st_baseline(input_text, num_sentences=3), " ".join(expected_result[:3]))

    def test_lexrank_st_baseline_with_max_length(self):
        input_text = ["This is a reference text.",
                      "It consists of multiple sentences.",
                      "This is the third sentence.",
                      "This is a fourth sentence.",
                      "This sentence should no longer be in the summary."]

        expected_result = ["It consists of multiple sentences.",
                           "This is the third sentence.",
                           "This is a fourth sentence."]

        self.assertEqual(lexrank_st_baseline(input_text, max_length=90), " ".join(expected_result[:3]))


