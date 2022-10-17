import unittest

from summaries.baselines import lead_3, lead_k, lexrank_st


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
        self.assertEqual(lead_3(input_text), " ".join(expected_result))
        # Assert it works with shorter inputs
        self.assertEqual(lead_3(input_text[:2]), " ".join(expected_result[:2]))
        # Empty input should raise a ValueError
        self.assertRaises(ValueError, lead_3, [])

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

        self.assertEqual(lead_k(input_text, k=4), " ".join(expected_result[:4]))
        self.assertEqual(lead_k(input_text, k=2), " ".join(expected_result[:2]))

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
        self.assertEqual(lead_k(" ".join(input_text), k=3, lang="de"), " ".join(expected_result[:3]))

    def test_lexrank_st_baseline(self):

        input_text = ["This is a reference text.",
                      "It consists of multiple sentences.",
                      "This is the third sentence.",
                      "This is a fourth sentence.",
                      "This sentence should no longer be in the summary."]

        expected_result = ["It consists of multiple sentences.",
                           "This is the third sentence.",
                           "This is a fourth sentence."]

        self.assertEqual(lexrank_st(input_text, num_sentences=3), " ".join(expected_result[:3]))

    def test_lexrank_st_baseline_with_max_length(self):
        input_text = ["This is a reference text.",
                      "It consists of multiple sentences.",
                      "This is the third sentence.",
                      "This is a fourth sentence.",
                      "This sentence should no longer be in the summary."]

        expected_result = ["It consists of multiple sentences.",
                           "This is the third sentence.",
                           "This is a fourth sentence."]

        self.assertEqual(lexrank_st(input_text, max_length=90), " ".join(expected_result[:3]))


