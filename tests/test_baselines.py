import unittest

from summaries.baselines import lead3_baseline


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
