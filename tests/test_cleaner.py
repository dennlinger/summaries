"""
Basic set of tests to verify the functionality of the preprocessing tool "Cleaner".
"""
import unittest

from summaries.preprocessing import Cleaner


class TestCleaner(unittest.TestCase):

    def test_minlength_filtering(self):
        cleaner = Cleaner(min_length_summary=5, min_length_reference=15, length_metric="char")

        # Lengths are 15 (reference) and 13 (summary), respectively.
        train_set = [{"reference": "This is a test.", "summary": "Summary text."}]

        cleaned = cleaner.clean_dataset("summary", "reference", train_set)
        self.assertEqual(1, len(cleaned["train"]))

        train_set_too_short_ref = [{"reference": "This is a.", "summary": "Summary text."}]
        cleaned = cleaner.clean_dataset("summary", "reference", train_set_too_short_ref)
        self.assertEqual(0, len(cleaned["train"]))

        train_set_too_short_summ = [{"reference": "This is a test.", "summary": "Summ"}]
        cleaned = cleaner.clean_dataset("summary", "reference", train_set_too_short_summ)
        self.assertEqual(0, len(cleaned["train"]))

    def test_identity_sample(self):
        cleaner = Cleaner()

        # Same input and output should be removed
        train_set = [{"reference": "This is a test.", "summary": "This is a test."}]
        cleaned = cleaner.clean_dataset("summary", "reference", train_set)
        self.assertEqual(0, len(cleaned["train"]))

    def test_shorter_summary_than_reference(self):
        cleaner = Cleaner()

        # Summary is longer than the reference
        train_set = [{"reference": "This is a.", "summary": "This is a test."}]
        cleaned = cleaner.clean_dataset("summary", "reference", train_set)
        self.assertEqual(0, len(cleaned["train"]))

    def test_duplicate_removal(self):
        cleaner = Cleaner(deduplication_method="first")

        # The second sample is not added, so the third one should still be added, despite having the same summary.
        train_set = [{"reference": "This is a test.", "summary": "Summary text."},
                     {"reference": "This is a test.", "summary": "Summary."},
                     {"reference": "This is a.", "summary": "Summary."}]
        cleaned = cleaner.clean_dataset("summary", "reference", train_set)
        self.assertEqual(2, len(cleaned["train"]))

        # This checks similar behavior, but with the summary
        train_set = [{"reference": "This is a test.", "summary": "Summary text."},
                     {"reference": "This is a.", "summary": "Summary text."},
                     {"reference": "This is a test text.", "summary": "Summary."}]
        cleaned = cleaner.clean_dataset("summary", "reference", train_set)
        self.assertEqual(2, len(cleaned["train"]))

    def test_leakage_removal(self):
        cleaner = Cleaner(deduplication_method="first")
        train_set = [{"reference": "This is a test.", "summary": "Summary text."}]
        test_set = [{"reference": "This is a.", "summary": "Summary text."}]
        cleaned = cleaner.clean_dataset("summary", "reference", train_set=train_set, test_set=test_set)
        self.assertEqual(1, len(cleaned["train"]))
        self.assertEqual(0, len(cleaned["test"]))

    def test_fully_extractive_removal(self):
        ngram_sim_range = (0.0, 0.9)
        cleaner = Cleaner(extractiveness=ngram_sim_range)

        # Have to remove the period after the "a", because this is not factored in?
        train_set = [{"reference": "This is a test.", "summary": "This is a"}]
        cleaned = cleaner.clean_dataset("summary", "reference", train_set)

        self.assertEqual(0, len(cleaned["train"]))

    def test_test_first_deduplication(self):
        cleaner = Cleaner(deduplication_method="test_first")
        train_set = [{"reference": "This is a test.", "summary": "Summary text."}]
        test_set = [{"reference": "This is a test sentence.", "summary": "Summary text."}]
        cleaned = cleaner.clean_dataset("summary", "reference", train_set=train_set, test_set=test_set)
        self.assertEqual(0, len(cleaned["train"]))
        self.assertEqual(1, len(cleaned["test"]))


