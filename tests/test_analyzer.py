"""
Collection of tests to verify that the Analyzer tools are working as intended.
"""

import unittest

from summaries.analysis import Analyzer


class TestAnalyzer(unittest.TestCase):

    def test_is_fully_extractive(self):
        test_summary = "Dies ist eine Zusammenfassung."
        test_reference = "Hier steht ein längerer Text. Dies ist eine Zusammenfassung."
        analyzer = Analyzer(lang="de")

        is_extractive = analyzer.is_fully_extractive(test_summary, test_reference)
        self.assertEqual(True, is_extractive)

        test_alternative_summary = "Dies ist keine Zusammenfassung."
        is_extractive = analyzer.is_fully_extractive(test_alternative_summary, test_reference)
        self.assertEqual(False, is_extractive)