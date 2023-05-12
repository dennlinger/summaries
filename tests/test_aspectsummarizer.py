"""
Tests for aspect summarizer class.
"""
import unittest

from summaries import AspectSummarizer


class TestAspectSummarizer(unittest.TestCase):
    def test_summarizer_init(self):
        summ = AspectSummarizer(segment_level="sentence")
