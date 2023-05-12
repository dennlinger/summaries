"""
Tests for document representations.
"""
import unittest

from summaries.document import Document, Segment
from summaries.utils import get_nlp_model


class TestDocument(unittest.TestCase):
    def test_document_init(self):
        nlp = get_nlp_model("sm", lang="de")
        raw_text = "Dies ist ein Test für Dokumente. Er beinhaltet zwei Sätze über das Dokument."
        processed = nlp(raw_text)
        doc = Document(processed, 0, segment_level="document")

        self.assertEqual(doc.raw_text, raw_text)
        self.assertEqual(doc.document_id, 0)
        self.assertEqual(len(doc), 1)

        doc = Document(processed, 0, segment_level="sentence")
        self.assertEqual(len(doc), 2)
        self.assertEqual(doc.segments[0].parent_doc, doc)


# class TestSegment(unittest.TestCase):
#     def test_paragraph_init(self):
#         paragraph_text = ["Dies ist ein Paragraph.", "Er beinhaltet zwei Sätze über Paragraphen."]


if __name__ == '__main__':
    unittest.main()
