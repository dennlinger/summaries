"""
Tests for document representations.
"""
import unittest

from summaries.document import Document, Paragraph, Sentence
from summaries.utils import get_nlp_model


class TestDocument(unittest.TestCase):
    def test_document_init(self):
        raw_text = "Dies ist ein Test für Dokumente. Er beinhaltet zwei Sätze über das Dokument."
        doc = Document(raw_text, determine_paragraphs=False, text_lang="de")

        self.assertEqual(doc.raw_text, raw_text)
        self.assertEqual(len(doc.text), 2)


class TestParagraph(unittest.TestCase):
    def test_paragraph_init(self):
        nlp = get_nlp_model("sm", lang="de")
        paragraph_text = ["Dies ist ein Paragraph.", "Er beinhaltet zwei Sätze über Paragraphen."]
        paragraph = Paragraph([Sentence(nlp(sentence)) for sentence in paragraph_text], determine_temporal_tags=False)


if __name__ == '__main__':
    unittest.main()
