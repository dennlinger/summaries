import unittest

from summaries.retrievers import FrequencyRetriever
from summaries.utils import get_nlp_model


class TestFrequencyRetriever(unittest.TestCase):

    def test_split_tokens(self):
        nlp = get_nlp_model("sm", disable=tuple("ner"), lang="de")
        retriever = FrequencyRetriever()
        self.assertEqual(retriever.split_tokens("Hildebrand und Söhne Co. K.G.", nlp),
                         ["Hildebrand", "und", "Söhne", "Co.", "K.G."])

    def test_empty_split_tokens(self):
        nlp = get_nlp_model("sm", disable=tuple("ner"), lang="de")
        retriever = FrequencyRetriever()
        self.assertRaises(ValueError, retriever.split_tokens, "", nlp)


if __name__ == '__main__':
    unittest.main()
