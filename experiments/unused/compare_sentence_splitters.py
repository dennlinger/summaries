"""
Puts NLTK's punkt sentencizer against spaCy's splitting.
Notably, our own splitting combines spac with some post-processing.
"""

from nltk.tokenize import sent_tokenize


if __name__ == '__main__':
    with open("../examples/Aachen_Wiki.txt", "r") as f:
        lines = f.readlines()

    text = " ".join([line.strip("\n ") for line in lines if line != "" and not line.startswith("=")])

    res = sent_tokenize(text)
    for sent in res:
        print(sent)




