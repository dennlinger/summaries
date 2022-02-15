"""
Demonstration of Rouge2Aligner
"""

from summaries.aligners import Rouge2Aligner


def load_text_file(fn: str) -> str:
    with open(fn, "r") as f:
        text = f.readlines()

    text = [line.strip("\n ") for line in text if line.strip("\n ") and not line.startswith("=")]
    return " ".join(text)


if __name__ == '__main__':
    reference = load_text_file("./Aachen_Wiki.txt")
    summary = load_text_file("Aachen_Klexikon.txt")

    aligner = Rouge2Aligner()

    extracted_sentences = aligner.extract_source_sentences(summary=summary, reference=reference)
    summary_doc = aligner.processor(summary)

    for summary_sentence, reference in zip(summary_doc.sents, extracted_sentences):
        print(summary_sentence)
        print(reference)
        print("------------------")
