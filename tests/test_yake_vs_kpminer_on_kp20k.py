"""
This script provides a comparison of results on the test set of the KP20k dataset.
The data has been downloaded from here: https://drive.google.com/open?id=1ZTQEGZSq06kzlPlOv4yGjbUpoDrNxebR
A pre-tokenized version also exists on Huggingface, but that defeats the purpose here.
"""
import json
import os

from pke.unsupervised.statistical import kpminer
from pke import compute_document_frequency
from pke.utils import get_stopwords, load_document_frequency_file
from summaries.extractors import YakeExtractor


def compute_doc_frequencies_for_kp20k(input_file, temp_dir, output_file):
    """
    Generates document frequency counts with the pke function.
    First converts the kp20k training set to individual documents, which are then taken to processing.
    """

    convert_jsonl_into_single_docs(input_file, temp_dir)
    stopwords = get_stopwords("en")
    compute_document_frequency(input_dir=temp_dir, output_file=output_file, extension="txt",
                               language="en", normalization="stemming", stoplist=stopwords)


def convert_jsonl_into_single_docs(input_file, output_dir):
    with open(input_file) as f:
        json_lines = f.readlines()

    for idx, line in enumerate(json_lines):
        data = json.loads(line)

        fn = str(idx).zfill(6) + ".txt"
        with open(os.path.join(output_dir, fn), "w") as f:
            f.write(data["abstract"])


if __name__ == '__main__':

    # Uses the same parameters as in the long journal paper (n-gram size = 3, window size = 1 [per default])
    # Stopwords are automatically loaded if not provided by users.
    yake10 = YakeExtractor(10, max_ngram_size=3)

    # First have to compute the document stats
    input_file = "/home/daumiller/kp20k_training.json"
    temp_dir = "/home/daumiller/kp20k_training_single_files"
    output_file = "/home/daumiller/kp20k_training_doc_freqs.tsv.gz"
    os.makedirs(temp_dir, exist_ok=True)

    # Careful, the execution of this takes quite some time!
    # compute_doc_frequencies_for_kp20k(input_file, temp_dir, output_file)

    # Similarly, for KPMiner we leave the default parameters
    kp = kpminer.KPMiner()
    df = load_document_frequency_file(input_file=output_file)

    test_file = "/home/daumiller/kp20k_test.json"

    with open(test_file) as f:
        lines = f.readlines()

    for line in tqdm(lines):
        sample = json.loads(line)
        gold_keywords = sample["keywords"].split(";")

        predicted_yake = yake10.extract_keywords(sample["abstract"])

        # temporarily write text to file for PKE
        with open("temp.txt", "w") as f:
            f.write(sample["abstract"])

        kp.load_document("temp.txt", language="en", normalization=None)
        kp.candidate_selection(lasf=2)  # Have lower least frequencly due to short texts

        kp.candidate_weighting(df=df, sigma=3.0, alpha=2.3)
        predicted_kp = kp.get_n_best(10)







