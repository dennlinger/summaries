"""
Similar to examples/density_plot.py, but computes it on the full WikiLingua (German) dataset.
"""

from datasets import load_dataset

from summaries import Analyzer, Cleaner


def expand_samples(split):
    samples = []
    for sample in split:
        doc = sample["article"]
        try:
            if not len(doc["document"]) == len(doc["summary"]):
                raise ValueError(f"Not the same number of sections! {doc}")
        except KeyError:
            raise ValueError(f"Incomplete sample detected: {doc}")
        for idx in range(len(doc["document"])):
            samples.append({
                "url": sample["url"],
                "section_name": doc["section_name"][idx],
                "document": doc["document"][idx],
                "summary": doc["summary"][idx]
            })
    return samples


if __name__ == '__main__':
    dataset = load_dataset("wiki_lingua", "german")
    train = expand_samples(dataset["train"])

    analyzer = Analyzer(lemmatize=True, lang="de")
    cleaner = Cleaner(analyzer, deduplication_method="test_first",
                      min_length_summary=5, min_length_reference=20, length_metric="char",
                      min_compression_ratio=1.25,
                      extractiveness="fully")

    clean_data = cleaner.clean_dataset("summary", "document", train, enable_tqdm=True)


    # # GEM/wiki_lingua version
    # dataset = load_dataset("GEM/wiki_lingua", "de")
    #
    # analyzer = Analyzer(lemmatize=True, lang="de")
    # cleaner = Cleaner(analyzer, deduplication_method="test_first",
    #                   min_length_summary=5, min_length_reference=20, length_metric="char",
    #                   extractiveness="fully")
    #
    # clean_splits = cleaner.clean_dataset("target", "source", dataset["train"], dataset["validation"], dataset["test"],
    #                                      enable_tqdm=True)