# `summaries`: A Toolkit for the Summarization Ecosystem

Author: Dennis Aumiller  
Heidelberg University

## Reproducibility of German Summarization Dataset Experiments
Part of this library has been officially accepted as a long paper at BTW'23!
If you are interested in reproducing the contents of this work, see the file `REPRODUCIBILITY.md`.

## Installation
During development you can install this framework by following the steps below:

1. Clone this github repository: `git clone https://github.com/dennlinger/summaries.git`
2. Navigate to the repository folder, and install all necessary dependencies: `python3 -m pip install -r requirements.txt` 
3. Set up the library with `python3 -m pip install .`. If you want an automatically updated development version, you can also add `-e` to the command.

You can now import the library with `import summaries`

## Usage
For some of the functionalities, there are existing scripts in `examples/` illustrating the basic use, or `experiments/`, documenting some concrete experimentation surrounding different (predominantly German) summarization datasets.

### Pre-Processing Data

Often overlooked is a sensible exploratory data analysis and thorough data pre-processing when working in a ML context.
The `summaries` package provides a number of functionalities surrounding this aspect, with a particular focus on summarization-specific filters and analysis functions.

#### `summaries.Analyzer`

The main purpose of the `Analyzer` class is to serve a collection of different tools for inspecting datasets both at the level of a singular sample or the entire subset of training/validation/test splits.
Currently, the `Analyzer` offers the following functionalities:

- `count_ngram_repetitions`: For a single text sample, will count $n$-gram repetitions. Helpful to primarily analyze generated samples.
- `lcs_overlap_fraction`: For a single reference-summary pair, will compute the longest common subsequence (LCS), divided by the length of the summary. A high score indicates that the summary is highly extractive.
- `ngram_overlap_fraction`: Similar to `lcs_overlap_fraction`, but utilizes $n$-gram occurrences to determine similarity.
- `novel_ngram_fraction`: Inverse score (`1 - ngram_overlap_fraction`), instead giving the fraction of $n$-grams that are novel in the summary with respect to the reference.
- `is_fully_extractive`: A less flexible, but decent heuristic to check for fully extractive samples. Works language-independent and much faster than other methods, since all it does is checking whether `summary in reference` evaluates to `True` or `False`.
- `is_summary_longer_than__reference`. Checks whether the summary text is longer than the reference. Can be specified to operate at different levels. Currently supported are `char` (default and fastest method), `whitespace` (approximate tokenization with whitespace splitting), or `token` (will use the `Analyzer` language processor to split into more accurate tokens). In most scenarios, `char` is a sufficient approximation for the length.
- `is_text_too_short`: Checks whether a supplied text is shorter than a minimum length. Supports the same metrics as `is_summary_longer_than_reference`; the minimum length requirement is in the units specified for `length_metric`.
- `is_either_text_empty`: Checks whether either of reference or summary are empty. Currently, this strips basic whitespace characters (e.g., space, newlines, tabs), but does not account for symbols in other encodings, such as `\xa0` in Unicode, or `&nbsp;` in HTML.
- `is_identity_sample`: Checks whether the reference and summary are exactly the same. Could also be checked with `is_fully_extractive`, but will hopefully have more extensive comparison methods in the future, where near-duplicates due to different encodings would also be caught.

Code example of detecting a faulty summarization sample:
```python
from summaries import Analyzer

analyzer = Analyzer(lemmatize=True, lang="en")

# An invalid summarization sample
reference = "A short text."
summary = "A slightly longer text."

print(analyzer.is_summary_longer_than_reference(summary, reference, length_metric="char"))
# True
```

#### `summaries.analysis.Stats`
An additional module similar to `Analyzer`, but more focused on dataset-wide computation of length statistics.

Offers the following functions:

- `density_plot`: Will generate a graph from a collection of references and summaries, split into sentences. For each sentence in the summary, this will determine the relative posiiton of the most related sentence in the reference text. The plot shows the aggregate across all sentences.
- `compute_length_statistics`: As the name suggests, computes length statistics for a dataset.


#### `summaries.Cleaner`

By itself, the `Analyzer` can already be used to streamline exploratory data analysis, however, more frequently the problematic samples should directly be removed from the dataset.
For this purpose, the library provides `summaries.Cleaner`, which internally uses a number of functionalities from `Analyzer` to remove samples.
In particular, for the main functionality `Cleaner.clean_dataset()`, it takes different splits of a dataset (splits are entirely optional), and will remove samples based on set criteria.
For inputs, `Cleaner` either accepts a list of `dict`-like data instances, or alternatively splits derived from a Huggingface `datasets.dataset`.
Additionally, the function will print a distribution of filtered sample by reason for filtering.

Currently, the following filters are applied:

- If `min_length_summary` is set, will remove any sample where the *summary* is shorter than this threshold (in `length_metric` units).
- Similarly, if `min_length_reference` is set, will remove any sample where the *reference text* is shorter than the specified threshold.
- If a sample's reference text and summary text are the exact same, the sample will be removed ("identity samples").
- Samples where the summary is longer than the reference (based on the specified `length_metric`) will be removed.
- If the `extractiveness` parameter is specified, will remove samples that do not satisfy the `extractiveness` criterion. Primarily accepts a `Tuple[float, float]`, which specifies a range in the interval $[0.0, 1.0]$, giving upper and lower bounds for the $n$-gram overlap between reference and summary texts. If a sample does not fall within the range, it will be discarded. Alternatively, also takes `fully` as an accepted parameter, which will filter out only those samples where the summary is *fully extractive* (see above description in the `Analyzer` section).
- Additionally, `Cleaner` will filter out duplicate samples, if the deduplication method is set to something other than `none`. For deduplication method `first`, the first encountered instance of a duplicate will be retained, and any further occurrences be removed. When talking about duplicates, we refer to samples where *either one* of the summary or reference matches a previously encountered text. This avoids ambiguity in the training process. Currently, `first` primarily retains instances in the training set, but would remove more in other splits (validation or test splits). Alternatively `test_first` works on the same general principle of keeping the first encountered instance, but reverses the order in which splits are iterated. `first` uses `(train, validation, test)`, `test_first` works on `(test, validation, train)` instead.

Duplications are expressed as four different types:
1. `exact_duplicate`, where the exact combination of `(reference, summary)` has been encountered before.
2. `both_duplicate`, where both the reference and summary have been encountered before, but in separate instances.
3. `reference_duplicate`, where only the reference has been encountered before.
4. `summary_duplicate`, where only the summary has been encountered before.

Code example of filtering a Huggingface dataset:
```python
from datasets import load_dataset
from summaries import Analyzer, Cleaner

analyzer = Analyzer(lemmatize=True, lang="de")
cleaner = Cleaner(analyzer, min_length_summary=20, length_metric="char", extractiveness="fully")

# The German subset of MLSUM has plenty of extractive samples that need to be filtered
data = load_dataset("mlsum", "de")

clean_data = cleaner.clean_dataset("summary", "text", data["train"], data["validation"], data["test"])
```

### `AspectSummarizer`
The main functionality is a summarizer that is based around a two-stage framework, that starts with a topical extraction component (keyphrase extraction at the moment), and uses these keyphrases as queries in a second stage retriever.

Currently, there are the following options for the respective `Extractor` and `Retriever` components:

- `Extractor`: The method to extract keyphrases from the text.
    - `YakeExtractor`: Uses [Yake](https://github.com/LIAAD/yake) to generate keyphrases from the text. Works reasonably well on a variety of texts that are similar to existing data sets (scholarly articles, newspapers, for example).
    - `OracleExtractor`: Allows users to pass a list of custom keyphrases to the algorithm. Both useful for debugging `Retriever` stages, as well as incorporating prior knowledge into the model.
- `Retriever`: Component to actually extract sentences from a source text as part of a summary.
    - `FrequencyRetriever`: Works with a simple term-based frequency scoring function, that selects the sentences with the highest overlap in lemmatized query tokens. Importantly, this could be improved with some IDF weighting, since individual excerpts (currently: sentences) can be inversely weighted that way.
    - `DPRRetriever`: This model is based on Dense Passage Retrieval, and uses a neural query and context encoder to search for relevant passages.

Per default, the `AspectSummarizer` will retriever *k* sentences for each of *N* topics. For single document summarization use cases, the resulting list of sentences will be ordered by the original sentence order, and also remove any duplicate sentences (this can occur if a sentence is relevant for several different topics).

### Alignment Strategies
For the creation of suitable training data (on a sentence level), it may be necessary to create alignments between source and summary texts.
In this toolkit, we provide several approaches to extract alignments.

#### `RougeNAligner`
This method follows prior work (TODO: Insert citation) in the creation of alignments, based on ROUGE-2 maximization. There are slight differences, however.
Whereas prior work uses a greedy algorithm that adds sentences until the metric is saturated, we proceed by adding a 1:1 alignment for each sentence in the summary.
This has both the advantage of covering a wider range of the source text (for some summary sentences, alignments might appear relatively late in the text), however, at the cost of getting stuck in a local minimum. Furthermore, 1:1 alignments are not the end-all truth, since sentence splitting/merging are also frequent operations, which are not covered with this alignment strategy.

**Usage:**
```python
from summaries.aligners import RougeNAligner

# Use ROUGE-2 optimization, with F1 scores as the maximizing attribute
aligner = RougeNAligner(n=2, optimization_attribute="fmeasure")
# Inputs can either be a raw document (string), or pre-split (sentencized) inputs (list of strings). 
relevant_source_sentences = aligner.extract_source_sentences(summary_text, source_text)
```


#### `SentenceTransformerAligner`
This method works similar in its strategy to the `RougeNAligner`, but instead uses a `sentence-transformer` model to compute the similarity between source and summary sentences (by default, this is `paraphrase-multilingual-MiniLM-L12-v2`).


## Evaluation

### Baseline Methods
The library provides unsupervised baselines for comparison. In particular, we implement the `lead_3`, `lead_k` and a modified LexRank baseline.

`lead_3` and `lead_k` simply copy and return the first few sentences of the input document as a summary. `lead_3` was mainly popularized by [(Nallapati et al, 2016)](https://arxiv.org/pdf/1611.04230.pdf). [Our own work](https://arxiv.org/abs/2201.07198) introduces a variant that accounts for slightly longer contexts, which is espeically useful for long-form summaries (e.g., Wikipedia or legal documents), where 3 sentences vastly underestimates the expected output length.

For the `lexrank_st` baseline, we adapt the modification suggested by [Nils Reimers](https://twitter.com/Nils_Reimers/status/1488213682236661774), which replaces the centrality computation with cosine similarity over the segment embeddings generated by `sentence-transformers` models.

By default, all the baselines will utilize a language-specific

**Usage:**
```python3
from summaries.baselines import lead_3, lexrank_st
import spacy

lead_3(input_text, lang="en")
lexrank_st(input_text, lang="en")

# or, alternatively:
nlp = spacy.load("en_core_web_sm")
lead_3(input_text, processor=nlp)
lexrank_st(input_text, processor=nlp)
```

### Significance Testing
For the sake of reproducible research, we also provide a simple implementation of paired bootstrap resampling, following [(Koehn, 2004)](https://aclanthology.org/W04-3250.pdf).
It allows the comparison of two systems, A and B, on a gold test set. The hypothesis is that *system A outperforms B*. The returned score is the p-value.

**Usage:**
```python3
from summaries.evaluation import paired_bootstrap_test

# Replace with any metric of your choice, but make sure it takes
# litss of system and gold inputs and returns a singular float "score"
def accuracy(system, gold):
    return sum([s == g for s, g in zip(system, gold)]) / len(system)
    
# By default performs 10k iterations of re-sampling
paired_bootstrap_test(gold_labels,
                      system_a_predictions,
                      system_b_predictions,
                      scoring_fucntion=accuracy,
                      n_resamples=1000,
                      seed=12345)
```

## Extending or Supplying Own Components

## Citation
If you found this library useful, please consider citing the following work:

```
@inproceedings{aumiller-etal-2023-on,
  author    = {Dennis Aumiller and
               Jing Fan and
               Michael Gertz},
  title     = {{On the State of German (Abstractive) Text Summarization}},
  booktitle = {Datenbanksysteme f{\"{u}}r Business, Technologie und Web {(BTW}
               2023)},
  series    = {{LNI}},
  publisher = {Gesellschaft f{\"{u}}r Informatik, Bonn},
  year      = {2023}
}
```
