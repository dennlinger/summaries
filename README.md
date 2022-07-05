# `aspect-summaries`: An Aspect-based Framework for Summarization

Author: Dennis Aumiller  
Heidelberg University

## Installation
During development you can install this framework by following the steps below:

1. Clone this github repository
2. Add all necessary dependencies with `python3 -m pip install -r requirements`
3. Install the library with `python3 -m pip install .`. If you want an automatically updated development version, you can also add `-e` to the command.

## Usage
For each of the different functionalities, there is a script in `examples/` illustrating the basic use.

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

```python3
from summaries.aligners import SentenceRougeNAligner

# Use ROUGE-2 optimization, with F1 scores as the maximizing attribute
aligner = SentenceRougeNAligner(n=2, optimization_attribute="fmeasure")
# Inputs can either be a raw document (string), or pre-split (sentencized) inputs (list of strings). 
relevant_source_sentences = aligner.extract_source_sentences(summary_text, source_text)
```


#### `SentenceTransformerAligner`
This method works similar in its strategy to the `RougeNAligner`, but instead uses a `sentence-transformer` model to compute the similarity between source and summary sentences (by default, this is `paraphrase-multilingual-MiniLM-L12-v2`).


## Extending or Supplying Own Components

## Citation
TODO
