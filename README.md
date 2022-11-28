# compcor
A python library of similarity measures which allow measuring the perceptual distance between text corpora.

You can use compcor to easily calculate the perceptual distance between two sets of sentences using many classical and SOTA metrics. 

# About
The ability to compare the semantic similarity between text corpora is important in a variety of natural language processing applications.
While one can reasonably measure the semantic distance between two individual sentences (e.g., by calculating the cosine distance between the sentence embeddings), measuring the dissimilarity between two text corpora remains a challenge.
Corpus-level metrics seek to assess semantic similarity at the group level.
Such metrics are essential for measuring how well corpus-based linguistic analysis generalizes from one data-set to another.
Specifically, the recent advances in generative language models have led to an increased interest in the study of content similarity between human and generated language, as a mean for comparing the quality of generative models.

# Usage
To calculate the perceptual distance between two corpora, the raw corpora can be simply provided as a list of strings:
```
setA = ['can you tell me how i would normally say thank you as a french person', 'can you translate hi into spanish for me', 'can you translate milk into spanish for me', 'how can i say thank you very much in chinese', 'how can i thank somebody in italian', 'how could i say twin in chinese', 'how do germans say goodnight','how do i ask about the weather in chinese', 'how do i say hotel in finnish', 'how do i say bathroom in italian']
setB = ['how can i say thank you very much in chinese', 'how can i thank somebody in italian', 'how could i say twin in chinese', 'how do they say tacos in mexico', 'how do they say yes in brazil', 'how do vietnameses people say hello', 'how do you say cat in spanish', 'how do you say dog in spanish', 'how do you say fast in spanish', 'how do you say good bye in french', 'how do you say goodbye in spanish', 'how do you say hello in french', 'how do you say hello in japanese', 'how do you say hello in mexico']
```

### Using raw data
By calling any metric provided in the package, all metrics return a single scalar capturing the perceptual distance between two corpora:
```
import compcor.corpus_metrics as corpus_metrics
distance = corpus_metrics.fid_distance(corpus1=setA, corpus2=setB)
```
In this case the sentences in both sets will be tokenized/embedded inside the function.

### Using embeddings
The metrics also accept already-embedded/tokenized corpora, possibly using a custom embedding.
```
distance = corpus_metrics.fid_distance(corpus1=embedA, corpus2=embedB)
```

We provide an embedding and tokanization utility class 'STTokenizerEmbedder' which is a shallow wrapper for the [sentence-transformer](https://www.sbert.net/)
SentenceTransformer class.
STTokenizerEmbedder implements two simple interfaces 'TextEmbedder' and 'TextTokenizer'. 
The hugging-face model can be determined using the 'embedding_model_name' parameter.
By default, we use the model 'all-MiniLM-L6-v2', but any other hugging-face model can be used.

```
from compcor.text_tokenizer_embedder import STTokenizerEmbedder
embedder = STTokenizerEmbedder(embedding_model_name = "all-MiniLM-L12-v2")

embedA = embedder.embed_sentences(setA)
embedB = embedder.embed_sentences(setB)
distance = corpus_metrics.fid_distance(corpus1=embedA, corpus2=embedB)
```

### Using tokens
If the distance operates on the tokens level, the tokenized sentences should be provided instead of the sentence embeddings.
```
tokensA = embedder.tokenize_sentences(setA)
tokensB = embedder.tokenize_sentences(setB)

distance = corpus_metrics.chi_square_distance(corpus1=tokensA, corpus2=tokensB)
print("chi_square_distance={}".format(distance))
#chi_square_distance=0.9987177546738071
```
# Full Metric list
Given two corpora of strings, we want to calculate the distance between them.
`comparing-corpora` provides the following distance metrics.
The first metrics operate on a sentence-level embedding, while the last two operate on the token frequencies.

| Name                                              |function| representation | description                                      |
|---------------------------------------------------|---|----------------|--------------------------------------------------|
| t-test                                            |`ttest_distance`| embbeding      | T-test p-value on difference in elementwise means |
| Medoid                                            |`medoid_distance`| embbeding      | Cosine distance between corpora centroids        |
| Classifier                                        |`classifier_distance`| embbeding      | Classifiability between reference and target     |
| PR (precision and recall)                         |`PR_distance`| embbeding      | Assessing distributional precision and recall    |
| IRPR (information-retrieval precision and recall) |`IRPR_distance`| embbeding      | Average distance between closest samples pairs   |
| DC (density and coverage)                         |`dc_distance`| embbeding      | Estimating manifolds density and coverage        |
| MAUVE                                             |`mauve_distance`| embbeding      | Quality and diversity via divergence frontiers   |
| FID (Frechet Inception Distance)                  |`fid_distance`| embbeding      | Wasserstein distance between densities           |
| Chi-squared ($\chi^2$)                            |`chi_square_distance`| token          | Word/Token count comparison                      |
| Zipf                                              |`zipf_distance`| token          | Unigram rank-frequency statistics                |

## Performance and Citation
The quality, time performance and other properties these metrics are studied in the following paper.
```
@inproceedings{kour2022measuring,
  title={Measuring the Measuring Tools: An Automatic Evaluation of Semantic Metrics for Text Corpora},
  author={Kour, George and Ackerman, Samuel and Farchi, Eitan and Raz, Orna and Carmeli, Boaz and Anaby-Tavor, Ateret},
  booktitle={Proceedings of the 2nd Workshop on Natural Language Generation, Evaluation, and Metrics (GEM 2022)},
  publisher = "Association for Computational Linguistics",
  year={2022}
}
```
If you use this package for your scientific publication please cite it.