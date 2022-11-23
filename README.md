# comparing-corpora
A python library of similarity measures which allow measuring the perceptual similarity between set embeddings corpora.

# Calculating distances between two text corpora

```
import compcor.corpus_metrics as corpus_metrics
from compcor.text_tokenizer_embedder import TextTokenizerEmbedder

setA = ['can you tell me how i would normally say thank you as a french person',
        'can you translate hi into spanish for me',
        'can you translate milk into spanish for me',
        'how can i say thank you very much in chinese',
        'how can i thank somebody in italian',
        'how could i say twin in chinese',
        'how do germans say goodnight',
        'how do i ask about the weather in chinese',
        'how do i say hotel in finnish',
        'how do i say bathroom in italian']

	setB = ['how can i say thank you very much in chinese',
			'how can i thank somebody in italian',
			'how could i say twin in chinese',
			'how do they say tacos in mexico',
			'how do they say yes in brazil',
			'how do vietnameses people say hello',
			'how do you say cat in spanish',
			'how do you say dog in spanish',
			'how do you say fast in spanish',
			'how do you say good bye in french',
			'how do you say goodbye in spanish',
			'how do you say hello in french',
			'how do you say hello in japanese',
			'how do you say hello in mexico']
```

Given two corpora of strings, we want to calculate the distance between them.
`comparing-corpora` provides the following distance metrics.
The first metrics operate on a sentence-level embedding, while the last two operate on the token frequencies.

|Name|function|level|description|
|---|---|---|---|
|T-test|`ttest_distance`|sentence||
|Medoid|`medoid_distance`|sentence||
|Classifier|`classifier_distance`|sentence|Classifiability between reference and target|
|PR (recision and recall)|`PR_distance`|sentence|Assessing distributional precision and recall|
|IRPR (information-retrieval precision and recall)|`IRPR_distance`|sentence|Average distance between closest samples pairs|
|DC (density and coverage)|`dc_distance`|sentence|Estimating manifolds density and coverage|
|MAUVE|`mauve_distance`|sentence|Quality and diversity via divergence frontiers|
|FID (Frechet Inception Distance)|`fid_distance`|sentence|Wasserstein distance between densities|
|Chi-squared ($\chi^2$)|`chi_square_distance`|token|Word/Token count comparison|
|Zipf|`zipf_distance`|token|Unigram rank-frequency statistics|

To calculate inter-corpora distance, the raw corpora can be provided, in which case they will be embedded inside the function; this is determined by the 'model' argument (by default, 'all-MiniLM-L6-v2'), if the distance operates on the sentence level, otherwise by the 'tokenizer' argument.
```
distance = corpus_metrics.fid_distance(corpus1=setA, corpus2=setB)
print("fid_distance={}".format(distance))
#fid_distance=0.7498146370755459
```

The metrics also accept already-embedded corpora (possibly using a custom embedding):
```
embedder = TextTokenizerEmbedder(embedding_model_name = "all-MiniLM-L12-v2")

# due to change of the embedding model the results should differ
embeddingA = embedder.embed_sentences(setA)
embeddingB = embedder.embed_sentences(setB)

distance = corpus_metrics.fid_distance(corpus1=embeddingA, corpus2=embeddingB)
print("fid_distance={}".format(distance))
#fid_distance=0.7671713341732544

tokensA = embedder.tokenize_sentences(setA)
tokensB = embedder.tokenize_sentences(setB)

distance = corpus_metrics.chi_square_distance(corpus1=tokensA, corpus2=tokensB)
print("chi_square_distance={}".format(distance))
#chi_square_distance=0.9987177546738071
```
