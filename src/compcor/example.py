import compcor.corpus_metrics as corpus_metrics
from compcor.text_tokenizer_embedder import STTokenizerEmbedder

### Example code

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
		'how do you say hello in mexico'
		]

distance = corpus_metrics.fid_distance(corpus1=setA, corpus2=setB)
print("fid_distance={}".format(distance))

distance = corpus_metrics.dc_distance(corpus1=setA, corpus2=setB)
print("dc_distance={}".format(distance))

distance = corpus_metrics.pr_distance(corpus1=setA, corpus2=setB)
print("pr_distance={}".format(distance))

distance = corpus_metrics.mauve_distance(corpus1=setA, corpus2=setB)
print("mauve_distance={}".format(distance))


distance = corpus_metrics.medoid_distance(corpus1=setA, corpus2=setB)
print("medoid_distance={}".format(distance))

distance = corpus_metrics.ttest_distance(corpus1=setA, corpus2=setB)
print("ttest_distance={}".format(distance))

distance = corpus_metrics.chi_square_distance(corpus1=setA, corpus2=setB)
print("chi_square_distance={}".format(distance))

distance = corpus_metrics.zipf_distance(corpus1=setA, corpus2=setB)
print("zipf_distance={}".format(distance))

print("Comparing corpora on embedding data...")

embedder = STTokenizerEmbedder(embedding_model_name="all-MiniLM-L12-v2")
embeddingA = embedder.embed_sentences(setA)
embeddingB = embedder.embed_sentences(setB)

distance = corpus_metrics.fid_distance(corpus1=embeddingA, corpus2=embeddingB)
print("fid_distance={}".format(distance))

distance = corpus_metrics.dc_distance(corpus1=embeddingA, corpus2=embeddingB)
print("dc_distance={}".format(distance))

distance = corpus_metrics.pr_distance(corpus1=embeddingA, corpus2=embeddingB)
print("pr_distance={}".format(distance))

distance = corpus_metrics.mauve_distance(corpus1=embeddingA, corpus2=embeddingB)
print("mauve_distance={}".format(distance))

distance = corpus_metrics.medoid_distance(corpus1=embeddingA, corpus2=embeddingB)
print("medoid_distance={}".format(distance))

distance = corpus_metrics.ttest_distance(corpus1=embeddingA, corpus2=embeddingB)
print("ttest_distance={}".format(distance))



embedder = STTokenizerEmbedder()
tokensA = embedder.tokenize_sentences(setA)
tokensB = embedder.tokenize_sentences(setB)

distance = corpus_metrics.chi_square_distance(corpus1=tokensA, corpus2=tokensB)
print("chi_square_distance={}".format(distance))

distance = corpus_metrics.zipf_distance(corpus1=tokensA, corpus2=tokensB)
print("zipf_distance={}".format(distance))
