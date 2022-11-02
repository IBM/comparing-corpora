import corpus_metrics
from TransformerTextEmbedder import TransformerTextEmbedder

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

distance = corpus_metrics.chi_square_distance(sentences1=setA, sentences2=setB)
print("distance={}".format(distance))

distance = corpus_metrics.fid_distance(sentences1=setA, sentences2=setB)
print("distance={}".format(distance))

distance = corpus_metrics.dc_distance(sentences1=setA, sentences2=setB)
print("distance={}".format(distance))

distance = corpus_metrics.pr_distance(sentences1=setA, sentences2=setB)
print("distance={}".format(distance))

distance = corpus_metrics.mauve_distance(sentences1=setA, sentences2=setB)
print("distance={}".format(distance))

distance = corpus_metrics.zipf_distance(sentences1=setA, sentences2=setB)
print("distance={}".format(distance))

distance = corpus_metrics.medoid_distance(sentences1=setA, sentences2=setB)
print("distance={}".format(distance))

distance = corpus_metrics.ttest_distance(sentences1=setA, sentences2=setB)
print("distance={}".format(distance))

embedder = TransformerTextEmbedder()
embeddingA = embedder.embed_sentences(setA)
embeddingB = embedder.embed_sentences(setB)

distance = corpus_metrics.fid_distance(embeddings1=embeddingA, embeddings2=embeddingB)
print("distance={}".format(distance))

distance = corpus_metrics.dc_distance(embeddings1=embeddingA, embeddings2=embeddingB)
print("distance={}".format(distance))

distance = corpus_metrics.pr_distance(embeddings1=embeddingA, embeddings2=embeddingB)
print("distance={}".format(distance))

distance = corpus_metrics.mauve_distance(embeddings1=embeddingA, embeddings2=embeddingB)
print("distance={}".format(distance))

distance = corpus_metrics.medoid_distance(embeddings1=embeddingA, embeddings2=embeddingB)
print("distance={}".format(distance))

distance = corpus_metrics.ttest_distance(embeddings1=embeddingA, embeddings2=embeddingB)
print("distance={}".format(distance))

embedder = TransformerTextEmbedder()
tokensA = embedder.tokenize_sentences(setA)
tokensB = embedder.tokenize_sentences(setB)

distance = corpus_metrics.chi_square_distance(tokens1=tokensA, tokens2=tokensB)
print("distance={}".format(distance))

distance = corpus_metrics.zipf_distance(tokens1=tokensA, tokens2=tokensB)
print("distance={}".format(distance))
