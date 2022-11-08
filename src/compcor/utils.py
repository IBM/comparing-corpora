import numpy as np
import operator
from sklearn import linear_model
from collections import Counter
from typing import List, Union

Corpus = Union[List[str], List[List[float]]]
TCorpus = Union[List[str], List[List[str]]]


def get_corpora_embeddings(corpus1, corpus2, model):
    if isinstance(corpus1[0], str):
        embeddings1 = model.embed_sentences(corpus1)
    else:
        embeddings1 = corpus1
    if isinstance(corpus2[0], str):
        embeddings2 = model.embed_sentences(corpus2)
    else:
        embeddings2 = corpus2
    return embeddings1, embeddings2

def get_corpora_tokens(corpus1:TCorpus, corpus2:TCorpus, model):
    if isinstance(corpus1[0], str):
        tokens1 = model.tokenize_sentences(corpus1)
    else:
        tokens1 = corpus1
    if isinstance(corpus2[0], str):
        tokens2 = model.tokenize_sentences(corpus2)
    else:
        tokens2 = corpus2
    return tokens1, tokens2


def zipf_coeff(samples, min_num=1, max_num=5000, stretch_factor=15):
	# samples: list of lists of tokens; max_num: how many top frequency words to consider
	counter = Counter()
	for s in samples:
		counter.update(s)
	top_freqs = np.array(sorted(counter.values(), key=operator.neg)[:max_num])
	# log scale overweights tail, so subsample the tail
	# this also helps the best-fit line look more reasonable when plotted in log-scale.
	xs, idxs_u = np.unique(np.round(
		stretch_factor * np.log(np.arange(min_num, min(len(counter), max_num)).astype(np.float64))) / stretch_factor,
						   return_index=True)
	ys = np.log(top_freqs[idxs_u])

	lr = linear_model.LinearRegression()
	lr.fit(xs.reshape(-1, 1), ys)
	slope = lr.coef_[0]

	return slope
