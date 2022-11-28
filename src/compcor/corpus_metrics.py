import random
from collections import Counter
from operator import itemgetter

import numpy as np
from prdc import compute_prdc
import mauve

from scipy.linalg import sqrtm
from scipy.stats import chisquare, ttest_ind
from sklearn import svm
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial
from sklearn.metrics import f1_score
import scipy

from compcor.text_embedder import TextTokenizer, TextEmbedder
import compcor.utils as utils
from compcor.utils import Corpus, TCorpus
from compcor.text_tokenizer_embedder import STTokenizerEmbedder


def ttest_distance(corpus1: Corpus, corpus2: Corpus, model: TextEmbedder = STTokenizerEmbedder()):
	# calculate mean and covariance statistics

	embeddings1, embeddings2 = utils.get_corpora_embeddings(corpus1, corpus2, model)

	res = ttest_ind(embeddings1, embeddings2)
	return 1 - np.nanmean(res.pvalue)


def IRPR_distance(corpus1: Corpus, corpus2: Corpus, model: TextEmbedder = STTokenizerEmbedder()):
	embeddings1, embeddings2 = utils.get_corpora_embeddings(corpus1, corpus2, model)

	cosine = np.clip(cosine_similarity(embeddings1, embeddings2), -1, 1)
	# this is because sometimes cosine_similarity return values larger than 1
	table = np.arccos(cosine) / np.pi
	precision = np.nansum(np.nanmin(table, axis=1)) / table.shape[1]
	recall = np.nansum(np.nanmin(table, axis=0)) / table.shape[0]
	return 2 * (precision * recall) / (precision + recall)


def classifier_distance(corpus1: Corpus, corpus2: Corpus, model: TextEmbedder = STTokenizerEmbedder()):
	# distance between corpora is the F1 score of a classifier trained to classify membership of a random sample of each
	embeddings1, embeddings2 = utils.get_corpora_embeddings(corpus1, corpus2, model)

	corpus1_vecs = embeddings1
	corpus1_train_indx = random.sample(range(len(embeddings1)), k=int(0.8 * len(embeddings1)))
	corpus1_train = itemgetter(*corpus1_train_indx)(corpus1_vecs)

	corpus1_test_indx = set(range(len(embeddings1))) - (set(corpus1_train_indx))
	corpus1_test = itemgetter(*corpus1_test_indx)(corpus1_vecs)

	corpus2_vecs = embeddings2
	corpus2_train_indx = random.sample(range(len(embeddings2)), k=int(0.8 * len(embeddings2)))
	corpus2_train = itemgetter(*corpus2_train_indx)(corpus2_vecs)

	corpus2_test_indx = set(range(len(embeddings2))) - (set(corpus2_train_indx))
	corpus2_test = itemgetter(*corpus2_test_indx)(corpus2_vecs)

	train_x = corpus1_train + corpus2_train
	train_y = [0] * len(corpus1_train) + [1] * len(corpus2_train)
	test_x = corpus1_test + corpus2_test
	test_y = [0] * len(corpus1_test) + [1] * len(corpus2_test)
	model = svm.SVC(random_state=1)
	model.fit(train_x, train_y)

	y_pred = model.predict(test_x)
	correct = f1_score(test_y, y_pred)

	return correct


def medoid_distance(corpus1: Corpus, corpus2: Corpus, model: TextEmbedder = STTokenizerEmbedder()):
	embeddings1, embeddings2 = utils.get_corpora_embeddings(corpus1, corpus2, model)

	# calculate mean and covariance statistics
	act1 = np.vstack(embeddings1)
	act2 = np.vstack(embeddings2)
	mu1 = np.mean(act1, axis=0)
	mu2 = np.mean(act2, axis=0)
	# calculate sum squared difference between means
	cosine = spatial.distance.cosine(mu1, mu2)
	return cosine


def fid_distance(corpus1: Corpus, corpus2: Corpus, model: TextEmbedder = STTokenizerEmbedder()):
	embeddings1, embeddings2 = utils.get_corpora_embeddings(corpus1, corpus2, model)
	# TODO: needs a note explaining what the resulting calculation is.  Is it an overlap/probability as approximated by Gaussian curve
	# Note that the paper says FID is a F1 score but this is a different calculation (unless it is in effect an F1 score)
	if len(embeddings1) == 0 or len(embeddings2) == 0:
		return 0
	act1 = np.vstack(embeddings1)
	act2 = np.vstack(embeddings2)
	mu1 = np.mean(act1, axis=0)
	sigma1 = np.cov(act1, rowvar=False)
	mu2 = np.mean(act2, axis=0)
	sigma2 = np.cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = np.sum((mu1 - mu2) ** 2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if np.iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid


def mauve_distance(corpus1: Corpus, corpus2: Corpus, model: TextEmbedder = STTokenizerEmbedder()):
	embeddings1, embeddings2 = utils.get_corpora_embeddings(corpus1, corpus2, model)

	out = mauve.compute_mauve(p_features=embeddings1, q_features=embeddings2, device_id=0, verbose=False)
	return 1 - out.mauve


def pr_distance(corpus1: Corpus, corpus2: Corpus, model: TextEmbedder = STTokenizerEmbedder(), nearest_k=5):
	embeddings1, embeddings2 = utils.get_corpora_embeddings(corpus1, corpus2, model)

	metric = compute_prdc(real_features=np.vstack(embeddings1),
						  fake_features=np.vstack(embeddings2),
						  nearest_k=nearest_k)
	precision = np.clip(metric['precision'], 0, 1)
	recall = np.clip(metric['recall'] + 1e-6, 0, 1)

	return 1 - 2 * (precision * recall) / (precision + recall)


def dc_distance(corpus1: Corpus, corpus2: Corpus, model: TextEmbedder = STTokenizerEmbedder(), nearest_k=5):
	embeddings1, embeddings2 = utils.get_corpora_embeddings(corpus1, corpus2, model)

	metric = compute_prdc(real_features=np.vstack(embeddings1),
						  fake_features=np.vstack(embeddings2),
						  nearest_k=nearest_k)

	density = np.clip(metric['density'], 0, 1)
	coverage = np.clip(metric['coverage'] + 1e-6, 0, 1)

	return 1 - 2 * (density * coverage) / (density + coverage)


def chi_square_distance(corpus1: TCorpus, corpus2: TCorpus, tokenizer: TextTokenizer = STTokenizerEmbedder(),
						top=5000):
	# calculate p-value of chi-square test between frequency counts of top most frequent shared tokens between corpora
	# note, does not normalize for the size of the corpora, so most common tokens may reflect more the larger corpus
	tokens1, tokens2 = utils.get_corpora_tokens(corpus1, corpus2, tokenizer)

	if type(tokens1[0]) is list:
		tokens1 = [x for xs in tokens1 for x in xs]
		tokens2 = [x for xs in tokens2 for x in xs]

	c1_word_count = Counter(tokens1)
	c2_word_count = Counter(tokens2)
	common_words = set([word for word, freq in Counter(tokens1 + tokens2).most_common(top)])
	sum_count = {word: c1_word_count[word] + c2_word_count[word] for word in common_words}

	N1 = sum([c1_word_count[word] for word in common_words])
	N2 = sum([c2_word_count[word] for word in common_words])
	N = N1 + N2
	o1 = []
	o2 = []
	e1 = []
	e2 = []
	for word in common_words:
		e1 += [sum_count[word] * N1 / N]
		o1 += [c1_word_count[word]]
		e2 += [sum_count[word] * N2 / N]
		o2 += [c2_word_count[word]]

	# low p value means two corpora are different.
	chi_stat = chisquare(f_exp=e1, f_obs=o1)[0] + chisquare(f_exp=e2, f_obs=o2)[0]
	return 1-scipy.stats.chi2.cdf(chi_stat, 2 * (len(common_words) - 1))


def zipf_distance(corpus1: TCorpus, corpus2: TCorpus, tokenizer: TextTokenizer = STTokenizerEmbedder()):
	tokens1, tokens2 = utils.get_corpora_tokens(corpus1, corpus2, tokenizer)
	
	zipf1 = utils.zipf_coeff(tokens1)
	zipf2 = utils.zipf_coeff(tokens2)
	return np.abs(zipf2 - zipf1)
