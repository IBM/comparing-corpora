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

# threshold below which to match distances to 0
ZERO_THRESH = 0.005

def cosine_arccos_transform(c1, c2=None):
	# c1 and c2 are lists of input arrays

	def process(input):
		if input is not None:
			if isinstance(input, list) or isinstance(input, tuple):
				return np.vstack(input)
			else:
				if isinstance(input, np.ndarray):
					if len(input.shape) == 1:
						# make it have one row
						return input.reshape(1,-1)
					else:
						return input
		else:
			return input

	c1, c2 = process(c1), process(c2)

	cosine = np.arccos(np.clip(cosine_similarity(X=c1, Y=c2), -1,1)) / np.pi # if None will be X with itself
	# due to numeric precision, sometimes cosine distance between identical vectors is not 0 exactly

	cosine[ cosine <= ZERO_THRESH] = 0.0

	return cosine



def ttest_distance(corpus1: Corpus, corpus2: Corpus, model: TextEmbedder = STTokenizerEmbedder()):
	# calculate mean and covariance statistics
	if model is not None:
		# if you just provide the matrices themselves
		embeddings1, embeddings2 = utils.get_corpora_embeddings(corpus1, corpus2, model)
	else:
		embeddings1, embeddings2 = corpus1, corpus2

	res = ttest_ind(embeddings1, embeddings2)
	return 1 - np.nanmean(res.pvalue)


def IRPR_distance(corpus1: Corpus, corpus2: Corpus, model: TextEmbedder = STTokenizerEmbedder()):
	if model is not None:
		embeddings1, embeddings2 = utils.get_corpora_embeddings(corpus1, corpus2, model)
	else:
		embeddings1, embeddings2 = corpus1, corpus2

	table = cosine_arccos_transform(c1=embeddings1, c2=embeddings2)
	precision = np.nansum(np.nanmin(table, axis=1)) / table.shape[1]
	recall = np.nansum(np.nanmin(table, axis=0)) / table.shape[0]
	return 2 * (precision * recall) / (precision + recall)


def classifier_distance(corpus1: Corpus, corpus2: Corpus, model: TextEmbedder = STTokenizerEmbedder()):
	# distance between corpora is the F1 score of a classifier trained to classify membership of a random sample of each
	if model is not None:
		embeddings1, embeddings2 = utils.get_corpora_embeddings(corpus1, corpus2, model)
	else:
		embeddings1, embeddings2 = corpus1, corpus2

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
	clf = svm.SVC(random_state=1)
	clf.fit(train_x, train_y)

	y_pred = clf.predict(test_x)
	correct = f1_score(test_y, y_pred)

	return correct


def medoid_distance(corpus1: Corpus, corpus2: Corpus, model: TextEmbedder = STTokenizerEmbedder()):
	if model is not None:
		embeddings1, embeddings2 = utils.get_corpora_embeddings(corpus1, corpus2, model)
	else:
		embeddings1, embeddings2 = corpus1, corpus2

	# calculate mean and covariance statistics
	act1 = np.vstack(embeddings1)
	act2 = np.vstack(embeddings2)
	mu1 = np.mean(act1, axis=0)
	mu2 = np.mean(act2, axis=0)
	# calculate sum squared difference between means
	cosine = spatial.distance.cosine(mu1, mu2)
	return cosine


def fid_distance(corpus1: Corpus, corpus2: Corpus, model: TextEmbedder = STTokenizerEmbedder()):
	if model is not None:
		embeddings1, embeddings2 = utils.get_corpora_embeddings(corpus1, corpus2, model)
	else:
		embeddings1, embeddings2 = corpus1, corpus2
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
	if model is not None:
		embeddings1, embeddings2 = utils.get_corpora_embeddings(corpus1, corpus2, model)
	else:
		embeddings1, embeddings2 = corpus1, corpus2

	out = mauve.compute_mauve(p_features=embeddings1, q_features=embeddings2, device_id=0, verbose=False)
	return 1 - out.mauve


def pr_distance(corpus1: Corpus, corpus2: Corpus, model: TextEmbedder = STTokenizerEmbedder(), nearest_k=5):
	if model is not None:
		embeddings1, embeddings2 = utils.get_corpora_embeddings(corpus1, corpus2, model)
	else:
		embeddings1, embeddings2 = corpus1, corpus2

	metric = compute_prdc(real_features=np.vstack(embeddings1),
						  fake_features=np.vstack(embeddings2),
						  nearest_k=nearest_k)
	precision = np.clip(metric['precision'], 0, 1)
	recall = np.clip(metric['recall'] + 1e-6, 0, 1)

	return 1 - 2 * (precision * recall) / (precision + recall)


def dc_distance(corpus1: Corpus, corpus2: Corpus, model: TextEmbedder = STTokenizerEmbedder(), nearest_k=5):
	if model is not None:
		embeddings1, embeddings2 = utils.get_corpora_embeddings(corpus1, corpus2, model)
	else:
		embeddings1, embeddings2 = corpus1, corpus2

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


def Directed_Hausdorff_distance(corpus1: Corpus, corpus2: Corpus, model: TextEmbedder = STTokenizerEmbedder()):
	# calculate nearest distance from each element in one corpus to an element in the other
	# like IRPR except take mean not harmonic mean (F1-score)
	if model is not None:
		embeddings1, embeddings2 = utils.get_corpora_embeddings(corpus1, corpus2, model)
	else:
		embeddings1, embeddings2 = corpus1, corpus2

	table = cosine_arccos_transform(c1=embeddings1, c2=embeddings2)
	nearest_1to2 = np.nanmin(table, axis=1) # nearest in c2 from each in c1, min in each row
	nearest_2to1 = np.nanmin(table, axis=0)  # nearest in c1 from each in c2, min in each column

	return np.mean([nearest_1to2.mean(), nearest_2to1.mean()])


def Energy_distance(corpus1: Corpus, corpus2: Corpus, model: TextEmbedder = STTokenizerEmbedder(), normalize=False):
	# https://en.wikipedia.org/wiki/Energy_distance
	if model is not None:
		embeddings1, embeddings2 = utils.get_corpora_embeddings(corpus1, corpus2, model)
	else:
		embeddings1, embeddings2 = corpus1, corpus2

	between = cosine_arccos_transform(c1=embeddings1, c2=embeddings2)
	within1 = cosine_arccos_transform(c1=embeddings1)
	within2 = cosine_arccos_transform(c1=embeddings2)
	A2 = 2 * between.mean()
	B = within1.mean()
	C = within2.mean()

	edist = A2 - B - C
	#  E-coefficient of inhomogeneity is between 0 and 1
	return edist/A2 if normalize else np.sqrt(edist)