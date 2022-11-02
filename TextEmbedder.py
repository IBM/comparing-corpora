from typing import List


class TextEmbedder:

	def embed_sentences(self, sentences: List[str], normalize=False):
		raise NotImplementedError()

	def tokenize_sentences(self, sentences):
		raise NotImplementedError()

