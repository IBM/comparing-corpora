from typing import List


class TextEmbedder:

	def embed_sentences(self, sentences: List[str], normalize=False) ->List[List[float]]:
		raise NotImplementedError()


class TextTokenizer:
	def tokenize_sentences(self, sentences:List[str]) -> List[List[str]]:
		raise NotImplementedError()

