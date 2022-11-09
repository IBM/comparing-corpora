
from typing import List

from compcor.text_embedder import TextEmbedder, TextTokenizer
from sentence_transformers import SentenceTransformer
from sklearn import preprocessing


class TextTokenizerEmbedder(TextEmbedder, TextTokenizer):
    def __init__(self, embedding_model_name="all-MiniLM-L6-v2"):
        self.model_name = embedding_model_name
        self.embedder: SentenceTransformer = None
        self.words_vectors = None

    def _initialize_embedder(self):
        if self.embedder is None:
            self.embedder = SentenceTransformer(self.model_name)

    def embed_sentences(self, sentences: List[str], normalize=False):
        self._initialize_embedder()

        vectors = self.embedder.encode(sentences, show_progress_bar=False)
        vectors = preprocessing.normalize(vectors) if normalize else vectors
        return vectors

    def tokenize_sentences(self, sentences):
        self._initialize_embedder()
        return self.embedder.tokenize(sentences)['input_ids'].numpy().tolist()

    def set_embedding_model(self, model:SentenceTransformer):
        self.embedder = model