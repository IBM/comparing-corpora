__author__ = 'gkour'

import time
from typing import List

from TextEmbedder import TextEmbedder
from sentence_transformers import SentenceTransformer
from sklearn import preprocessing


class TransformerTextEmbedder(TextEmbedder):
    def __init__(self, embedding_model_name="all-MiniLM-L6-v2"):
        self.model_name = embedding_model_name
        self.embedder: SentenceTransformer = None
        self.words_vectors = None

    def _initialize_embedder(self):
        if self.embedder is None:
            tick = time.time()
            print("Started Loading embedding model...")
            self.embedder = SentenceTransformer(self.model_name)
            print("Loading model {} took {} sec".format(self.model_name, time.time() - tick))

    def embed_sentences(self, sentences: List[str], normalize=False):
        self._initialize_embedder()

        print("Started Embedding Sentences...")
        tick = time.time()
        vectors = self.embedder.encode(sentences, show_progress_bar=True)
        print("Embedding Sentences took {} sec".format(time.time()-tick))
        vectors = preprocessing.normalize(vectors) if normalize else vectors
        return vectors

    def tokenize_sentences(self, sentences):
        self._initialize_embedder()
        return self.embedder.tokenize(sentences)['input_ids'].numpy().tolist()

    def set_embedding_model(self, model:SentenceTransformer):
        self.embedder = model