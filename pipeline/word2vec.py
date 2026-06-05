# -*- coding: utf-8 -*-

import numpy as np
from typing import List, Callable, Optional


class MeanEmbeddingVectorizer:
    """
    Un trasformer compatible con scikit-learn, que usa un embedding para vectorizar documentos.
    https://towardsdatascience.com/nlp-performance-of-different-word-embeddings-on-text-classification-de648c6262b
    Puede conseguir un modelo en español de word2vec en https://github.com/dccuchile/spanish-word-embeddings
    """

    def __init__(self, word_embedding, tokenizer: Callable[[str], List[str]], stopwords:Optional[List[str]] = None):
        """
        Inicializar un Vectorizer compatible con scikit-learn que utiliza un embedding para crear la representacion vectorial de un documento.
        :param word_embedding: KeyedVectors (gensim) o cualquier objeto con vector_size, __contains__ y get_vector.
                               También acepta un modelo gensim.models.Word2Vec completo (usa su atributo .wv).
        :param tokenizer: El tokenizador; una funcion que reciba un texto y retorne una secuencia de tokens de ese texto.
        :param stopwords: Una lista opcional de stopwords, 1 palabra por linea.
        """
        # Soportar tanto gensim.models.Word2Vec (tiene .wv) como KeyedVectors directamente
        self.word_embedding = getattr(word_embedding, 'wv', word_embedding)
        self.vector_size = self.word_embedding.vector_size
        self.tokenizer = tokenizer
        if stopwords:
            self.stopwords = set(stopwords)
        else:
            self.stopwords = set()

    def fit(self):
        """
        implementar fit() para cumplir con los requerimientos de un trasformed de  scikit-learn.
        """
        return self

    def transform(self, docs:List[str]) -> np.ndarray:  # implementar transform() para cumplir con los requerimientos de un trasformed de  scikit-learn
        """
        Transforma una lista de documentos en una matriz, utilizando los vectores del embedding.
        :param docs: Una lista de texto de documentos.
        :return: Una matriz, la fila #i es la representacion vectorial del documento #i.
        """
        if isinstance(docs,list):
            return self.word_average_list(docs)
        else:
            raise ValueError("docs debe ser una lista de documentos")

    def word_average(self, doc: str) -> np.ndarray:
        """
        Computa el promedio (centroide) de los vectors de los tokens en doc.
        :param doc: El texto de un documento
        :return: mean: El promedio (centroide) de los vectores de tokens que están en el embedding
        """
        words = [word.lower() for word in self.tokenizer(doc) if word.lower() not in self.stopwords]
        valid_vectors = [
            self.word_embedding.get_vector(word)
            for word in words
            if word in self.word_embedding
        ]
        if not valid_vectors:
            return np.zeros(self.vector_size)
        return np.mean(valid_vectors, axis=0)


    def word_average_list(self, docs: List[str]) -> np.ndarray:
        """
        Computa el vector promedio de cada doc.
        :param docs: Una lista c/item es el texto de 1 documento.
        :return: Un array de numpy con el vector promedio de cad doc, de shape (len(docs),)
        """
        return np.vstack([self.word_average(doc) for doc in docs])

    def fit_transform(self, docs: List[str], y=None) -> np.ndarray:
        return self.transform(docs)
