from flair.embeddings import CharacterEmbeddings, TokenEmbeddings, StackedEmbeddings, FlairEmbeddings, TransformerWordEmbeddings
from flair.data import Sentence, Token
from gensim.models import KeyedVectors
from typing import List
import re 
import numpy as np
import torch
import os

class W2vWordEmbeddings(TokenEmbeddings):

    def __init__(self, embeddings):
        super().__init__()
        self.name = embeddings
        self.static_embeddings = False
        self.precomputed_word_embeddings = KeyedVectors.load_word2vec_format(embeddings, binary=False)
        self.__embedding_length: int = self.precomputed_word_embeddings.vector_size
        

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:
        for i, sentence in enumerate(sentences):
            for token, token_idx in zip(sentence.tokens, range(len(sentence.tokens))):
                token: Token = token
                if token.text in self.precomputed_word_embeddings:
                    word_embedding = self.precomputed_word_embeddings[token.text]
                elif token.text.lower() in self.precomputed_word_embeddings:
                    word_embedding = self.precomputed_word_embeddings[token.text.lower()]
                elif re.sub('\d', '#', token.text.lower()) in self.precomputed_word_embeddings:
                    word_embedding = self.precomputed_word_embeddings[re.sub('\d', '#', token.text.lower())]
                elif re.sub('\d', '0', token.text.lower()) in self.precomputed_word_embeddings:
                    word_embedding = self.precomputed_word_embeddings[re.sub('\d', '0', token.text.lower())]
                else:
                    word_embedding = np.zeros(self.embedding_length, dtype='float')
                word_embedding = torch.FloatTensor(word_embedding)
                token.set_embedding(self.name, word_embedding)
        return sentences

class Embeddings:
    def __init__(self, config) -> None:
        self.config = config
   
    def create_embeddings(self) -> StackedEmbeddings:

        embedding_types: List[FlairEmbeddings] = []
        
        if self.config['use_word_embeddings']:
            embedding_types.append(W2vWordEmbeddings(self.config['word_embeddings_path']))

        if self.config['use_char_embeddings']:
            embedding_types.append(CharacterEmbeddings())


        if self.config['use_flair_embeddings']:
            embedding_types.append(FlairEmbeddings('es-clinical-forward'))
            embedding_types.append(FlairEmbeddings('es-clinical-backward'))
        
        if self.config['use_beto_embeddings']:
            embedding_types.append(
                TransformerWordEmbeddings(
                    'dccuchile/bert-base-spanish-wwm-cased',
                    layers = self.config['layers'], 
                    layer_mean = self.config['layer_mean'], 
                    subtoken_pooling = self.config['subtoken_pooling']))

        embeddings: StackedEmbeddings = StackedEmbeddings(embeddings = embedding_types)
        return embeddings