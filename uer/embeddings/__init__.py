from uer.embeddings.embedding import Embedding
from uer.embeddings.word_embedding import WordEmbedding


str2embedding = {"word": WordEmbedding}

__all__ = ["Embedding", "WordEmbedding", "str2embedding"]
