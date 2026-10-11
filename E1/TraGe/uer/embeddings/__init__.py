from uer.embeddings.embedding import Embedding
from uer.embeddings.word_embedding import WordEmbedding
from uer.embeddings.pos_embedding import PosEmbedding
from uer.embeddings.seg_embedding import SegEmbedding

str2embedding = {"word": WordEmbedding, "pos": PosEmbedding, "seg": SegEmbedding}
