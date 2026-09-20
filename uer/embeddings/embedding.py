import torch.nn as nn

from uer.layers.layer_norm import LayerNorm


class Embedding(nn.Module):
    def __init__(self, args):
        super(Embedding, self).__init__()
        self.embedding_name_list = []
        self.dropout = nn.Dropout(args.dropout)
        self.remove_embedding_layernorm = args.remove_embedding_layernorm
        if not self.remove_embedding_layernorm:
            self.layer_norm = LayerNorm(args.emb_size)

    def update(self, embedding, embedding_name):
        setattr(self, embedding_name, embedding)
        self.embedding_name_list.append(embedding_name)

    def forward(self, src, seg):
        emb = None
        for embedding_name in self.embedding_name_list:
            embedding = getattr(self, embedding_name)
            if emb is None:
                emb = embedding(src, seg)
            else:
                emb = emb + embedding(src, seg)

        if emb is None:
            raise ValueError("No embedding module has been registered.")
        if not self.remove_embedding_layernorm:
            emb = self.layer_norm(emb)
        return self.dropout(emb)
