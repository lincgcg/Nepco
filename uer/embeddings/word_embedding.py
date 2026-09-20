import torch.nn as nn


class WordEmbedding(nn.Module):
    def __init__(self, args, vocab_size):
        super(WordEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, args.emb_size)

    def forward(self, src, _):
        return self.embedding(src)
