import torch.nn as nn


class Model(nn.Module):
    def __init__(self, args, embedding, encoder, target):
        super(Model, self).__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.target = target

        if "mlm" in args.target and args.tie_weights:
            self.target.mlm.linear_2.weight = self.embedding.word.embedding.weight

    def forward(self, src, tgt, seg):
        emb = self.embedding(src, seg)
        memory_bank = self.encoder(emb, seg)
        return self.target(memory_bank, tgt, seg)
