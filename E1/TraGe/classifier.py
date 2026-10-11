"""TraGe inference model; parameter names match the retained checkpoints."""
import torch
from torch import nn
from uer.embeddings import Embedding, str2embedding
from uer.encoders import str2encoder
from uer.utils.misc import pooling


class Classifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embedding = Embedding(args)
        for name in args.embedding:
            self.embedding.update(str2embedding[name](args, len(args.tokenizer.vocab)), name)
        self.encoder = str2encoder[args.encoder](args)
        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_2 = nn.Linear(args.hidden_size, args.labels_num)

    def forward(self, src, seg):
        output = self.encoder(self.embedding(src, seg), seg)
        output = pooling(output, seg, self.args.pooling)
        return self.output_layer_2(torch.tanh(self.output_layer_1(output)))
