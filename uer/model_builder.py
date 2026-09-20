from uer.embeddings import Embedding, str2embedding
from uer.encoders import str2encoder
from uer.models.model import Model
from uer.targets import Target, str2target


def build_model(args):
    embedding = Embedding(args)
    for embedding_name in args.embedding:
        tmp_emb = str2embedding[embedding_name](args, len(args.tokenizer.vocab))
        embedding.update(tmp_emb, embedding_name)

    encoder = str2encoder[args.encoder](args)

    target = Target()
    for target_name in args.target:
        tmp_target = str2target[target_name](args, len(args.tokenizer.vocab))
        target.update(tmp_target, target_name)

    return Model(args, embedding, encoder, target)
