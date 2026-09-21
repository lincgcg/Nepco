#!/usr/bin/env python3
"""Smoke-test Nepco CPU latency with batch 1 on eight fixed CPU cores."""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from latency_common import load_checkpoint_strict, run_benchmark


E2_DIR = Path(__file__).resolve().parent
AE_ROOT = E2_DIR.parent
sys.path.insert(0, str(AE_ROOT))

from uer.embeddings import Embedding, str2embedding
from uer.encoders import str2encoder
from uer.utils import str2tokenizer
from uer.utils import constants
from uer.utils.misc import pooling


class Classifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embedding = Embedding(args)
        for name in args.embedding:
            self.embedding.update(str2embedding[name](args, len(args.tokenizer.vocab)), name)
        self.encoder = str2encoder[args.encoder](args)
        self.pooling_type = args.pooling
        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_2 = nn.Linear(args.hidden_size, args.labels_num)

    def infer(self, src, seg):
        output = self.encoder(self.embedding(src, seg), seg)
        output = pooling(output, seg, self.pooling_type)
        return self.output_layer_2(torch.tanh(self.output_layer_1(output)))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_model_path", default=str(AE_ROOT / "E1/artifacts/DataCon/finetuned_model.bin"))
    parser.add_argument("--test_path", default=str(AE_ROOT / "E1/artifacts/DataCon/test_dataset.tsv"))
    parser.add_argument("--vocab_path", default=str(AE_ROOT / "vocab/hex_vocab.txt"))
    parser.add_argument("--config_path", default=str(AE_ROOT / "configs/nepco_config.json"))
    parser.add_argument("--output_dir", default=str(E2_DIR / "results/DataCon"))
    parser.add_argument("--max_samples", type=int, default=8)
    return parser.parse_args()


def model_args(cli):
    defaults = {
        "dropout": 0.1,
        "max_seq_length": 512,
        "remove_embedding_layernorm": False,
        "relative_position_embedding": False,
        "factorized_embedding_parameterization": False,
        "parameter_sharing": False,
        "layernorm_positioning": "post",
        "feed_forward": "dense",
        "relative_attention_buckets_num": 32,
        "remove_attention_scale": False,
        "remove_transformer_bias": False,
        "layernorm": "normal",
        "has_residual_attention": False,
    }
    with open(cli.config_path, encoding="utf-8") as handle:
        defaults.update(json.load(handle))
    defaults.update({
        "vocab_path": cli.vocab_path,
        "tokenizer": "bert",
        "do_lower_case": "true",
        "spm_model_path": None,
        "merges_path": None,
        "embedding": ["word"],
        "encoder": "Nepco",
        "pooling": "max",
        "labels_num": 10,
        "seq_length": 128,
        "mask": "fully_visible",
    })
    args = SimpleNamespace(**defaults)
    args.tokenizer = str2tokenizer["bert"](args)
    args.test_path = cli.test_path
    args.output_model_path = cli.output_model_path
    return args


def main():
    cli = parse_args()
    for path in (cli.output_model_path, cli.test_path, cli.vocab_path, cli.config_path):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    args = model_args(cli)
    model = Classifier(args)
    load_checkpoint_strict(model, cli.output_model_path)
    model.eval()
    if hasattr(model.encoder, "prepare_for_onnx_export"):
        model.encoder.prepare_for_onnx_export()
    elif hasattr(model.encoder, "refresh_infer_cache"):
        model.encoder.refresh_infer_cache()
    elif hasattr(model.encoder, "update_softmax_weights"):
        model.encoder.update_softmax_weights()
    run_benchmark("nepco", model, args, constants, cli.output_dir, cli.max_samples)


if __name__ == "__main__":
    main()
