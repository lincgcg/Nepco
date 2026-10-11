"""Historical TraGe special tokens, loaded relative to the standalone runtime."""
import json
from pathlib import Path

with (Path(__file__).resolve().parents[2]/"special_tokens_map.json").open(encoding="utf-8") as f:
    special_tokens_map = json.load(f)
UNK_TOKEN = special_tokens_map["unk_token"]
CLS_TOKEN = special_tokens_map["cls_token"]
SEP_TOKEN = special_tokens_map["sep_token"]
MASK_TOKEN = special_tokens_map["mask_token"]
PAD_TOKEN = special_tokens_map["pad_token"]
if "sentinel_token" in special_tokens_map:
    SENTINEL_TOKEN = special_tokens_map["sentinel_token"]
