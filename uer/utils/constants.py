import json
from pathlib import Path


def _load_special_tokens_map():
    candidates = [
        Path.cwd() / "models" / "special_tokens_map.json",
        Path(__file__).resolve().parents[2] / "models" / "special_tokens_map.json",
    ]
    for path in candidates:
        if path.exists():
            with path.open(mode="r", encoding="utf-8") as f:
                return json.load(f)
    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError("Cannot find models/special_tokens_map.json. Tried: {}".format(tried))


special_tokens_map = _load_special_tokens_map()

UNK_TOKEN = special_tokens_map["unk_token"]
CLS_TOKEN = special_tokens_map["cls_token"]
SEP_TOKEN = special_tokens_map["sep_token"]
MASK_TOKEN = special_tokens_map["mask_token"]
PAD_TOKEN = special_tokens_map["pad_token"]
try:
    # e.g. <extra_id_0>, <extra_id_1>, ... , should have consecutive IDs.
    SENTINEL_TOKEN = special_tokens_map["sentinel_token"]
except KeyError:
    pass
