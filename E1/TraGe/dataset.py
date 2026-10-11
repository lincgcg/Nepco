"""Read the full accepted packet TSV using the original TraGe tokenization."""
from uer.utils.constants import CLS_TOKEN, SEP_TOKEN, PAD_TOKEN


def read_dataset(args, path):
    rows = []
    pad = args.tokenizer.convert_tokens_to_ids([PAD_TOKEN])[0]
    with path.open(encoding="utf-8") as f:
        columns = {name: i for i, name in enumerate(next(f).rstrip("\r\n").split("\t"))}
        for line in f:
            fields = line.rstrip("\r\n").split("\t")
            gold = int(fields[columns["label"]])
            if not 0 <= gold < args.labels_num:
                raise ValueError("Test label out of range")
            src = args.tokenizer.convert_tokens_to_ids(
                [CLS_TOKEN] + args.tokenizer.tokenize(fields[columns["text_a"]]) + [SEP_TOKEN])
            seg = [1] * len(src)
            if "text_b" in columns:
                src_b = args.tokenizer.convert_tokens_to_ids(
                    args.tokenizer.tokenize(fields[columns["text_b"]]) + [SEP_TOKEN])
                src += src_b
                seg += [2] * len(src_b)
            src, seg = src[:128], seg[:128]
            missing = 128 - len(src)
            rows.append((src + [pad] * missing, gold, seg + [0] * missing))
    return rows
