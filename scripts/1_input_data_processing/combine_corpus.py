#!/usr/bin/env python3
"""Combine one or more corpus text files into a single corpus file."""

import argparse
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_files", nargs="+", required=True, help="Input corpus files.")
    parser.add_argument("--output_file", required=True, help="Combined corpus file.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle non-empty lines before writing.")
    parser.add_argument("--seed", type=int, default=7, help="Shuffle seed.")
    args = parser.parse_args()

    lines = []
    for input_file in args.input_files:
        with Path(input_file).open("r", encoding="utf-8") as fin:
            lines.extend(line.rstrip("\n") for line in fin if line.strip())

    if args.shuffle:
        random.Random(args.seed).shuffle(lines)

    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as fout:
        for line in lines:
            fout.write(line + "\n")

    print("combined lines: {}".format(len(lines)))
    print("output corpus: {}".format(output_file))


if __name__ == "__main__":
    main()
