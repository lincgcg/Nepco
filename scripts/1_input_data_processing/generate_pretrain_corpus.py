#!/usr/bin/env python3
"""Generate a whitespace-tokenized hexadecimal corpus from packet captures."""

import argparse
from pathlib import Path

from scapy.all import Ether, IP, IPv6, TCP, UDP
from scapy.utils import PcapReader


def parse_suffixes(value):
    return tuple(item.strip().lower() for item in value.split(",") if item.strip())


def iter_pcaps(pcap_dir, suffixes):
    for path in sorted(Path(pcap_dir).rglob("*")):
        if path.is_file() and path.suffix.lower() in suffixes:
            yield path


def anonymize_packet(packet):
    packet = packet.copy()
    if Ether in packet:
        packet[Ether].src = "00:00:00:00:00:00"
        packet[Ether].dst = "00:00:00:00:00:00"
    if IP in packet:
        packet[IP].src = "0.0.0.0"
        packet[IP].dst = "0.0.0.0"
        for field in ("chksum", "len"):
            if hasattr(packet[IP], field):
                delattr(packet[IP], field)
    if IPv6 in packet:
        packet[IPv6].src = "::"
        packet[IPv6].dst = "::"
        if hasattr(packet[IPv6], "plen"):
            delattr(packet[IPv6], "plen")
    if TCP in packet:
        packet[TCP].sport = 0
        packet[TCP].dport = 0
        if hasattr(packet[TCP], "chksum"):
            delattr(packet[TCP], "chksum")
    if UDP in packet:
        packet[UDP].sport = 0
        packet[UDP].dport = 0
        for field in ("chksum", "len"):
            if hasattr(packet[UDP], field):
                delattr(packet[UDP], field)
    return packet


def hex_to_tokens(hex_string, token_nibbles):
    usable_len = len(hex_string) - (len(hex_string) % token_nibbles)
    return [hex_string[i:i + token_nibbles] for i in range(0, usable_len, token_nibbles)]


def flow_tokens(pcap_path, args):
    tokens = []
    with PcapReader(str(pcap_path)) as reader:
        for packet_id, packet in enumerate(reader):
            if args.max_packets_per_flow > 0 and packet_id >= args.max_packets_per_flow:
                break
            if not args.keep_addresses:
                packet = anonymize_packet(packet)
            packet_hex = bytes(packet).hex()[:args.packet_hex_chars]
            tokens.extend(hex_to_tokens(packet_hex, args.token_nibbles))
    return tokens


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--pcap_dir", required=True, help="Directory containing pcap or pcapng files.")
    parser.add_argument("--output_corpus", required=True, help="Output plain-text corpus path.")
    parser.add_argument("--suffixes", default=".pcap,.pcapng", help="Comma-separated pcap file suffixes.")
    parser.add_argument("--packet_hex_chars", type=int, default=256, help="Max hexadecimal characters kept per packet.")
    parser.add_argument("--max_packets_per_flow", type=int, default=5, help="Max packets read from each pcap; <=0 means all.")
    parser.add_argument("--token_nibbles", type=int, default=4, help="Hex characters per token. Use 4 for hex_vocab.txt.")
    parser.add_argument("--keep_addresses", action="store_true", help="Keep MAC/IP addresses and transport ports.")
    args = parser.parse_args()

    if args.token_nibbles <= 0:
        raise ValueError("--token_nibbles must be positive.")
    if args.packet_hex_chars <= 0:
        raise ValueError("--packet_hex_chars must be positive.")

    output_path = Path(args.output_corpus)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    suffixes = parse_suffixes(args.suffixes)
    pcaps = list(iter_pcaps(args.pcap_dir, suffixes))
    written, skipped = 0, 0

    with output_path.open("w", encoding="utf-8") as fout:
        for pcap_path in pcaps:
            try:
                tokens = flow_tokens(pcap_path, args)
            except Exception as exc:
                skipped += 1
                print("skip {}: {}".format(pcap_path, exc))
                continue
            if not tokens:
                skipped += 1
                continue
            fout.write(" ".join(tokens) + "\n")
            written += 1

    print("pcap files scanned: {}".format(len(pcaps)))
    print("corpus lines written: {}".format(written))
    print("pcap files skipped: {}".format(skipped))
    print("output corpus: {}".format(output_path))


if __name__ == "__main__":
    main()
