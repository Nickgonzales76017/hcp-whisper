#!/usr/bin/env python3
"""Generate hcp_bigram.h — token-level bigram frequency table for HCP semantic channel.

Usage:
    python3 tools/gen_bigrams.py [corpus.txt ...] > src/hcp_bigram.h

If no files given, reads from stdin. Uses GPT-2 BPE tokenizer (same as Whisper).
Re-run on more/better text to improve the table.
"""

import struct
import sys
from collections import Counter

import tiktoken

SLOTS = 262144  # 256K slots — 256KB table

def fnv1a(data: bytes) -> int:
    h = 0xCBF29CE484222325
    for b in data:
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h

def bigram_key(t1: int, t2: int) -> int:
    return fnv1a(struct.pack("<II", t1, t2)) % SLOTS

def main():
    enc = tiktoken.encoding_for_model("gpt2")

    # Read corpus text
    if len(sys.argv) > 1:
        text = ""
        for path in sys.argv[1:]:
            with open(path, "r", errors="ignore") as f:
                text += f.read() + "\n"
    else:
        text = sys.stdin.read()

    # Tokenize
    tokens = enc.encode(text)
    sys.stderr.write(f"[gen_bigrams] {len(tokens)} tokens from {len(text)} chars\n")

    # Count bigrams
    counts = Counter()
    for i in range(len(tokens) - 1):
        counts[(tokens[i], tokens[i + 1])] += 1

    sys.stderr.write(f"[gen_bigrams] {len(counts)} unique bigrams\n")

    # Build table (percentile normalization to avoid outlier squashing)
    sorted_counts = sorted(counts.values())
    p99_idx = int(len(sorted_counts) * 0.99)
    p99_count = sorted_counts[p99_idx] if p99_idx < len(sorted_counts) else 1
    if p99_count < 1:
        p99_count = 1
    sys.stderr.write(f"[gen_bigrams] p99 count: {p99_count}, max count: {max(sorted_counts)}\n")

    table = [0] * SLOTS
    collisions = 0

    for (t1, t2), count in counts.items():
        slot = bigram_key(t1, t2)
        val = max(1, min(255, int(count * 255 / p99_count)))
        if table[slot] != 0:
            collisions += 1
        table[slot] = max(table[slot], val)

    occupied = sum(1 for v in table if v > 0)
    sys.stderr.write(f"[gen_bigrams] {occupied}/{SLOTS} slots occupied "
                     f"({collisions} collisions, {occupied*100/SLOTS:.1f}% load)\n")

    # Emit C header
    print("/* hcp_bigram.h — auto-generated token bigram frequencies")
    print(f" * Corpus: {len(tokens)} tokens, {len(counts)} unique bigrams")
    print(f" * Slots: {SLOTS}, occupied: {occupied} ({occupied*100/SLOTS:.1f}%)")
    print(" * Regenerate: python3 tools/gen_bigrams.py corpus.txt > src/hcp_bigram.h")
    print(" */")
    print("#ifndef HCP_BIGRAM_H")
    print("#define HCP_BIGRAM_H")
    print()
    print(f"#define HCP_BIGRAM_SLOTS {SLOTS}")
    print()
    print(f"static const unsigned char hcp_bigram_table[{SLOTS}] = {{")

    for i in range(0, SLOTS, 32):
        row = ",".join(f"{table[j]:3d}" for j in range(i, min(i + 32, SLOTS)))
        comma = "," if i + 32 < SLOTS else ""
        print(f"    {row}{comma}")

    print("};")
    print()
    print("#endif /* HCP_BIGRAM_H */")


if __name__ == "__main__":
    main()
