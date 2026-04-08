#!/usr/bin/env python3
"""Generate hcp_trigram.h — token-level trigram frequency table for HCP semantic channel.

Usage:
    python3 tools/gen_trigrams.py [corpus.txt ...] > src/hcp_trigram.h

Uses GPT-2 BPE tokenizer (same as Whisper). Trigrams provide wider context
than bigrams for catching confident-but-wrong substitutions.
"""

import struct
import sys
from collections import Counter

import tiktoken

SLOTS = 524288  # 512K slots

def fnv1a(data: bytes) -> int:
    h = 0xCBF29CE484222325
    for b in data:
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h

def trigram_key(t1: int, t2: int, t3: int) -> int:
    return fnv1a(struct.pack("<III", t1, t2, t3)) % SLOTS

def main():
    enc = tiktoken.encoding_for_model("gpt2")

    if len(sys.argv) > 1:
        text = ""
        for path in sys.argv[1:]:
            with open(path, "r", errors="ignore") as f:
                text += f.read() + "\n"
    else:
        text = sys.stdin.read()

    tokens = enc.encode(text)
    sys.stderr.write(f"[gen_trigrams] {len(tokens)} tokens from {len(text)} chars\n")

    counts = Counter()
    for i in range(len(tokens) - 2):
        counts[(tokens[i], tokens[i + 1], tokens[i + 2])] += 1

    sys.stderr.write(f"[gen_trigrams] {len(counts)} unique trigrams\n")

    sorted_counts = sorted(counts.values())
    p99_idx = int(len(sorted_counts) * 0.99)
    p99_count = sorted_counts[p99_idx] if p99_idx < len(sorted_counts) else 1
    if p99_count < 1:
        p99_count = 1
    sys.stderr.write(f"[gen_trigrams] p99 count: {p99_count}, max count: {max(sorted_counts)}\n")

    table = [0] * SLOTS
    collisions = 0

    for (t1, t2, t3), count in counts.items():
        slot = trigram_key(t1, t2, t3)
        val = max(1, min(255, int(count * 255 / p99_count)))
        if table[slot] != 0:
            collisions += 1
        table[slot] = max(table[slot], val)

    occupied = sum(1 for v in table if v > 0)
    sys.stderr.write(f"[gen_trigrams] {occupied}/{SLOTS} slots occupied "
                     f"({collisions} collisions, {occupied*100/SLOTS:.1f}% load)\n")

    print("/* hcp_trigram.h — auto-generated token trigram frequencies")
    print(f" * Corpus: {len(tokens)} tokens, {len(counts)} unique trigrams")
    print(f" * Slots: {SLOTS}, occupied: {occupied} ({occupied*100/SLOTS:.1f}%)")
    print(" * Regenerate: python3 tools/gen_trigrams.py corpus.txt > src/hcp_trigram.h")
    print(" */")
    print("#ifndef HCP_TRIGRAM_H")
    print("#define HCP_TRIGRAM_H")
    print()
    print(f"#define HCP_TRIGRAM_SLOTS {SLOTS}")
    print()
    print(f"static const unsigned char hcp_trigram_table[{SLOTS}] = {{")

    for i in range(0, SLOTS, 32):
        row = ",".join(f"{table[j]:3d}" for j in range(i, min(i + 32, SLOTS)))
        comma = "," if i + 32 < SLOTS else ""
        print(f"    {row}{comma}")

    print("};")
    print()
    print("#endif /* HCP_TRIGRAM_H */")


if __name__ == "__main__":
    main()
