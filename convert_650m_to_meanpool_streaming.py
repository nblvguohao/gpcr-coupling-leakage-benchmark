#!/usr/bin/env python3
"""
纯流式转换：esm_features_650m_paired.json -> esm_features_650m_meanpool.json
不使用 numpy，也不将整个子数组载入内存，逐 key 处理。
"""

import ijson
import json
from pathlib import Path

BASE = Path(__file__).parent
INPUT_FILE = BASE / "server_sync" / "extended_data" / "features" / "esm_features_650m_paired.json"
OUTPUT_FILE = BASE / "server_sync" / "extended_data" / "features" / "esm_features_650m_meanpool.json"


def mean_pool(tokens):
    """tokens: list[list[float]]; return list[float] mean vector."""
    if not tokens:
        return []
    n = len(tokens)
    dim = len(tokens[0])
    sums = [0.0] * dim
    for row in tokens:
        for i, v in enumerate(row):
            sums[i] += float(v)
    return [s / n for s in sums]


def main():
    print("[INFO] Streaming parse with ijson (no numpy) ...")
    with open(INPUT_FILE, "rb") as fin, open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        fout.write("{")
        first = True
        count = 0
        for uid, tokens in ijson.kvitems(fin, ""):
            if not first:
                fout.write(",")
            first = False
            vec = mean_pool(tokens)
            fout.write(f'\n  {json.dumps(uid)}: {json.dumps(vec)}')
            count += 1
            if count % 50 == 0:
                print(f"  ... processed {count} sequences")
        fout.write("\n}\n")

    print(f"[OK] Saved mean-pooled features for {count} sequences to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
