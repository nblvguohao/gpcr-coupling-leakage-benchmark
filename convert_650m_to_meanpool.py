#!/usr/bin/env python3
"""
将 esm_features_650m_paired.json (5.7GB, per-token) 转换为 mean-pooled JSON。
输出: server_sync/extended_data/features/esm_features_650m_meanpool.json
"""

import orjson
import numpy as np
from pathlib import Path

BASE = Path(__file__).parent
INPUT_FILE = BASE / "server_sync" / "extended_data" / "features" / "esm_features_650m_paired.json"
OUTPUT_FILE = BASE / "server_sync" / "extended_data" / "features" / "esm_features_650m_meanpool.json"


def main():
    print("[INFO] Loading large JSON with orjson ...")
    with open(INPUT_FILE, "rb") as f:
        data = orjson.loads(f.read())

    print(f"[INFO] Total sequences: {len(data)}")
    meanpool = {}
    for uid, arr in data.items():
        vec = np.array(arr).mean(axis=0)
        meanpool[uid] = vec.tolist()

    with open(OUTPUT_FILE, "wb") as f:
        f.write(orjson.dumps(meanpool, option=orjson.OPT_SERIALIZE_NUMPY))
    print(f"[OK] Saved mean-pooled features to {OUTPUT_FILE}")

    # Verify
    import json
    with open(OUTPUT_FILE) as f:
        check = json.load(f)
    first = np.array(list(check.values())[0])
    print(f"[OK] Verified shape: {first.shape}")


if __name__ == "__main__":
    main()
