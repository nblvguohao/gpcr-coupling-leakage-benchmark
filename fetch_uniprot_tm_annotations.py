#!/usr/bin/env python3
"""
从 UniProt API 批量获取 GPCR 的 Transmembrane 注释，生成精确的 TM1-7 拓扑。

输出:
- paired_dataset/uniprot_topology.json  (与 tmhmm_topology.json 格式兼容)
"""

import json
import time
import requests
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

BASE = Path(__file__).parent
SEQUENCES_FILE = BASE / "merged_dataset" / "extended_sequences.json"
OUTPUT_FILE = BASE / "paired_dataset" / "uniprot_topology.json"


def fetch_tm_regions(uniprot_id: str) -> List[Tuple[int, int]]:
    """通过 UniProt API 获取单个蛋白的 Transmembrane 区域。"""
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    params = {"fields": "ft_transmem"}
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        regions = []
        for feat in data.get("features", []):
            if feat.get("type") == "Transmembrane":
                loc = feat.get("location", {})
                s = loc.get("start", {}).get("value")
                e = loc.get("end", {}).get("value")
                if s is not None and e is not None:
                    regions.append((int(s), int(e)))
        return regions
    except Exception as e:
        print(f"  [WARN] {uniprot_id} 获取失败: {e}")
        return []


def derive_loops(tm_regions: List[Tuple[int, int]], seq_len: int) -> Dict[str, Tuple[int, int]]:
    loops = {}
    if not tm_regions:
        return loops

    # N-tail
    if tm_regions[0][0] > 1:
        loops["N-tail"] = (1, tm_regions[0][0] - 1)

    loop_names = ["ICL1", "ECL1", "ICL2", "ECL2", "ICL3", "ECL3"]
    for i in range(len(tm_regions) - 1):
        s = tm_regions[i][1] + 1
        e = tm_regions[i + 1][0] - 1
        if e >= s:
            name = loop_names[i] if i < len(loop_names) else f"loop_{i}"
            loops[name] = (s, e)

    # C-tail
    loops["C-tail"] = (tm_regions[-1][1] + 1, seq_len)
    return loops


def main():
    print("=" * 70)
    print("  UniProt Transmembrane 注释批量获取")
    print("=" * 70)

    with open(SEQUENCES_FILE, encoding="utf-8") as f:
        sequences = json.load(f)

    results = {}
    failed = []
    success = 0
    total = len(sequences)

    for idx, (uid, rec) in enumerate(sequences.items(), 1):
        seq = rec["sequence"] if isinstance(rec, dict) else rec
        seq_len = len(seq) if isinstance(seq, str) else len(str(seq))

        # 如果 ID 是 prefix 形式，尝试 base ID
        base_id = uid.split("_", 1)[1] if "_" in uid and len(uid.split("_")[0]) <= 2 else uid

        tm_regions = fetch_tm_regions(base_id)
        if not tm_regions and base_id != uid:
            tm_regions = fetch_tm_regions(uid)

        if tm_regions:
            loops = derive_loops(tm_regions, seq_len)
            results[uid] = {
                "tm_regions": tm_regions,
                "loops": loops,
            }
            success += 1
        else:
            failed.append(uid)

        if idx % 50 == 0:
            print(f"  进度: {idx}/{total} (成功 {success}, 失败 {len(failed)})")
        time.sleep(0.1)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] 完成: {success}/{total}")
    print(f"    失败: {len(failed)}")
    if failed:
        print(f"    失败列表 (前10): {failed[:10]}")
    print(f"    输出: {OUTPUT_FILE}")

    # 统计 TM 数量分布
    tm_count_dist = defaultdict(int)
    for info in results.values():
        tm_count_dist[len(info["tm_regions"])] += 1
    print("\n  TM 数量分布:")
    for n, c in sorted(tm_count_dist.items()):
        print(f"    {n} TM: {c} ({c / len(results) * 100:.1f}%)")


if __name__ == "__main__":
    main()
