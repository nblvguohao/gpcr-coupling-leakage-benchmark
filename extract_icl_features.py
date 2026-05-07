#!/usr/bin/env python3
"""
基于序列和 ESM-2 per-token 特征提取 ICL2/ICL3 局部特征。

在没有 TMHMM/Phobius 结果时，使用改进的 Kyte-Doolittle 滑动窗口预测 TM1-7，
然后切分出 ICL2 (TM3-TM4) 和 ICL3 (TM5-TM6)。

输出:
- paired_dataset/icl_features.json
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

BASE = Path(__file__).parent
SEQUENCES_FILE = BASE / "merged_dataset" / "extended_sequences.json"
ESM_FEATURES_FILE = BASE / "server_sync" / "extended_data" / "features" / "esm_features_100samples.json"
TOPOLOGY_FILE = BASE / "paired_dataset" / "uniprot_topology.json"
OUTPUT_FILE = BASE / "paired_dataset" / "icl_features.json"

# Kyte-Doolittle 疏水性尺度
KD_SCALE = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
    "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}

# 电荷
POSITIVE = {"K", "R", "H"}
NEGATIVE = {"D", "E"}
HYDROPHOBIC = {"A", "V", "I", "L", "M", "F", "W", "G", "P", "C"}
AROMATIC = {"F", "W", "Y"}


def predict_tm_helices(seq: str, window: int = 19, threshold: float = 1.4, min_len: int = 18, max_len: int = 35) -> List[Tuple[int, int]]:
    """
    基于 Kyte-Doolittle 滑动窗口预测跨膜螺旋位置。
    参数经过调整以更好地匹配 GPCR 的 7TM 结构。
    """
    scores = []
    for i in range(len(seq) - window + 1):
        seg = seq[i:i + window]
        score = sum(KD_SCALE.get(aa, 0) for aa in seg) / window
        scores.append(score)

    # 找出连续高于 threshold 的区域
    regions = []
    in_tm = False
    start = 0
    for i, sc in enumerate(scores):
        if sc >= threshold and not in_tm:
            in_tm = True
            start = i
        elif sc < threshold - 0.4 and in_tm:
            in_tm = False
            end = i + window - 1
            seg_len = end - start + 1
            if min_len <= seg_len <= max_len:
                regions.append((start + 1, end))

    # 合并过近的片段 (<10 aa gap)
    if not regions:
        return []
    merged = [regions[0]]
    for s, e in regions[1:]:
        last_s, last_e = merged[-1]
        if s - last_e < 10:
            merged[-1] = (last_s, e)
        else:
            merged.append((s, e))

    # 如果超过7个，按评分保留最好的7个
    if len(merged) > 7:
        seg_scores = []
        for s, e in merged:
            seg = seq[s - 1:e]
            sc = sum(KD_SCALE.get(aa, 0) for aa in seg) / len(seg)
            seg_scores.append((sc, s, e))
        seg_scores.sort(reverse=True)
        merged = [(s, e) for _, s, e in seg_scores[:7]]
        merged.sort(key=lambda x: x[0])

    return merged


def derive_loops(tm_regions: List[Tuple[int, int]], seq_len: int) -> Dict[str, Tuple[int, int]]:
    """从 TM 区域推导 loops。"""
    loops = {}
    if not tm_regions:
        return loops

    loop_names = ["ICL1", "ECL1", "ICL2", "ECL2", "ICL3", "ECL3"]
    for i in range(len(tm_regions) - 1):
        gap_s = tm_regions[i][1] + 1
        gap_e = tm_regions[i + 1][0] - 1
        if gap_e >= gap_s:
            name = loop_names[i] if i < len(loop_names) else f"loop_{i}"
            loops[name] = (gap_s, gap_e)

    # N-tail / C-tail
    if tm_regions[0][0] > 1:
        loops["N-tail"] = (1, tm_regions[0][0] - 1)
    loops["C-tail"] = (tm_regions[-1][1] + 1, seq_len)
    return loops


def sequence_stats(seq: str) -> Dict:
    """计算序列段的理化统计量。"""
    n = len(seq)
    if n == 0:
        return {}
    hydro_scores = [KD_SCALE.get(aa, 0) for aa in seq]
    pos = sum(1 for aa in seq if aa in POSITIVE)
    neg = sum(1 for aa in seq if aa in NEGATIVE)
    return {
        "length": n,
        "mean_hydro": float(np.mean(hydro_scores)),
        "std_hydro": float(np.std(hydro_scores)),
        "net_charge": (pos - neg) / n,
        "pos_charge_ratio": pos / n,
        "neg_charge_ratio": neg / n,
        "hydrophobic_ratio": sum(1 for aa in seq if aa in HYDROPHOBIC) / n,
        "aromatic_ratio": sum(1 for aa in seq if aa in AROMATIC) / n,
    }


def extract_local_esm(esm_tokens: np.ndarray, seq: str, start: int, end: int) -> np.ndarray:
    """
    提取序列某段的 ESM-2 per-token 特征的局部 mean pooling。
    esm_tokens 形状: (seq_len, 320)
    start/end: 1-based, inclusive
    """
    if esm_tokens.ndim != 2:
        raise ValueError(f"Expected 2-d ESM tokens, got {esm_tokens.ndim}")
    seq_len = len(seq)
    token_len = esm_tokens.shape[0]
    # ESM tokens 长度可能与序列长度有微小差异（特殊AA如U等）
    valid_len = min(seq_len, token_len)
    if end > valid_len:
        end = valid_len
    if start < 1:
        start = 1
    if end >= start:
        local = esm_tokens[start - 1:end, :]
        return local.mean(axis=0)
    return np.zeros(esm_tokens.shape[1])


def main():
    print("=" * 70)
    print("  ICL2/ICL3 局部特征提取")
    print("=" * 70)

    with open(SEQUENCES_FILE, encoding="utf-8") as f:
        sequences = json.load(f)
    with open(ESM_FEATURES_FILE, encoding="utf-8") as f:
        esm_features = json.load(f)

    topo_cache = {}
    if TOPOLOGY_FILE.exists():
        with open(TOPOLOGY_FILE, encoding="utf-8") as f:
            topo_cache = json.load(f)
        print(f"[INFO] 加载 UniProt 拓扑: {len(topo_cache)} 条")

    def get_topology(uid: str, seq: str):
        rec = topo_cache.get(uid)
        if rec is None and "_" in uid:
            base = uid.split("_", 1)[1]
            rec = topo_cache.get(base)
        if rec is None:
            for key in topo_cache:
                if "_" in key and key.split("_", 1)[1] == uid:
                    rec = topo_cache[key]
                    break
        if rec and rec.get("tm_regions"):
            return rec["tm_regions"], rec.get("loops", {})
        return predict_tm_helices(seq), {}

    results = {}
    stats = defaultdict(list)
    heuristic_fallback = 0

    for uid, record in sequences.items():
        seq = record["sequence"] if isinstance(record, dict) else record
        if isinstance(seq, dict):
            seq = seq.get("sequence", "")
        if not seq:
            continue

        esm_arr = np.array(esm_features.get(uid, []))
        if esm_arr.size == 0:
            continue

        tm_regions, loops = get_topology(uid, seq)
        if not loops:
            loops = derive_loops(tm_regions, len(seq))
            heuristic_fallback += 1

        feat = {
            "tm_regions": tm_regions,
            "loops": {k: list(v) for k, v in loops.items()},
        }

        # 提取 ICL2 / ICL3 / N-tail / C-tail 的局部特征
        for region_name in ["ICL2", "ICL3", "N-tail", "C-tail"]:
            if region_name not in loops:
                continue
            s, e = loops[region_name]
            subseq = seq[s - 1:e]
            local_stats = sequence_stats(subseq)
            local_esm = extract_local_esm(esm_arr, seq, s, e)

            feat[f"{region_name}_stats"] = local_stats
            feat[f"{region_name}_esm"] = local_esm.tolist()

            for k, v in local_stats.items():
                stats[f"{region_name}_{k}"].append(v)

        results[uid] = feat

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"[OK] 已提取 {len(results)} 条序列的 ICL 特征，保存至 {OUTPUT_FILE}")
    print(f"[INFO] 启发式回退 (Kyte-Doolittle): {heuristic_fallback} 条")

    # 打印统计摘要
    print("\n--- ICL2 / ICL3 统计摘要 ---")
    for key in sorted(stats.keys()):
        vals = stats[key]
        if vals:
            print(f"  {key:30s}: mean={np.mean(vals):.3f}, std={np.std(vals):.3f}, n={len(vals)}")


if __name__ == "__main__":
    main()
