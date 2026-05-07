#!/usr/bin/env python3
"""
将全局 SHAP 维度重要性映射回残基位置。

逻辑:
1. 读取 shap_results_paired/shap_values_class1.npy (n_samples x 640)
2. 对每个 top GPCR SHAP 维度，找出各样本中该维度取值最大的前 K 个残基位置
3. 结合 TM 拓扑预测结果，统计这些残基落在 TM / ICL / ECL / tail 的比例
4. 输出 JSON 供后续插图使用

输出:
- shap_results_paired/residue_shap_mapping.json
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

BASE = Path(__file__).parent
SHAP_DIR = BASE / "shap_results_paired"
SHAP_VALUES_FILE = SHAP_DIR / "shap_values_class1.npy"
MEAN_ABS_SHAP_FILE = SHAP_DIR / "mean_abs_shap.npy"
ESM_FEATURES_FILE = BASE / "server_sync" / "extended_data" / "features" / "esm_features_100samples.json"
SEQUENCES_FILE = BASE / "merged_dataset" / "extended_sequences.json"
PAIRING_FILE = BASE / "paired_dataset" / "pairing_matrix_raw.csv"
TOPOLOGY_FILE = BASE / "paired_dataset" / "tmhmm_topology.json"
OUTPUT_FILE = SHAP_DIR / "residue_shap_mapping.json"

# 若未找到拓扑文件，使用简易 TM 预测 fallback
KD_SCALE = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
    "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}


def simple_predict_tm(seq: str) -> List[Tuple[int, int]]:
    window = 19
    scores = []
    for i in range(len(seq) - window + 1):
        seg = seq[i:i + window]
        score = sum(KD_SCALE.get(aa, 0) for aa in seg) / window
        scores.append(score)
    regions = []
    in_tm = False
    start = 0
    for i, sc in enumerate(scores):
        if sc >= 1.4 and not in_tm:
            in_tm = True
            start = i
        elif sc < 1.0 and in_tm:
            in_tm = False
            end = i + window - 1
            if 18 <= (end - start + 1) <= 35:
                regions.append((start + 1, end))
    if not regions:
        return []
    merged = [regions[0]]
    for s, e in regions[1:]:
        ls, le = merged[-1]
        if s - le < 10:
            merged[-1] = (ls, e)
        else:
            merged.append((s, e))
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


def get_topology(uid: str, seq: str, topo_cache: Dict) -> Dict:
    if uid in topo_cache:
        return topo_cache[uid]
    # fallback: simple prediction
    tm = simple_predict_tm(seq)
    loops = {}
    if tm:
        loop_names = ["ICL1", "ECL1", "ICL2", "ECL2", "ICL3", "ECL3"]
        for i in range(len(tm) - 1):
            s = tm[i][1] + 1
            e = tm[i + 1][0] - 1
            if e >= s:
                loops[loop_names[i] if i < len(loop_names) else f"loop_{i}"] = (s, e)
        if tm[0][0] > 1:
            loops["N-tail"] = (1, tm[0][0] - 1)
        loops["C-tail"] = (tm[-1][1] + 1, len(seq))
    return {"tm_regions": tm, "loops": loops}


def classify_residue(pos: int, topo: Dict) -> str:
    """将残基位置归类为 TM / ICL / ECL / N-tail / C-tail。"""
    tm_list = topo.get("tm_regions", [])
    for idx, (s, e) in enumerate(tm_list):
        if s <= pos <= e:
            return f"TM{idx + 1}"
    loops = topo.get("loops", {})
    for name, (s, e) in loops.items():
        if s <= pos <= e:
            if name.startswith("ICL"):
                return name
            if name.startswith("ECL"):
                return name
            if name in ("N-tail", "C-tail"):
                return name
    return "other"


def main():
    print("=" * 70)
    print("  SHAP 维度 → 残基位置 映射")
    print("=" * 70)

    sv = np.load(SHAP_VALUES_FILE)  # (n_shap_samples, 640)
    mean_abs = np.load(MEAN_ABS_SHAP_FILE)  # (640,)
    gpcr_importance = mean_abs[:320]
    top_dims = np.argsort(gpcr_importance)[::-1][:20]

    with open(ESM_FEATURES_FILE, encoding="utf-8") as f:
        esm_features = json.load(f)
    with open(SEQUENCES_FILE, encoding="utf-8") as f:
        sequences = json.load(f)

    topo_cache = {}
    if TOPOLOGY_FILE.exists():
        with open(TOPOLOGY_FILE, encoding="utf-8") as f:
            topo_cache = json.load(f)

    # 获取 SHAP 子集中对应的 GPCR ID
    import pandas as pd
    df = pd.read_csv(PAIRING_FILE)
    gpcr_feats_raw, _ = None, None
    # 复用 paired_cross_validation.py 的 load_features 逻辑太麻烦，直接手动构造
    from paired_cross_validation import load_features, build_paired_vectors
    gpcr_feats, gprot_feats = load_features()
    _, y, meta = build_paired_vectors(df, gpcr_feats, gprot_feats)

    n_shap = min(50, len(y))
    rng = np.random.RandomState(42)
    pos_idx = rng.permutation(np.where(y == 1)[0])[:n_shap // 2]
    neg_idx = rng.permutation(np.where(y == 0)[0])[:n_shap - len(pos_idx)]
    shap_idx = np.concatenate([pos_idx, neg_idx])

    print(f"[INFO] SHAP 子集样本数: {len(shap_idx)}")
    print(f"[INFO] 分析 Top {len(top_dims)} GPCR 维度")

    mapping_results = {}

    for dim_idx in top_dims:
        dim_shap_values = sv[:, dim_idx]  # 该维度在各样本上的 SHAP 值
        residue_counter = defaultdict(list)  # region -> list of avg |SHAP|
        all_positions = []

        for i, sample_idx in enumerate(shap_idx):
            m = meta[sample_idx]
            gid = m["gpcr_id"]
            seq_record = sequences.get(gid)
            if seq_record is None:
                # 尝试带 prefix 的匹配
                for k in sequences:
                    if "_" in k and k.split("_", 1)[1] == gid:
                        seq_record = sequences[k]
                        gid = k
                        break
            if seq_record is None:
                continue

            seq = seq_record["sequence"] if isinstance(seq_record, dict) else seq_record
            esm_arr = np.array(esm_features.get(gid, []))
            if esm_arr.size == 0 or len(seq) == 0:
                continue

            topo = get_topology(gid, seq, topo_cache)
            token_len = min(esm_arr.shape[0], len(seq))

            # 取该维度上 token 值最大的前 5 个残基
            dim_tokens = esm_arr[:token_len, dim_idx]
            top_k = min(5, token_len)
            top_pos = np.argsort(np.abs(dim_tokens))[-top_k:][::-1]  # 0-based

            for pos in top_pos:
                region = classify_residue(int(pos) + 1, topo)
                residue_counter[region].append(float(np.abs(dim_tokens[pos])))
                all_positions.append({
                    "gpcr_id": gid,
                    "residue_idx": int(pos) + 1,
                    "aa": seq[int(pos)],
                    "region": region,
                    "token_value": float(dim_tokens[pos]),
                    "shap_value": float(dim_shap_values[i]),
                })

        # 汇总统计
        region_summary = {}
        for region, vals in residue_counter.items():
            region_summary[region] = {
                "count": len(vals),
                "mean_abs_token": round(float(np.mean(vals)), 5),
            }

        mapping_results[int(dim_idx)] = {
            "global_shap_importance": round(float(gpcr_importance[dim_idx]), 6),
            "region_summary": region_summary,
            "top_positions": all_positions[:50],  # 只保留前50条明细避免文件过大
        }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(mapping_results, f, indent=2, ensure_ascii=False)

    print(f"[OK] 映射结果保存至: {OUTPUT_FILE}")

    # 打印全局汇总
    print("\n--- Top 20 GPCR SHAP 维度 的残基区域分布 ---")
    global_region_counts = defaultdict(int)
    for dim_idx, info in mapping_results.items():
        for reg, s in info["region_summary"].items():
            global_region_counts[reg] += s["count"]

    total = sum(global_region_counts.values())
    for reg, cnt in sorted(global_region_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {reg:12s}: {cnt:4d} / {total} ({cnt / total * 100:.1f}%)")

    # 检查 ICL 富集
    icl_total = sum(c for r, c in global_region_counts.items() if r.startswith("ICL"))
    print(f"\n  ICL 总计: {icl_total} / {total} ({icl_total / total * 100:.1f}%)")


if __name__ == "__main__":
    main()
