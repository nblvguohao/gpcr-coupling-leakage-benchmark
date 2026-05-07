#!/usr/bin/env python3
"""
构建 GPCR-G蛋白 配对矩阵 (Strategy C Phase 1 Week 1)。

功能:
1. 读取现有 local 标签 (extended_labels.json) 作为 Gαq seed
2. 尝试加载手动下载的 GPCRdb coupling 文件 (CSV/TSV/XLSX)
3. 去重 (移除 prefix 重复项, 与 homology_clustered_cv.py 逻辑一致)
4. 同源聚类 (30% k-mer Jaccard threshold, 同一 GPCR 的所有配对必须在同一 cluster)
5. 输出:
   - paired_dataset/pairing_matrix_raw.csv
   - paired_dataset/sequence_clusters.json
   - paired_dataset/deduplicated_pairs.json

使用示例:
  python build_paired_matrix.py
  python build_paired_matrix.py --gpcrdb paired_dataset/gpcrdb_couplings.xlsx
"""

import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional

BASE = Path(__file__).parent
LABELS_FILE = BASE / "merged_dataset" / "extended_labels.json"
SEQUENCES_FILE = BASE / "merged_dataset" / "extended_sequences.json"
OUTPUT_DIR = BASE / "paired_dataset"
OUTPUT_DIR.mkdir(exist_ok=True)

LONG_CSV = OUTPUT_DIR / "gpcrdb_coupling_long.csv"
GPCRDB_GLOB = ["*.csv", "*.tsv", "*.txt", "*.xlsx", "*.xls"]
G_PROTEIN_FAMILIES = ["Gq", "Gi", "Gs", "G12_13"]


def load_local_seed():
    """加载本地 100 样本并生成 Gαq seed 配对。"""
    with open(LABELS_FILE) as f:
        labels = json.load(f)
    with open(SEQUENCES_FILE) as f:
        sequences = json.load(f)

    # 去重: 移除 prefix 重复 (与 homology_clustered_cv.py 一致)
    prefixed = set()
    for k in labels:
        if "_" in k and len(k.split("_")[0]) <= 2:
            base = k.split("_", 1)[1]
            if base in labels:
                prefixed.add(k)

    seed_pairs = []
    for uid in labels:
        if uid in prefixed:
            continue
        seq_rec = sequences.get(uid, {})
        seq = seq_rec["sequence"] if isinstance(seq_rec, dict) else seq_rec
        gq_label = int(labels[uid])
        seed_pairs.append({
            "gpcr_id": uid,
            "gpcr_sequence": seq,
            "g_protein_family": "Gq",
            "coupling": gq_label,
            "source": "local_seed",
        })

    return seed_pairs, sequences


def find_gpcrdb_file(specified_path: Optional[Path] = None) -> Optional[Path]:
    """查找本地 GPCRdb coupling 文件。"""
    if specified_path and specified_path.exists():
        return specified_path
    if LONG_CSV.exists():
        return LONG_CSV
    for pattern in GPCRDB_GLOB:
        candidates = list(OUTPUT_DIR.glob(pattern))
        # 排除我们已经生成的输出文件
        candidates = [c for c in candidates if c.name != "pairing_matrix_raw.csv"]
        if candidates:
            return candidates[0]
    return None


def parse_gpcrdb_file(path: Path) -> pd.DataFrame:
    """解析 GPCRdb coupling 文件为长表格式。"""
    if path.name == LONG_CSV.name:
        print(f"[INFO] 直接读取已整理的长格式文件: {path.name}")
        return pd.read_csv(path)
    suffix = path.suffix.lower()
    if suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in [".tsv", ".txt"]:
        df = pd.read_csv(path, sep="\t")
    else:
        raise ValueError(f"不支持的文件格式: {suffix}")

    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    # --- 启发式列名映射 ---
    # 常见列名变体
    gpcr_col_candidates = [
        "uniprot", "uniprot_id", "entry", "entry_name", "protein",
        "receptor", "gpcr", "receptor_name"
    ]
    gpcr_col = None
    for cand in gpcr_col_candidates:
        if cand in df.columns:
            gpcr_col = cand
            break

    if gpcr_col is None:
        # 如果找不到，假设第一列是 GPCR ID
        gpcr_col = df.columns[0]
        print(f"[WARN] 未找到 GPCR ID 列，假设第一列 '{gpcr_col}' 为 ID")

    # 尝试识别宽格式 (列是 G 蛋白 family)
    wide_cols = [c for c in df.columns if c in ["gq", "gi", "gs", "g12_13", "g12/13"]]

    records = []
    if wide_cols:
        print(f"[INFO] 检测到宽格式矩阵，G蛋白列: {wide_cols}")
        for _, row in df.iterrows():
            gpcr_id = str(row[gpcr_col]).strip()
            if not gpcr_id or gpcr_id.lower() in ["nan", "none"]:
                continue
            for gp_col in wide_cols:
                val = row[gp_col]
                label = _to_binary_label(val)
                if label is not None:
                    family = gp_col.upper().replace("G12_13", "G12_13").replace("G12/13", "G12_13")
                    records.append({
                        "gpcr_id": gpcr_id,
                        "g_protein_family": family,
                        "coupling": label,
                        "source": "gpcrdb_wide",
                    })
    else:
        # 长格式: 需要 g_protein_family 和 coupling 列
        gp_col_candidates = ["g_protein", "gprotein", "g_protein_family", "family", "subtype"]
        gp_col = None
        for cand in gp_col_candidates:
            if cand in df.columns:
                gp_col = cand
                break
        if gp_col is None:
            raise ValueError("无法自动识别 GPCRdb 文件格式。请确保包含 G蛋白列或宽格式矩阵。")

        coup_col_candidates = ["coupling", "value", "label", "coupled", "active"]
        coup_col = None
        for cand in coup_col_candidates:
            if cand in df.columns:
                coup_col = cand
                break
        if coup_col is None:
            # 找不到标签列则退后: 如果值是 0/1 的数值列
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            coup_col = numeric_cols[0] if numeric_cols else df.columns[-1]
            print(f"[WARN] 未找到 coupling 列，假设 '{coup_col}' 为标签")

        print(f"[INFO] 检测到长格式，G蛋白列='{gp_col}', 标签列='{coup_col}'")
        for _, row in df.iterrows():
            gpcr_id = str(row[gpcr_col]).strip()
            gp_family = str(row[gp_col]).strip()
            val = row[coup_col]
            label = _to_binary_label(val)
            if gpcr_id and gp_family and label is not None:
                records.append({
                    "gpcr_id": gpcr_id,
                    "g_protein_family": gp_family,
                    "coupling": label,
                    "source": "gpcrdb_long",
                })

    return pd.DataFrame.from_records(records)


def _to_binary_label(val):
    """将各种表示统一转为 0/1 或 None。"""
    if pd.isna(val):
        return None
    if isinstance(val, (int, float, np.floating)):
        if val >= 0.5:
            return 1
        elif val < 0.5:
            return 0
        return None
    s = str(val).strip().lower()
    if s in ["1", "yes", "true", "coupled", "active", "+", "primary", "secondary"]:
        return 1
    if s in ["0", "no", "false", "uncoupled", "inactive", "-", "none", "not"]:
        return 0
    return None


def sequence_identity(seq1: str, seq2: str) -> float:
    """k-mer Jaccard 相似度，与 homology_clustered_cv.py 保持一致。"""
    k = 3
    kmers1 = set(seq1[i : i + k] for i in range(len(seq1) - k + 1))
    kmers2 = set(seq2[i : i + k] for i in range(len(seq2) - k + 1))
    if not kmers1 or not kmers2:
        return 0.0
    inter = len(kmers1 & kmers2)
    union = len(kmers1 | kmers2)
    return inter / union


def cluster_sequences(seq_dict: Dict[str, str], threshold: float = 0.30):
    """
    单链接聚类。
    返回: cluster_list (list of list of IDs), sample_to_cluster (dict ID->cluster_index)
    """
    ids = list(seq_dict.keys())
    n = len(ids)
    print(f"[聚类] 计算 {n} 条序列的 k-mer Jaccard 相似度 (threshold={threshold}) ...")

    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            sim = sequence_identity(seq_dict[ids[i]], seq_dict[ids[j]])
            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] >= threshold:
                if find(i) != find(j):
                    union(i, j)

    clusters = defaultdict(list)
    for i in range(n):
        clusters[find(i)].append(ids[i])

    cluster_list = list(clusters.values())
    cluster_list.sort(key=len, reverse=True)

    sample_to_cluster = {}
    for cid, members in enumerate(cluster_list):
        for m in members:
            sample_to_cluster[m] = cid

    size_counts = defaultdict(int)
    for c in cluster_list:
        size_counts[len(c)] += 1
    print(f"       共 {len(cluster_list)} 个簇，大小分布: ", end="")
    for sz in sorted(size_counts.keys(), reverse=True):
        print(f"{sz}×{size_counts[sz]}  ", end="")
    print()

    return cluster_list, sample_to_cluster


def deduplicate_pairs(pairs_df: pd.DataFrame) -> pd.DataFrame:
    """去除完全重复的 (gpcr_id, g_protein_family) 行。冲突时优先 GPCRdb。"""
    # 排序让 gpcrdb 来源排前面 (source 字母序 gpcrdb < local)
    df = pairs_df.sort_values(by=["gpcr_id", "g_protein_family", "source"])
    df = df.drop_duplicates(subset=["gpcr_id", "g_protein_family"], keep="first")
    return df.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="构建 GPCR-G蛋白 配对矩阵")
    parser.add_argument("--gpcrdb", type=Path, default=None,
                        help="手动指定的 GPCRdb coupling 文件路径")
    args = parser.parse_args()

    print("=" * 70)
    print("  构建 GPCR-G蛋白 配对矩阵")
    print("=" * 70)

    # 1. 加载本地 seed
    seed_pairs, sequences = load_local_seed()
    print(f"[INFO] 本地 Gαq seed 配对数: {len(seed_pairs)}")

    all_records = seed_pairs.copy()

    # 2. 尝试加载 GPCRdb 数据
    gpcrdb_path = find_gpcrdb_file(args.gpcrdb)
    if gpcrdb_path:
        print(f"[INFO] 发现 GPCRdb 文件: {gpcrdb_path}")
        try:
            gpcrdb_df = parse_gpcrdb_file(gpcrdb_path)
            print(f"[INFO] 解析出 GPCRdb 记录数: {len(gpcrdb_df)}")
            # 将 DataFrame 转为 dict records 并合并
            gpcrdb_records = gpcrdb_df.to_dict(orient="records")
            # 保留所有 GPCRdb 记录 (前提: fetch_uniprot_sequences.py 已下载全部序列)
            print(f"[INFO] 保留 GPCRdb 全部记录数: {len(gpcrdb_records)}")
            all_records.extend(gpcrdb_records)
        except Exception as e:
            print(f"[ERROR] 解析 GPCRdb 文件失败: {e}")
    else:
        print("[WARN] 未找到 GPCRdb coupling 文件。")
        print("       当前仅使用本地 Gαq seed 配对。")
        print("       请从 https://gpcrdb.org/couplings/ 下载 coupling 数据")
        print("       并放入 paired_dataset/ 目录后重新运行。")

    # 3. 组装 DataFrame
    df = pd.DataFrame.from_records(all_records)

    # 4. 去重
    df = deduplicate_pairs(df)
    print(f"[INFO] 去重后总配对数: {len(df)}")

    # 5. 聚类 (基于 GPCR 唯一序列集合)
    unique_gpcrs = df["gpcr_id"].unique().tolist()

    def get_sequence(seq_dict, gid):
        if gid in seq_dict:
            rec = seq_dict[gid]
            return rec["sequence"] if isinstance(rec, dict) else rec
        for prefix in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789":
            prefixed = f"{prefix}_{gid}"
            if prefixed in seq_dict:
                rec = seq_dict[prefixed]
                return rec["sequence"] if isinstance(rec, dict) else rec
        return None

    gpcr_seqs = {}
    for gid in unique_gpcrs:
        seq = get_sequence(sequences, gid)
        if seq is not None:
            gpcr_seqs[gid] = seq
        else:
            print(f"[WARN] Missing sequence for GPCR {gid}; excluded from clustering.")

    cluster_list, sample_to_cluster = cluster_sequences(gpcr_seqs, threshold=0.30)

    # 6. 为每条配对附加 cluster_id
    df["cluster_id"] = df["gpcr_id"].map(sample_to_cluster)

    # 7. 保存
    csv_path = OUTPUT_DIR / "pairing_matrix_raw.csv"
    df.to_csv(csv_path, index=False)

    clusters_out = {
        "n_clusters": len(cluster_list),
        "threshold": 0.30,
        "clusters": [
            {"cluster_id": i, "members": members}
            for i, members in enumerate(cluster_list)
        ],
    }
    clusters_path = OUTPUT_DIR / "sequence_clusters.json"
    with open(clusters_path, "w") as f:
        json.dump(clusters_out, f, indent=2, ensure_ascii=False)

    dedup_pairs = df[["gpcr_id", "g_protein_family", "coupling", "cluster_id", "source"]].to_dict(orient="records")
    dedup_path = OUTPUT_DIR / "deduplicated_pairs.json"
    with open(dedup_path, "w") as f:
        json.dump({
            "n_pairs": len(dedup_pairs),
            "n_unique_gpcrs": len(unique_gpcrs),
            "pairs": dedup_pairs,
        }, f, indent=2, ensure_ascii=False)

    # 8. 统计报告
    print("\n" + "=" * 70)
    print("  统计报告")
    print("=" * 70)
    print(f"  总配对数:        {len(df)}")
    print(f"  独立 GPCR 数:    {len(unique_gpcrs)}")
    print(f"  簇数:            {len(cluster_list)}")
    print(f"  各 G蛋白 family 配对分布:")
    for fam, count in df["g_protein_family"].value_counts().items():
        print(f"    {fam:12s}: {count}")
    print(f"  正/负样本分布:")
    for lbl, count in df["coupling"].value_counts().items():
        print(f"    label={lbl}: {count}")
    print(f"\n[OK] 输出文件:")
    print(f"    {csv_path}")
    print(f"    {clusters_path}")
    print(f"    {dedup_path}")

    # Go/No-Go checkpoint 提示
    print("\n" + "=" * 70)
    print("  Week 1 Go/No-Go Checkpoint")
    print("=" * 70)
    if len(df) >= 400:
        print("  [PASS] 去重后配对数 >= 400，通过最低标准。")
    elif len(df) >= 300:
        print("  [WARN] 配对数 300-400，勉强可用，建议继续扩充。")
    else:
        print("  [FAIL] 配对数 < 300，必须扩大数据源 (加入 orthologs / 其他物种)。")


if __name__ == "__main__":
    main()
