#!/usr/bin/env python3
"""
Build the GPCR–G protein pairing dataset from scratch.

This script implements the full data pipeline for Strategy C Phase 1 Week 1,
combining two previously-separate stages into a single entry point:

  Step "fetch":
    - Downloads GPCRdb coupling data from the protwis/gpcrdb_data GitHub
      repository (iuphar_coupling_data.csv and GPCR-G_protein_couplings.xlsx).
    - Parses the IUPHAR CSV into long-format (gpcr_id, g_protein_family,
      coupling) records covering Gq, Gi, Gs, and G12_13 families.
    - Optionally parses the XLSX GtoPdb sheet as a supplementary source
      (function defined but not automatically invoked — structure-dependent).
    - Matches IUPHAR GPCR IDs against the local sequence dataset, supporting
      prefix-style IDs (e.g. T_P21731 -> P21731).
    - Merges IUPHAR data with local Galpha-q seed labels, preferring IUPHAR
      annotations when both sources cover the same (gpcr_id, family) pair.
    - Outputs:
        data/gpcrdb_coupling_long.csv
        data/gpcrdb_coupling_summary.json

  Step "build":
    - Loads local Galpha-q seed labels and sequences from the merged_dataset
      directory.
    - Loads the GPCRdb coupling long-format CSV (produced by "fetch") or an
      alternative GPCRdb file supplied via --gpcrdb.
    - Parses GPCRdb files with heuristic column detection that handles both
      wide-format matrices (G-protein family columns) and long-format tables.
    - Deduplicates (gpcr_id, g_protein_family) pairs, preferring GPCRdb
      sources over local seed where they conflict.
    - Performs single-linkage clustering of GPCR sequences using k-mer
      Jaccard similarity (k=3, threshold=0.30), consistent with
      homology_clustered_cv.py.
    - Attaches cluster IDs to every pairing record.
    - Outputs:
        data/pairing_matrix_raw.csv
        data/sequence_clusters.json
        data/deduplicated_pairs.json

  Step "all" (default):
    - Runs fetch, then build sequentially.  The fetch output
      (gpcrdb_coupling_long.csv) is consumed directly by the build step.

Usage examples:
  python build_dataset.py --step fetch --skip-download
  python build_dataset.py --step fetch --keep-all-gpcrs
  python build_dataset.py --step build --gpcrdb data/some_couplings.xlsx
  python build_dataset.py --step all

Author: Claude Code (Strategy C Phase 1 Week 1)
"""

import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict

# ============================================================================
# Constants
# ============================================================================

BASE = Path(__file__).parent
OUTPUT_DIR = BASE.parent / "data"
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Fetch step ---
GITHUB_RAW = "https://raw.githubusercontent.com/protwis/gpcrdb_data/master/g_protein_data"
FILES = {
    "iuphar_csv": f"{GITHUB_RAW}/iuphar_coupling_data.csv",
    "couplings_xlsx": f"{GITHUB_RAW}/GPCR-G_protein_couplings.xlsx",
}

# --- Shared (used by both fetch and build) ---
SEQUENCES_FILE = BASE / "merged_dataset" / "extended_sequences.json"
LABELS_FILE = BASE / "merged_dataset" / "extended_labels.json"

# --- Build step ---
LONG_CSV = OUTPUT_DIR / "gpcrdb_coupling_long.csv"
GPCRDB_GLOB = ["*.csv", "*.tsv", "*.txt", "*.xlsx", "*.xls"]
G_PROTEIN_FAMILIES = ["Gq", "Gi", "Gs", "G12_13"]


# ============================================================================
# Section 1: Fetch — Download & Parse GPCRdb Coupling Data
# ============================================================================

def normalize_prefix(gpcr_id: str) -> str:
    """If the ID has a prefix form (e.g. T_P21731), return the base UniProt ID."""
    if "_" in gpcr_id and len(gpcr_id.split("_")[0]) <= 2:
        return gpcr_id.split("_", 1)[1]
    return gpcr_id


def download_file(url: str, out_path: Path) -> bool:
    """Download a file from *url* to *out_path* using requests."""
    import requests
    try:
        print(f"[DOWNLOAD] {url}")
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        out_path.write_bytes(r.content)
        print(f"  -> Saved: {out_path} ({len(r.content)} bytes)")
        return True
    except Exception as e:
        print(f"  -> ERROR: {e}")
        return False


def load_local_gpcr_ids() -> Set[str]:
    """Load all local GPCR IDs (including prefix variants)."""
    with open(SEQUENCES_FILE) as f:
        sequences = json.load(f)
    ids = set()
    for uid in sequences:
        ids.add(uid)
        if "_" in uid and len(uid.split("_")[0]) <= 2:
            base = uid.split("_", 1)[1]
            ids.add(base)
    return ids


def parse_iuphar_csv(path: Path) -> pd.DataFrame:
    """
    Parse iuphar_coupling_data.csv into long-format pairing records.

    Columns: uniprot, primarytransducer, secondarytransducer
    G-protein families: Gq, Gi, Gs, G12_13
    """
    df = pd.read_csv(path)
    # Normalise column names
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    records: List[Dict] = []
    for _, row in df.iterrows():
        uniprot = str(row.get("uniprot", "")).strip()
        if not uniprot or uniprot.lower() == "nan":
            continue

        primary = str(row.get("primarytransducer", "")).strip()
        secondary = str(row.get("secondarytransducer", "")).strip()

        families: Set[str] = set()
        for val in [primary, secondary]:
            if not val or val.lower() == "nan":
                continue
            v = val.lower()
            if "gi" in v or "go" in v:
                families.add("Gi")
            if "gs" in v:
                families.add("Gs")
            if "gq" in v or "g11" in v:
                families.add("Gq")
            if "g12" in v or "g13" in v:
                families.add("G12_13")

        for fam in ["Gq", "Gi", "Gs", "G12_13"]:
            records.append({
                "gpcr_id": uniprot,
                "g_protein_family": fam,
                "coupling": 1 if fam in families else 0,
                "source": "gpcrdb_iuphar",
            })

    out = pd.DataFrame.from_records(records)
    # Deduplicate: only one row per (gpcr_id, family)
    out = out.drop_duplicates(subset=["gpcr_id", "g_protein_family"])
    return out


def parse_xlsx_gtodb(path: Path) -> pd.DataFrame:
    """
    Parse the 'GtoPdb' sheet from the XLSX as supplementary data.

    The sheet usually shares the same source as IUPHAR data but may contain
    finer-grained classifications.  Currently returns an empty DataFrame
    because the column structure is dataset-dependent.
    """
    try:
        xls = pd.ExcelFile(path)
        if "GtoPdb" not in xls.sheet_names:
            print("[WARN] XLSX 中没有 'GtoPdb' sheet，跳过补充解析。")
            return pd.DataFrame()
        df = pd.read_excel(xls, sheet_name="GtoPdb")
        # Normalise column names
        df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
        print(f"[INFO] GtoPdb sheet columns: {df.columns.tolist()}")
        # Currently only prints column information; automatic parsing is not
        # yet implemented because the column layout is unknown and the IUPHAR
        # CSV already provides sufficient coverage.
        return pd.DataFrame()
    except Exception as e:
        print(f"[WARN] XLSX 解析失败: {e}")
        return pd.DataFrame()


def run_fetch(args):
    """Execute the fetch step: download and preprocess GPCRdb coupling data."""
    print("=" * 70)
    print("  STEP 1: GPCRdb Coupling 数据自动获取与整理")
    print("=" * 70)

    # 1. Download
    csv_path = OUTPUT_DIR / "iuphar_coupling_data.csv"
    xlsx_path = OUTPUT_DIR / "GPCR-G_protein_couplings.xlsx"

    if not args.skip_download:
        download_file(FILES["iuphar_csv"], csv_path)
        download_file(FILES["couplings_xlsx"], xlsx_path)
    else:
        print("[INFO] 跳过下载，使用本地缓存。")

    # 2. Parse IUPHAR CSV
    print("\n[INFO] 解析 IUPHAR coupling CSV ...")
    df_iuphar = parse_iuphar_csv(csv_path)
    print(f"[INFO] IUPHAR 解析出 {len(df_iuphar)} 条 (gpcr_id, family) 配对")

    # 3. Match against local GPCR IDs
    local_ids = load_local_gpcr_ids()
    print(f"[INFO] 本地 GPCR ID 集合大小: {len(local_ids)}")

    # Normalise prefix IDs in IUPHAR data for matching
    df_iuphar["gpcr_id_norm"] = df_iuphar["gpcr_id"].apply(normalize_prefix)

    if args.keep_all_gpcrs:
        df_matched = df_iuphar.copy()
        df_matched["gpcr_id"] = df_matched["gpcr_id_norm"]
        df_matched = df_matched.drop(columns=["gpcr_id_norm"])
    else:
        mask = df_iuphar["gpcr_id_norm"].isin(local_ids) | df_iuphar["gpcr_id"].isin(local_ids)
        df_matched = df_iuphar[mask].copy()
        df_matched["gpcr_id"] = df_matched["gpcr_id_norm"]
        df_matched = df_matched.drop(columns=["gpcr_id_norm"])

    df_matched = df_matched.drop_duplicates(subset=["gpcr_id", "g_protein_family"])
    print(f"[INFO] 匹配到本地/保留的配对数: {len(df_matched)}")

    # 4. Merge local Gq seed data (ensures all 86 original GPCRs are present)
    with open(LABELS_FILE) as f:
        local_labels = json.load(f)

    # Deduplicate prefix entries
    prefixed = set()
    for k in local_labels:
        if "_" in k and len(k.split("_")[0]) <= 2:
            base = k.split("_", 1)[1]
            if base in local_labels:
                prefixed.add(k)

    seed_records = []
    for uid in local_labels:
        if uid in prefixed:
            continue
        seed_records.append({
            "gpcr_id": uid,
            "g_protein_family": "Gq",
            "coupling": int(local_labels[uid]),
            "source": "local_seed",
        })
    df_seed = pd.DataFrame.from_records(seed_records)

    # Merge: IUPHAR preferred (more comprehensive), local seed as supplement
    combined = pd.concat([df_matched, df_seed], ignore_index=True)
    # Sort so gpcrdb_iuphar rows appear before local_seed
    combined = combined.sort_values("source", key=lambda col: col.map(lambda x: 0 if x == "gpcrdb_iuphar" else 1))
    combined = combined.drop_duplicates(subset=["gpcr_id", "g_protein_family"], keep="first")
    combined = combined.reset_index(drop=True)

    print(f"\n[INFO] 合并后总配对数: {len(combined)}")
    print("[INFO] 各 G蛋白 family 分布:")
    for fam, cnt in combined["g_protein_family"].value_counts().items():
        print(f"    {fam:12s}: {cnt}")
    print("[INFO] 正/负样本分布:")
    for lbl, cnt in combined["coupling"].value_counts().items():
        print(f"    label={lbl}: {cnt}")

    # 5. Save outputs
    long_csv = OUTPUT_DIR / "gpcrdb_coupling_long.csv"
    combined.to_csv(long_csv, index=False)
    print(f"\n[OK] 长格式配对矩阵已保存: {long_csv}")

    summary = {
        "n_pairs": int(len(combined)),
        "n_unique_gpcrs": int(combined["gpcr_id"].nunique()),
        "family_counts": combined["g_protein_family"].value_counts().to_dict(),
        "coupling_counts": combined["coupling"].value_counts().to_dict(),
        "sources": combined["source"].value_counts().to_dict(),
    }
    summary_json = OUTPUT_DIR / "gpcrdb_coupling_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[OK] 摘要已保存: {summary_json}")

    # 6. Go/No-Go checkpoint
    print("\n" + "=" * 70)
    print("  Week 1 Go/No-Go Checkpoint (Fetch)")
    print("=" * 70)
    if len(combined) >= 400:
        print(f"  [PASS] 配对数 {len(combined)} >= 400")
    elif len(combined) >= 300:
        print(f"  [WARN] 配对数 {len(combined)} (300-400)")
    else:
        print(f"  [FAIL] 配对数 {len(combined)} < 300，仍需扩充")


# ============================================================================
# Section 2: Build — Cluster & Build Final Pairing Matrix
# ============================================================================

def load_local_seed():
    """Load local 100-sample labels and generate Galpha-q seed pairs."""
    with open(LABELS_FILE) as f:
        labels = json.load(f)
    with open(SEQUENCES_FILE) as f:
        sequences = json.load(f)

    # Deduplicate: remove prefix duplicates (consistent with homology_clustered_cv.py)
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
    """Locate a local GPCRdb coupling file."""
    if specified_path and specified_path.exists():
        return specified_path
    if LONG_CSV.exists():
        return LONG_CSV
    for pattern in GPCRDB_GLOB:
        candidates = list(OUTPUT_DIR.glob(pattern))
        # Exclude files that we have already generated as output
        candidates = [c for c in candidates if c.name != "pairing_matrix_raw.csv"]
        if candidates:
            return candidates[0]
    return None


def parse_gpcrdb_file(path: Path) -> pd.DataFrame:
    """Parse a GPCRdb coupling file into long-table format."""
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

    # --- Heuristic column-name mapping ---
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
        # If no recognised column found, assume the first column is the GPCR ID
        gpcr_col = df.columns[0]
        print(f"[WARN] 未找到 GPCR ID 列，假设第一列 '{gpcr_col}' 为 ID")

    # Detect wide format (columns are G-protein families)
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
        # Long format: need g_protein_family and coupling columns
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
            # Fallback: use the first numeric column as the label column
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
    """Convert various representations to 0/1 or None."""
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
    """k-mer Jaccard similarity, consistent with homology_clustered_cv.py."""
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
    Single-linkage clustering using k-mer Jaccard similarity.

    Returns:
        cluster_list: list of lists, each inner list contains member IDs.
        sample_to_cluster: dict mapping ID -> cluster_index.
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
    """Remove duplicate (gpcr_id, g_protein_family) rows.  Conflicts prefer GPCRdb sources."""
    # Sort so GPCRdb-sourced rows appear before local seed
    df = pairs_df.sort_values(by=["gpcr_id", "g_protein_family", "source"])
    df = df.drop_duplicates(subset=["gpcr_id", "g_protein_family"], keep="first")
    return df.reset_index(drop=True)


def run_build(args):
    """Execute the build step: cluster and build the final pairing matrix."""
    print("=" * 70)
    print("  STEP 2: 构建 GPCR-G蛋白 配对矩阵")
    print("=" * 70)

    # 1. Load local seed
    seed_pairs, sequences = load_local_seed()
    print(f"[INFO] 本地 Gαq seed 配对数: {len(seed_pairs)}")

    all_records = seed_pairs.copy()

    # 2. Attempt to load GPCRdb data
    gpcrdb_path = find_gpcrdb_file(args.gpcrdb)
    if gpcrdb_path:
        print(f"[INFO] 发现 GPCRdb 文件: {gpcrdb_path}")
        try:
            gpcrdb_df = parse_gpcrdb_file(gpcrdb_path)
            print(f"[INFO] 解析出 GPCRdb 记录数: {len(gpcrdb_df)}")
            # Convert DataFrame to dict records and merge
            gpcrdb_records = gpcrdb_df.to_dict(orient="records")
            # Retain all GPCRdb records (assuming fetch_uniprot_sequences.py
            # has already downloaded all sequences)
            print(f"[INFO] 保留 GPCRdb 全部记录数: {len(gpcrdb_records)}")
            all_records.extend(gpcrdb_records)
        except Exception as e:
            print(f"[ERROR] 解析 GPCRdb 文件失败: {e}")
    else:
        print("[WARN] 未找到 GPCRdb coupling 文件。")
        print("       当前仅使用本地 Gαq seed 配对。")
        print("       请从 https://gpcrdb.org/couplings/ 下载 coupling 数据")
        print("       并放入 paired_dataset/ 目录后重新运行。")

    # 3. Assemble DataFrame
    df = pd.DataFrame.from_records(all_records)

    # 4. Deduplicate
    df = deduplicate_pairs(df)
    print(f"[INFO] 去重后总配对数: {len(df)}")

    # 5. Cluster (based on unique GPCR sequence set)
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

    # 6. Attach cluster_id to each pair
    df["cluster_id"] = df["gpcr_id"].map(sample_to_cluster)

    # 7. Save outputs
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

    # 8. Statistics report
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

    # Go/No-Go checkpoint
    print("\n" + "=" * 70)
    print("  Week 1 Go/No-Go Checkpoint (Build)")
    print("=" * 70)
    if len(df) >= 400:
        print("  [PASS] 去重后配对数 >= 400，通过最低标准。")
    elif len(df) >= 300:
        print("  [WARN] 配对数 300-400，勉强可用，建议继续扩充。")
    else:
        print("  [FAIL] 配对数 < 300，必须扩大数据源 (加入 orthologs / 其他物种)。")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build the GPCR–G protein pairing dataset "
                    "(fetch GPCRdb data and/or build the paired matrix with clustering)."
    )
    parser.add_argument(
        "--step", choices=["fetch", "build", "all"], default="all",
        help="Pipeline step to run: 'fetch' (download & preprocess), "
             "'build' (cluster & build matrix), or 'all' (both, default)."
    )
    # Fetch-specific arguments
    parser.add_argument(
        "--skip-download", action="store_true",
        help="[fetch] Skip download, use locally cached files."
    )
    parser.add_argument(
        "--keep-all-gpcrs", action="store_true",
        help="[fetch] Keep all GPCRs from IUPHAR, not just those matching local IDs."
    )
    # Build-specific arguments
    parser.add_argument(
        "--gpcrdb", type=Path, default=None,
        help="[build] Path to a specific GPCRdb coupling file."
    )
    args = parser.parse_args()

    if args.step in ("fetch", "all"):
        run_fetch(args)
    if args.step in ("build", "all"):
        run_build(args)


if __name__ == "__main__":
    main()
