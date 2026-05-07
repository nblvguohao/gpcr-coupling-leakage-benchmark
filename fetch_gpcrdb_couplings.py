#!/usr/bin/env python3
"""
批量爬取/整理 GPCRdb coupling 数据 (备用脚本)。

功能:
1. 从 protwis/gpcrdb_data GitHub 自动下载:
   - iuphar_coupling_data.csv (主要)
   - GPCR-G_protein_couplings.xlsx (补充)
2. 解析 IUPHAR CSV 为 (GPCR UniProt, G-protein-family, coupling=0/1) 长格式
3. 可选解析 XLSX 的 'GtoPdb' sheet 作为补充/验证
4. 与本地序列 ID 做匹配 (支持 prefix ID 如 T_P21731 -> P21731)
5. 生成交付物供 build_paired_matrix.py 直接消费:
   - paired_dataset/gpcrdb_coupling_long.csv
   - paired_dataset/gpcrdb_coupling_summary.json

作者: Claude Code (Strategy C Phase 1 Week 1 备用数据方案)
"""

import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Set
from collections import defaultdict

BASE = Path(__file__).parent
OUTPUT_DIR = BASE / "paired_dataset"
OUTPUT_DIR.mkdir(exist_ok=True)

GITHUB_RAW = "https://raw.githubusercontent.com/protwis/gpcrdb_data/master/g_protein_data"
FILES = {
    "iuphar_csv": f"{GITHUB_RAW}/iuphar_coupling_data.csv",
    "couplings_xlsx": f"{GITHUB_RAW}/GPCR-G_protein_couplings.xlsx",
}

LOCAL_SEQUENCES_FILE = BASE / "merged_dataset" / "extended_sequences.json"
LOCAL_LABELS_FILE = BASE / "merged_dataset" / "extended_labels.json"


def download_file(url: str, out_path: Path) -> bool:
    """用 requests 下载文件到本地。"""
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
    """加载本地所有 GPCR ID (含 prefix 变体)。"""
    with open(LOCAL_SEQUENCES_FILE) as f:
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
    解析 iuphar_coupling_data.csv 为长格式配对矩阵。
    列: uniprot, primarytransducer, secondarytransducer
    G蛋白 family: Gq, Gi, Gs, G12_13
    """
    df = pd.read_csv(path)
    # 标准化列名
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
    # 去重: 同一个 (gpcr_id, family) 只保留一条
    out = out.drop_duplicates(subset=["gpcr_id", "g_protein_family"])
    return out


def parse_xlsx_gtodb(path: Path) -> pd.DataFrame:
    """
    解析 XLSX 中的 'GtoPdb' sheet 作为补充。
    该 sheet 通常与 IUPHAR 数据同源，但可能包含更细的分类。
    """
    try:
        xls = pd.ExcelFile(path)
        if "GtoPdb" not in xls.sheet_names:
            print("[WARN] XLSX 中没有 'GtoPdb' sheet，跳过补充解析。")
            return pd.DataFrame()
        df = pd.read_excel(xls, sheet_name="GtoPdb")
        # 标准化列名
        df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
        print(f"[INFO] GtoPdb sheet columns: {df.columns.tolist()}")
        # 目前只打印信息，暂不开发自动解析(因为列结构未知且IUPHAR CSV已足够)
        return pd.DataFrame()
    except Exception as e:
        print(f"[WARN] XLSX 解析失败: {e}")
        return pd.DataFrame()


def normalize_prefix(gpcr_id: str) -> str:
    """如果 ID 是 prefix 形式 (如 T_P21731)，返回基础 UniProt ID。"""
    if "_" in gpcr_id and len(gpcr_id.split("_")[0]) <= 2:
        return gpcr_id.split("_", 1)[1]
    return gpcr_id


def main():
    parser = argparse.ArgumentParser(description="爬取并整理 GPCRdb coupling 数据")
    parser.add_argument("--skip-download", action="store_true", help="跳过下载，使用本地已有文件")
    parser.add_argument("--keep-all-gpcrs", action="store_true",
                        help="保留 IUPHAR 中所有 GPCR (不仅限于本地已有的86个)")
    args = parser.parse_args()

    print("=" * 70)
    print("  GPCRdb Coupling 数据自动获取与整理")
    print("=" * 70)

    # 1. 下载
    csv_path = OUTPUT_DIR / "iuphar_coupling_data.csv"
    xlsx_path = OUTPUT_DIR / "GPCR-G_protein_couplings.xlsx"

    if not args.skip_download:
        download_file(FILES["iuphar_csv"], csv_path)
        download_file(FILES["couplings_xlsx"], xlsx_path)
    else:
        print("[INFO] 跳过下载，使用本地缓存。")

    # 2. 解析 IUPHAR CSV
    print("\n[INFO] 解析 IUPHAR coupling CSV ...")
    df_iuphar = parse_iuphar_csv(csv_path)
    print(f"[INFO] IUPHAR 解析出 {len(df_iuphar)} 条 (gpcr_id, family) 配对")

    # 3. 本地 ID 匹配
    local_ids = load_local_gpcr_ids()
    print(f"[INFO] 本地 GPCR ID 集合大小: {len(local_ids)}")

    # 将 IUPHAR 的 gpcr_id 也做 normalize_prefix 尝试匹配
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

    # 4. 合并本地 Gq seed 数据 (保证我们原有的86个GPCR的序列都在)
    with open(LOCAL_LABELS_FILE) as f:
        local_labels = json.load(f)

    # 去重 prefix
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

    # 合并: IUPHAR 优先 (因为更全)，local seed 补充
    combined = pd.concat([df_matched, df_seed], ignore_index=True)
    # 去重: local seed 中已有的 (gpcr_id, Gq) 如果在 IUPHAR 中也有，用 IUPHAR 的
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

    # 5. 保存
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
    print("  Week 1 Go/No-Go Checkpoint")
    print("=" * 70)
    if len(combined) >= 400:
        print(f"  [PASS] 配对数 {len(combined)} >= 400")
    elif len(combined) >= 300:
        print(f"  [WARN] 配对数 {len(combined)} (300-400)")
    else:
        print(f"  [FAIL] 配对数 {len(combined)} < 300，仍需扩充")


if __name__ == "__main__":
    main()
