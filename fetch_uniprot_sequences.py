#!/usr/bin/env python3
"""
批量下载 GPCR UniProt 序列并合并到本地序列库。

功能:
1. 读取 paired_dataset/iuphar_coupling_data.csv 获取所有 UniProt ID
2. 与 merged_dataset/extended_sequences.json 比对，找出缺失序列
3. 使用 UniProt REST API 批量下载缺失序列 (每批 100 个)
4. 合并保存回 extended_sequences.json

输出:
- merged_dataset/extended_sequences.json (更新后)
- paired_dataset/uniprot_fetch_log.json (下载日志)
"""

import json
import re
import time
import argparse
import requests
import pandas as pd
from pathlib import Path
from typing import Set, Dict, List, Optional

BASE = Path(__file__).parent
SEQUENCES_FILE = BASE / "merged_dataset" / "extended_sequences.json"
IUPHAR_CSV = BASE / "paired_dataset" / "iuphar_coupling_data.csv"
XLSX_FILE = BASE / "paired_dataset" / "GPCR-G_protein_couplings.xlsx"
OUTPUT_LOG = BASE / "paired_dataset" / "uniprot_fetch_log.json"

UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
BATCH_SIZE = 50
REQUEST_TIMEOUT = 60
SLEEP_BETWEEN_BATCHES = 1.0

_UNIPROT_ACC_RE = re.compile(
    r"^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$|^[OPQ][0-9][A-Z0-9]{3}[0-9]$"
)


def is_valid_uniprot_accession(acc: str) -> bool:
    return bool(_UNIPROT_ACC_RE.match(acc))


def load_existing_sequences() -> Dict[str, dict]:
    """加载本地已有序列库。"""
    if not SEQUENCES_FILE.exists():
        return {}
    with open(SEQUENCES_FILE, encoding="utf-8") as f:
        return json.load(f)


def extract_uniprot_ids_from_iuphar(path: Path) -> Set[str]:
    """从 IUPHAR CSV 提取所有 UniProt accession。"""
    df = pd.read_csv(path)
    col = None
    for c in df.columns:
        if str(c).strip().lower() in {"uniprot", "uniprot id", "uniprot_id"}:
            col = c
            break
    if col is None:
        raise ValueError(f"在 {path} 中未找到 UniProt 列。列名: {list(df.columns)}")
    ids = set(df[col].dropna().astype(str).str.strip())
    ids = {uid for uid in ids if uid and uid.lower() != "nan" and is_valid_uniprot_accession(uid)}
    return ids


def extract_uniprot_ids_from_xlsx(path: Path) -> Set[str]:
    """从 XLSX 的 GtoPdb / RecNamesAll sheet 提取额外 UniProt accession。"""
    try:
        xls = pd.ExcelFile(path)
    except Exception as e:
        print(f"[WARN] 无法读取 XLSX: {e}")
        return set()

    ids: Set[str] = set()
    sheet_cfg = {
        "GtoPdb": ["uniprot", "uniprotid"],
        "RecNamesAll": ["accession"],
    }
    for sheet_name, allowed_cols in sheet_cfg.items():
        if sheet_name not in xls.sheet_names:
            continue
        df = pd.read_excel(xls, sheet_name=sheet_name)
        for c in df.columns:
            low = str(c).strip().lower().replace(" ", "").replace("\n", "")
            if low in allowed_cols:
                sheet_ids = set(df[c].dropna().astype(str).str.strip())
                sheet_ids = {uid for uid in sheet_ids if uid and uid.lower() != "nan" and is_valid_uniprot_accession(uid)}
                ids.update(sheet_ids)
                break
    return ids


def fetch_uniprot_batch(accessions: List[str]) -> Dict[str, dict]:
    """
    通过 UniProt search API 批量获取序列信息。
    返回: {accession: {sequence, protein_name, organism, length}}
    """
    if not accessions:
        return {}

    query = " OR ".join(f"accession:{acc}" for acc in accessions)
    params = {
        "query": query,
        "format": "json",
        "size": len(accessions),
    }

    try:
        r = requests.get(UNIPROT_SEARCH, params=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"  [ERROR] 批量请求失败 ({len(accessions)} IDs): {e}")
        return {}

    results: Dict[str, dict] = {}
    for entry in data.get("results", []):
        acc = entry.get("primaryAccession", "")
        seq_info = entry.get("sequence", {})
        sequence = seq_info.get("value", "")
        length = seq_info.get("length", 0)

        # 提取推荐蛋白名
        protein_name = ""
        prot_desc = entry.get("proteinDescription", {})
        rec_name = prot_desc.get("recommendedName", {})
        protein_name = rec_name.get("fullName", {}).get("value", "")
        if not protein_name:
            # 尝试 shortName
            short_names = rec_name.get("shortNames", [])
            if short_names:
                protein_name = short_names[0].get("value", "")

        # 提取物种
        organism = ""
        orgs = entry.get("organisms", [])
        if orgs:
            organism = orgs[0].get("scientificName", "")

        if sequence and acc:
            results[acc] = {
                "sequence": sequence,
                "protein_name": protein_name,
                "organism": organism,
                "length": length or len(sequence),
            }

    return results


def fetch_missing_sequences(missing_ids: List[str]) -> Dict[str, dict]:
    """分批下载缺失序列，带重试。"""
    fetched: Dict[str, dict] = {}
    failed: List[str] = []

    total_batches = (len(missing_ids) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(total_batches):
        batch = missing_ids[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
        print(f"[BATCH {i+1}/{total_batches}] 请求 {len(batch)} 个序列 ...")
        batch_results = fetch_uniprot_batch(batch)

        for acc in batch:
            if acc in batch_results:
                fetched[acc] = batch_results[acc]
            else:
                failed.append(acc)

        if i < total_batches - 1:
            time.sleep(SLEEP_BETWEEN_BATCHES)

    # 对失败项进行 1 次单条重试
    if failed:
        print(f"[RETRY] 对 {len(failed)} 个失败项进行单条重试 ...")
        still_failed = []
        for acc in failed:
            result = fetch_uniprot_batch([acc])
            if acc in result:
                fetched[acc] = result[acc]
            else:
                still_failed.append(acc)
            time.sleep(0.3)
        failed = still_failed

    if failed:
        print(f"[WARN] 最终仍有 {len(failed)} 个序列无法获取: {failed[:10]}...")

    return fetched


def main():
    parser = argparse.ArgumentParser(description="批量下载 GPCR UniProt 序列")
    parser.add_argument("--dry-run", action="store_true", help="仅统计缺失数量，不实际下载")
    parser.add_argument("--from-xlsx", action="store_true", help="同时扫描 XLSX 中的额外 UniProt ID")
    args = parser.parse_args()

    print("=" * 70)
    print("  批量下载 GPCR UniProt 序列")
    print("=" * 70)

    # 1. 收集目标 UniProt ID
    target_ids = extract_uniprot_ids_from_iuphar(IUPHAR_CSV)
    print(f"[INFO] IUPHAR CSV 中 UniProt ID 数量: {len(target_ids)}")

    if args.from_xlsx and XLSX_FILE.exists():
        xlsx_ids = extract_uniprot_ids_from_xlsx(XLSX_FILE)
        extra = xlsx_ids - target_ids
        target_ids.update(xlsx_ids)
        print(f"[INFO] XLSX 补充额外 ID 数量: {len(extra)} (总计 {len(target_ids)})")

    # 2. 加载已有序列
    existing = load_existing_sequences()
    have_bases: Set[str] = set()
    for k in existing:
        base = k.split("_", 1)[1] if "_" in k else k
        have_bases.add(base)

    missing = sorted(target_ids - have_bases)
    print(f"[INFO] 本地已有序列 (去重后): {len(have_bases)}")
    print(f"[INFO] 待下载缺失序列: {len(missing)}")

    if args.dry_run or not missing:
        if args.dry_run:
            print("[DRY-RUN] 未执行下载。")
        print("[INFO] 全部序列已存在，无需下载。" if not missing else "")
        return

    # 3. 批量下载
    print(f"\n[INFO] 开始批量下载 {len(missing)} 个序列 ...")
    fetched = fetch_missing_sequences(missing)
    print(f"[INFO] 成功下载: {len(fetched)} / {len(missing)}")

    # 4. 合并保存
    for acc, info in fetched.items():
        existing[acc] = info

    SEQUENCES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SEQUENCES_FILE, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
    print(f"[OK] 更新后序列库已保存: {SEQUENCES_FILE} (共 {len(existing)} 条)")

    # 5. 保存日志
    log = {
        "target_total": len(target_ids),
        "already_have": len(have_bases),
        "missing_requested": len(missing),
        "downloaded_success": len(fetched),
        "downloaded_failed": sorted(set(missing) - set(fetched.keys())),
        "downloaded_ids": sorted(fetched.keys()),
    }
    with open(OUTPUT_LOG, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    print(f"[OK] 下载日志: {OUTPUT_LOG}")

    print("\n" + "=" * 70)
    print("  下一步建议")
    print("=" * 70)
    print("  运行: python fetch_gpcrdb_couplings.py --keep-all-gpcrs")
    print("  运行: python build_paired_matrix.py")
    print("  运行: python paired_cross_validation.py")


if __name__ == "__main__":
    main()
