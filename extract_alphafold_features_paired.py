#!/usr/bin/env python3
"""
AlphaFold 结构特征提取 (Strategy C Phase 1 Week 2)

基于现有 structure_features.py 改造:
- 输入: merged_dataset/extended_sequences.json
- 输出: paired_dataset/alphafold_structure_features.json
- 额外保存 per-residue pLDDT，供后续 TMHMM 拓扑解析 ICL 平均 pLDDT
"""

import json
import requests
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from Bio import PDB
from Bio.PDB.DSSP import DSSP
from Bio.PDB.SASA import ShrakeRupley
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
SEQUENCES_FILE = BASE / "merged_dataset" / "extended_sequences.json"
OUTPUT_DIR = BASE / "paired_dataset"
OUTPUT_DIR.mkdir(exist_ok=True)
STRUCT_DIR = OUTPUT_DIR / "alphafold_pdbs"
STRUCT_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "alphafold_structure_features.json"


class AlphaFoldDownloader:
    """AlphaFold PDB 下载器"""
    BASE_URL = "https://alphafold.ebi.ac.uk/files"

    def __init__(self):
        self.session = requests.Session()

    def download_pdb(self, uniprot_id: str) -> Optional[Path]:
        # AlphaFold DB 当前主要提供 v6，旧版可能已被移除；尝试 v6→v1 回退
        for version in [6, 5, 4, 3, 2, 1]:
            pdb_filename = f"AF-{uniprot_id}-F1-model_v{version}.pdb"
            url = f"{self.BASE_URL}/{pdb_filename}"
            local_path = STRUCT_DIR / pdb_filename
            if local_path.exists():
                return local_path
            try:
                response = self.session.get(url, timeout=60)
                if response.status_code == 200:
                    local_path.write_bytes(response.content)
                    return local_path
            except Exception as e:
                print(f"  [WARN] {uniprot_id} v{version} 下载异常: {e}")
                continue
        print(f"  [WARN] {uniprot_id} 所有版本下载失败")
        return None


class StructureFeatureExtractor:
    def __init__(self):
        self.parser = PDB.PDBParser(QUIET=True)

    def load_structure(self, pdb_path: Path) -> Optional[PDB.Structure]:
        try:
            return self.parser.get_structure("protein", pdb_path)
        except Exception as e:
            print(f"  [WARN] 加载PDB失败: {e}")
            return None

    def extract_plddt_scores(self, pdb_path: Path) -> np.ndarray:
        structure = self.load_structure(pdb_path)
        if structure is None:
            return np.array([])
        scores = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if "CA" in residue:
                        scores.append(residue["CA"].get_bfactor())
        return np.array(scores)

    def extract_secondary_structure(self, pdb_path: Path) -> Dict[str, float]:
        structure = self.load_structure(pdb_path)
        if structure is None:
            return {}
        try:
            model = structure[0]
            dssp = DSSP(model, str(pdb_path), dssp="mkdssp")
            ss_counts = {"H": 0, "B": 0, "E": 0, "G": 0, "I": 0, "T": 0, "S": 0, "-": 0}
            total = 0
            for key in dssp.keys():
                ss = dssp[key][2]
                if ss in ss_counts:
                    ss_counts[ss] += 1
                    total += 1
            if total == 0:
                return {}
            return {
                "helix_ratio": (ss_counts["H"] + ss_counts["G"] + ss_counts["I"]) / total,
                "sheet_ratio": (ss_counts["E"] + ss_counts["B"]) / total,
                "turn_ratio": ss_counts["T"] / total,
                "coil_ratio": (ss_counts["S"] + ss_counts["-"]) / total,
            }
        except Exception as e:
            print(f"  [WARN] DSSP分析失败: {e}")
            return {}

    def calculate_sasa(self, pdb_path: Path) -> Dict[str, float]:
        structure = self.load_structure(pdb_path)
        if structure is None:
            return {}
        try:
            sr = ShrakeRupley()
            sr.compute(structure[0], level="R")
            sasa_values = []
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if hasattr(residue, "sasa"):
                            sasa_values.append(residue.sasa)
            if not sasa_values:
                return {}
            return {
                "mean_sasa": float(np.mean(sasa_values)),
                "std_sasa": float(np.std(sasa_values)),
                "buried_ratio": float(sum(1 for s in sasa_values if s < 20) / len(sasa_values)),
            }
        except Exception as e:
            print(f"  [WARN] SASA计算失败: {e}")
            return {}

    def calculate_contact_features(self, pdb_path: Path, threshold: float = 8.0) -> Dict[str, float]:
        structure = self.load_structure(pdb_path)
        if structure is None:
            return {}
        ca_coords = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if "CA" in residue:
                        ca_coords.append(residue["CA"].get_coord())
        if not ca_coords:
            return {}
        ca_coords = np.array(ca_coords)
        n = len(ca_coords)
        contacts = 0
        for i in range(n):
            for j in range(i + 1, n):
                if np.linalg.norm(ca_coords[i] - ca_coords[j]) < threshold:
                    contacts += 1
        total_possible = n * (n - 1) / 2
        density = contacts / total_possible if total_possible > 0 else 0.0
        # mean contacts per residue (undirected, exclude self)
        mean_contacts = (contacts * 2) / n if n > 0 else 0.0
        return {
            "contact_density": float(density),
            "mean_contacts_per_residue": float(mean_contacts),
        }

    def extract_all_features(self, pdb_path: Path) -> Dict:
        features = {}
        plddt_scores = self.extract_plddt_scores(pdb_path)
        if len(plddt_scores) > 0:
            features["plddt"] = {
                "mean": float(np.mean(plddt_scores)),
                "std": float(np.std(plddt_scores)),
                "high_confidence_ratio_70": float(np.sum(plddt_scores > 70) / len(plddt_scores)),
                "high_confidence_ratio_90": float(np.sum(plddt_scores > 90) / len(plddt_scores)),
                "per_residue": plddt_scores.tolist(),
            }
        ss = self.extract_secondary_structure(pdb_path)
        if ss:
            features["secondary_structure"] = ss
        sasa = self.calculate_sasa(pdb_path)
        if sasa:
            features["sasa"] = sasa
        contacts = self.calculate_contact_features(pdb_path)
        if contacts:
            features["contact_map"] = contacts
        return features


def main():
    print("=" * 70)
    print("  AlphaFold 结构特征提取 (paired_dataset)")
    print("=" * 70)

    with open(SEQUENCES_FILE) as f:
        sequences = json.load(f)

    # 去重: 只保留无 prefix 的原始 ID (与 build_paired_matrix.py 一致)
    prefixed = set()
    for k in sequences:
        if "_" in k and len(k.split("_")[0]) <= 2:
            base = k.split("_", 1)[1]
            if base in sequences:
                prefixed.add(k)
    uniprot_ids = [k for k in sequences if k not in prefixed]
    print(f"[INFO] 待处理独立 GPCR 数: {len(uniprot_ids)}")

    downloader = AlphaFoldDownloader()
    extractor = StructureFeatureExtractor()

    results = {}
    failed = []

    for uid in tqdm(uniprot_ids, desc="Downloading & extracting"):
        pdb_path = downloader.download_pdb(uid)
        if pdb_path is None:
            failed.append(uid)
            continue
        feats = extractor.extract_all_features(pdb_path)
        if feats:
            results[uid] = feats
        else:
            failed.append(uid)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] 成功提取: {len(results)} / {len(uniprot_ids)}")
    print(f"    失败: {len(failed)}")
    if failed:
        print(f"    失败列表: {failed}")
    print(f"    结果保存: {OUTPUT_FILE}")

    # 检查 DSSP 可用性提示
    print("\n[NOTE] DSSP 需要本地安装 mkdssp。如大量样本缺少 secondary_structure，")
    print("       请安装 DSSP (https://github.com/PDB-REDO/dssp) 并加入 PATH。")


if __name__ == "__main__":
    main()
