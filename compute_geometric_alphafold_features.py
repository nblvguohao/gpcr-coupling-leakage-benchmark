#!/usr/bin/env python3
"""
从 AlphaFold PDB 中提取 GPCR-G 蛋白结合界面的几何结构特征。

输入:
  - paired_dataset/alphafold_pdbs/*.pdb
  - paired_dataset/uniprot_topology.json

输出:
  - paired_dataset/alphafold_geometric_features.json
    { uniprot_id: {
        "tm5_tm6_cyto_ca_distance": float,
        "icl2_end_to_end_ca_distance": float,
        "icl3_end_to_end_ca_distance": float,
        "tm5_tm6_cyto_dihedral_angle": float,
        "icl2_aromatic_centroid_depth": float,
        "interface_patch_sasa": float,
        "interface_patch_sasa_ratio": float,
        "icl2_helix_ratio": float,
        "icl2_sheet_ratio": float,
        "icl2_coil_ratio": float,
        "icl3_helix_ratio": float,
        "icl3_sheet_ratio": float,
        "icl3_coil_ratio": float,
    }}
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pydssp
from Bio import PDB
from Bio.PDB.SASA import ShrakeRupley
from tqdm import tqdm

BASE = Path(__file__).parent
PDB_DIR = BASE / "paired_dataset" / "alphafold_pdbs"
TOPO_FILE = BASE / "paired_dataset" / "uniprot_topology.json"
OUTPUT_FILE = BASE / "paired_dataset" / "alphafold_geometric_features.json"

AROMATIC_RESIDUES = {"PHE", "TRP", "TYR"}


class GeometricFeatureExtractor:
    def __init__(self):
        self.parser = PDB.PDBParser(QUIET=True)

    def load_structure(self, pdb_path: Path) -> Optional[PDB.Structure]:
        try:
            return self.parser.get_structure("protein", str(pdb_path))
        except Exception as e:
            print(f"  [WARN] 加载 PDB 失败 {pdb_path.name}: {e}")
            return None

    @staticmethod
    def get_ca_coords(structure: PDB.Structure) -> np.ndarray:
        coords = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if "CA" in residue:
                        coords.append(residue["CA"].get_coord())
        return np.array(coords)

    @staticmethod
    def get_residue_dict(structure: PDB.Structure) -> Dict[int, PDB.Residue]:
        residues = {}
        idx = 1
        for model in structure:
            for chain in model:
                for residue in chain:
                    if "CA" in residue:
                        residues[idx] = residue
                        idx += 1
        return residues

    @staticmethod
    def ca_coord(residues: Dict[int, PDB.Residue], index: int) -> Optional[np.ndarray]:
        res = residues.get(index)
        if res is None or "CA" not in res:
            return None
        return res["CA"].get_coord()

    @staticmethod
    def compute_dihedral(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> float:
        b1 = p2 - p1
        b2 = p3 - p2
        b3 = p4 - p3

        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)

        norm_n1 = np.linalg.norm(n1)
        norm_n2 = np.linalg.norm(n2)
        norm_b2 = np.linalg.norm(b2)

        if norm_n1 == 0 or norm_n2 == 0 or norm_b2 == 0:
            return 0.0

        n1 = n1 / norm_n1
        n2 = n2 / norm_n2
        b2_unit = b2 / norm_b2

        m1 = np.cross(n1, b2_unit)
        x = np.dot(n1, n2)
        y = np.dot(m1, n2)
        return math.atan2(y, x)

    @staticmethod
    def fit_membrane_plane(tm_regions: List[Tuple[int, int]], residues: Dict[int, PDB.Residue]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        centroids = []
        for start, end in tm_regions:
            coords = []
            for i in range(start, end + 1):
                coord = GeometricFeatureExtractor.ca_coord(residues, i)
                if coord is not None:
                    coords.append(coord)
            if coords:
                centroids.append(np.mean(coords, axis=0))
        if len(centroids) < 3:
            return None, None
        centroids = np.array(centroids)
        center = np.mean(centroids, axis=0)
        centered = centroids - center
        _, _, vt = np.linalg.svd(centered)
        normal = vt[-1]
        normal = normal / np.linalg.norm(normal)
        return normal, center

    def extract_secondary_structure_for_range(
        self, pdb_path: Path, start: int, end: int
    ) -> Dict[str, float]:
        try:
            with open(pdb_path, encoding="utf-8") as f:
                coord = pydssp.read_pdbtext(f.read())
            ss = pydssp.assign(coord, out_type="c3")
            total = 0
            helix = 0
            sheet = 0
            coil = 0
            for i in range(start, end + 1):
                idx = i - 1
                if 0 <= idx < len(ss):
                    total += 1
                    if ss[idx] == "H":
                        helix += 1
                    elif ss[idx] == "E":
                        sheet += 1
                    else:
                        coil += 1
            if total == 0:
                return {"helix_ratio": 0.0, "sheet_ratio": 0.0, "coil_ratio": 0.0}
            return {
                "helix_ratio": float(helix / total),
                "sheet_ratio": float(sheet / total),
                "coil_ratio": float(coil / total),
            }
        except Exception:
            return {"helix_ratio": 0.0, "sheet_ratio": 0.0, "coil_ratio": 0.0}

    @staticmethod
    def compute_interface_patch_sasa(
        structure: PDB.Structure,
        tm5: Tuple[int, int],
        tm6: Tuple[int, int],
        icl2: Tuple[int, int],
        icl3: Tuple[int, int],
    ) -> Dict[str, float]:
        try:
            sr = ShrakeRupley()
            sr.compute(structure[0], level="R")
            total_sasa = 0.0
            interface_sasa = 0.0
            residue_idx = 1
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if not hasattr(residue, "sasa"):
                            continue
                        sasa = residue.sasa
                        total_sasa += sasa
                        in_interface = False
                        if icl2[0] <= residue_idx <= icl2[1]:
                            in_interface = True
                        if icl3[0] <= residue_idx <= icl3[1]:
                            in_interface = True
                        if tm5[0] <= residue_idx <= tm5[1]:
                            in_interface = True
                        if tm6[0] <= residue_idx <= tm6[1]:
                            in_interface = True
                        if in_interface:
                            interface_sasa += sasa
                        residue_idx += 1
            ratio = interface_sasa / total_sasa if total_sasa > 0 else 0.0
            return {
                "interface_patch_sasa": float(interface_sasa),
                "interface_patch_sasa_ratio": float(ratio),
            }
        except Exception as e:
            print(f"  [WARN] SASA 计算失败: {e}")
            return {"interface_patch_sasa": 0.0, "interface_patch_sasa_ratio": 0.0}

    def extract_geometric_features(
        self, pdb_path: Path, tm_regions: List[Tuple[int, int]], loops: Dict[str, Tuple[int, int]]
    ) -> Dict[str, float]:
        structure = self.load_structure(pdb_path)
        if structure is None:
            return {}

        residues = self.get_residue_dict(structure)
        if not residues:
            return {}

        rec: Dict[str, float] = {}

        # 1. TM5-TM6 cytoplasmic Cα distance
        if len(tm_regions) >= 6:
            tm5_end = tm_regions[4][1]
            tm6_start = tm_regions[5][0]
            ca_tm5_end = self.ca_coord(residues, tm5_end)
            ca_tm6_start = self.ca_coord(residues, tm6_start)
            if ca_tm5_end is not None and ca_tm6_start is not None:
                rec["tm5_tm6_cyto_ca_distance"] = float(np.linalg.norm(ca_tm5_end - ca_tm6_start))
            else:
                rec["tm5_tm6_cyto_ca_distance"] = 0.0
        else:
            rec["tm5_tm6_cyto_ca_distance"] = 0.0

        # 2 & 3. ICL2/ICL3 end-to-end Cα distances
        for loop_name, key in [("ICL2", "icl2"), ("ICL3", "icl3")]:
            region = loops.get(loop_name)
            if region:
                ca_start = self.ca_coord(residues, region[0])
                ca_end = self.ca_coord(residues, region[1])
                if ca_start is not None and ca_end is not None:
                    rec[f"{key}_end_to_end_ca_distance"] = float(np.linalg.norm(ca_start - ca_end))
                else:
                    rec[f"{key}_end_to_end_ca_distance"] = 0.0
            else:
                rec[f"{key}_end_to_end_ca_distance"] = 0.0

        # 4. TM5-TM6 cytoplasmic dihedral angle
        if len(tm_regions) >= 6:
            tm5_start = tm_regions[4][0]
            tm5_end = tm_regions[4][1]
            tm6_start = tm_regions[5][0]
            tm6_end = tm_regions[5][1]
            p1 = self.ca_coord(residues, tm5_start)
            p2 = self.ca_coord(residues, tm5_end)
            p3 = self.ca_coord(residues, tm6_start)
            p4 = self.ca_coord(residues, tm6_end)
            if all(p is not None for p in (p1, p2, p3, p4)):
                rec["tm5_tm6_cyto_dihedral_angle"] = float(self.compute_dihedral(p1, p2, p3, p4))
            else:
                rec["tm5_tm6_cyto_dihedral_angle"] = 0.0
        else:
            rec["tm5_tm6_cyto_dihedral_angle"] = 0.0

        # 5. ICL2 aromatic centroid depth relative to membrane plane
        plane_normal, plane_center = self.fit_membrane_plane(tm_regions, residues)
        icl2_region = loops.get("ICL2")
        if plane_normal is not None and icl2_region is not None:
            aromatic_coords = []
            for i in range(icl2_region[0], icl2_region[1] + 1):
                res = residues.get(i)
                if res is None:
                    continue
                if res.get_resname() in AROMATIC_RESIDUES and "CA" in res:
                    aromatic_coords.append(res["CA"].get_coord())
            if aromatic_coords:
                aromatic_centroid = np.mean(aromatic_coords, axis=0)
                depth = np.dot(aromatic_centroid - plane_center, plane_normal)
                rec["icl2_aromatic_centroid_depth"] = float(depth)
            else:
                rec["icl2_aromatic_centroid_depth"] = 0.0
        else:
            rec["icl2_aromatic_centroid_depth"] = 0.0

        # 6 & 7. Interface patch SASA
        tm5c = (tm_regions[4][1] - 3, tm_regions[4][1]) if len(tm_regions) >= 5 else (0, 0)
        tm6c = (tm_regions[5][0], tm_regions[5][0] + 3) if len(tm_regions) >= 6 else (0, 0)
        sasa_dict = self.compute_interface_patch_sasa(structure, tm5c, tm6c, icl2_region or (0, 0), loops.get("ICL3") or (0, 0))
        rec.update(sasa_dict)

        # 8-13. ICL2/ICL3 secondary structure ratios
        for loop_name, key in [("ICL2", "icl2"), ("ICL3", "icl3")]:
            region = loops.get(loop_name)
            if region:
                ss = self.extract_secondary_structure_for_range(pdb_path, region[0], region[1])
                rec[f"{key}_helix_ratio"] = ss["helix_ratio"]
                rec[f"{key}_sheet_ratio"] = ss["sheet_ratio"]
                rec[f"{key}_coil_ratio"] = ss["coil_ratio"]
            else:
                rec[f"{key}_helix_ratio"] = 0.0
                rec[f"{key}_sheet_ratio"] = 0.0
                rec[f"{key}_coil_ratio"] = 0.0

        return rec


def parse_uniprot_from_filename(filename: str) -> Optional[str]:
    """Parse UniProt ID from AlphaFold PDB filename like AF-O00144-F1-model_v6.pdb"""
    if filename.startswith("AF-") and filename.endswith(".pdb"):
        parts = filename.split("-")
        if len(parts) >= 2:
            return parts[1]
    return None


def main():
    print("=" * 70)
    print("  AlphaFold 几何结构特征提取")
    print("=" * 70)

    if not TOPO_FILE.exists():
        print(f"[ERR] 拓扑文件不存在: {TOPO_FILE}")
        return

    with open(TOPO_FILE, encoding="utf-8") as f:
        topo_data = json.load(f)

    pdb_files = sorted(PDB_DIR.glob("*.pdb"))
    print(f"[INFO] 发现 PDB 文件: {len(pdb_files)}")
    print(f"[INFO] 拓扑注释条目: {len(topo_data)}")

    extractor = GeometricFeatureExtractor()
    results = {}
    failed = []

    for pdb_path in tqdm(pdb_files, desc="提取几何特征"):
        uid = parse_uniprot_from_filename(pdb_path.name)
        if uid is None:
            continue

        topo = topo_data.get(uid)
        if topo is None:
            continue

        tm_regions = topo.get("tm_regions", [])
        loops = topo.get("loops", {})

        if len(tm_regions) < 7:
            failed.append((uid, "tm_regions < 7"))
            continue

        feats = extractor.extract_geometric_features(pdb_path, tm_regions, loops)
        if feats:
            results[uid] = feats
        else:
            failed.append((uid, "extraction_failed"))

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] 成功提取: {len(results)} / {len(pdb_files)}")
    print(f"    失败: {len(failed)}")
    if failed:
        print(f"    失败原因示例: {failed[:5]}")
    print(f"    输出: {OUTPUT_FILE}")

    # 快速统计
    if results:
        dists = [v["tm5_tm6_cyto_ca_distance"] for v in results.values() if v["tm5_tm6_cyto_ca_distance"] > 0]
        depths = [v["icl2_aromatic_centroid_depth"] for v in results.values() if v["icl2_aromatic_centroid_depth"] != 0]
        if dists:
            print(f"    TM5-TM6 距离范围: {min(dists):.2f} - {max(dists):.2f} A (n={len(dists)})")
        if depths:
            print(f"    ICL2 芳香族深度非零: {len(depths)} / {len(results)}")


if __name__ == "__main__":
    main()
