#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实AlphaFold结构特征提取
从AlphaFold DB下载PDB文件并提取结构特征
"""

import os
import json
import gzip
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from Bio import PDB
from Bio.PDB.DSSP import DSSP
from Bio.PDB.SASA import ShrakeRupley
from tqdm import tqdm

# 设置路径
DATA_DIR = Path("/mnt/okcomputer/output/real_data")
STRUCT_DIR = DATA_DIR / "structures"
STRUCT_DIR.mkdir(exist_ok=True)
STRUCT_FEATURES_DIR = DATA_DIR / "structure_features"
STRUCT_FEATURES_DIR.mkdir(exist_ok=True)

print("="*80)
print("🧬 真实AlphaFold结构特征提取")
print("="*80)

# ==============================================================================
# 第一部分：AlphaFold PDB下载
# ==============================================================================

class AlphaFoldDownloader:
    """AlphaFold PDB文件下载器"""
    
    BASE_URL = "https://alphafold.ebi.ac.uk/files"
    
    def __init__(self):
        self.session = requests.Session()
    
    def download_pdb(self, uniprot_id: str) -> Optional[Path]:
        """
        下载AlphaFold预测的PDB文件
        
        Args:
            uniprot_id: UniProt ID
            
        Returns:
            PDB文件路径或None
        """
        # AlphaFold PDB文件名格式: AF-{uniprot_id}-F1-model_v4.pdb
        pdb_filename = f"AF-{uniprot_id}-F1-model_v4.pdb"
        url = f"{self.BASE_URL}/{pdb_filename}"
        
        local_path = STRUCT_DIR / pdb_filename
        
        # 如果已存在，直接返回
        if local_path.exists():
            return local_path
        
        try:
            print(f"  下载 {uniprot_id}...", end=' ')
            response = self.session.get(url, timeout=60)
            
            if response.status_code == 200:
                with open(local_path, 'wb') as f:
                    f.write(response.content)
                print(f"✅")
                return local_path
            else:
                print(f"❌ (HTTP {response.status_code})")
                return None
                
        except Exception as e:
            print(f"❌ ({e})")
            return None
    
    def download_pae(self, uniprot_id: str) -> Optional[Path]:
        """下载预测对齐误差(PAE)文件"""
        pae_filename = f"AF-{uniprot_id}-F1-predicted_aligned_error_v4.json"
        url = f"{self.BASE_URL}/{pae_filename}"
        
        local_path = STRUCT_DIR / pae_filename
        
        if local_path.exists():
            return local_path
        
        try:
            response = self.session.get(url, timeout=60)
            if response.status_code == 200:
                with open(local_path, 'wb') as f:
                    f.write(response.content)
                return local_path
            return None
        except:
            return None


# ==============================================================================
# 第二部分：结构特征提取
# ==============================================================================

class StructureFeatureExtractor:
    """蛋白质结构特征提取器"""
    
    def __init__(self):
        self.parser = PDB.PDBParser(QUIET=True)
    
    def load_structure(self, pdb_path: Path) -> Optional[PDB.Structure]:
        """加载PDB结构"""
        try:
            structure = self.parser.get_structure("protein", pdb_path)
            return structure
        except Exception as e:
            print(f"  ⚠️ 加载PDB失败: {e}")
            return None
    
    def extract_plddt_scores(self, pdb_path: Path) -> np.ndarray:
        """
        从PDB文件的B-factor列提取pLDDT分数
        
        pLDDT范围: 0-100
        - >90: 高置信度（非常准确）
        - 70-90: 良好置信度
        - 50-70: 低置信度
        - <50: 非常低的置信度
        """
        structure = self.load_structure(pdb_path)
        if structure is None:
            return np.array([])
        
        plddt_scores = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if 'CA' in residue:  # 只考虑有CA原子的残基
                        plddt = residue['CA'].get_bfactor()
                        plddt_scores.append(plddt)
        
        return np.array(plddt_scores)
    
    def extract_secondary_structure(self, pdb_path: Path) -> Dict[str, float]:
        """
        提取二级结构比例
        
        使用DSSP算法（需要安装mkdssp）
        """
        structure = self.load_structure(pdb_path)
        if structure is None:
            return {}
        
        try:
            # 创建DSSP对象
            model = structure[0]
            dssp = DSSP(model, str(pdb_path), dssp='mkdssp')
            
            # 统计二级结构
            ss_counts = {'H': 0, 'B': 0, 'E': 0, 'G': 0, 'I': 0, 'T': 0, 'S': 0, '-': 0}
            total = 0
            
            for key in dssp.keys():
                ss = dssp[key][2]
                if ss in ss_counts:
                    ss_counts[ss] += 1
                    total += 1
            
            if total == 0:
                return {}
            
            # 计算比例
            return {
                'helix': (ss_counts['H'] + ss_counts['G'] + ss_counts['I']) / total,  # α-螺旋 + 3-10螺旋 + π-螺旋
                'sheet': (ss_counts['E'] + ss_counts['B']) / total,  # β-折叠 + β-桥
                'turn': ss_counts['T'] / total,  # 转角
                'coil': (ss_counts['S'] + ss_counts['-']) / total  # 无规卷曲
            }
            
        except Exception as e:
            print(f"  ⚠️ DSSP分析失败: {e}")
            return {}
    
    def calculate_sasa(self, pdb_path: Path) -> Dict[str, float]:
        """
        计算溶剂可及表面积(SASA)
        """
        structure = self.load_structure(pdb_path)
        if structure is None:
            return {}
        
        try:
            sr = ShrakeRupley()
            sr.compute(structure[0], level="R")  # 残基级别
            
            sasa_values = []
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if hasattr(residue, 'sasa'):
                            sasa_values.append(residue.sasa)
            
            if len(sasa_values) == 0:
                return {}
            
            return {
                'mean_sasa': np.mean(sasa_values),
                'std_sasa': np.std(sasa_values),
                'total_sasa': np.sum(sasa_values),
                'buried_residues': sum(1 for s in sasa_values if s < 20) / len(sasa_values)
            }
            
        except Exception as e:
            print(f"  ⚠️ SASA计算失败: {e}")
            return {}
    
    def calculate_contact_map(self, pdb_path: Path, threshold: float = 8.0) -> np.ndarray:
        """
        计算残基接触图
        
        Args:
            pdb_path: PDB文件路径
            threshold: 接触距离阈值（Å）
            
        Returns:
            接触图矩阵
        """
        structure = self.load_structure(pdb_path)
        if structure is None:
            return np.array([])
        
        # 提取CA原子坐标
        ca_coords = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if 'CA' in residue:
                        ca_coords.append(residue['CA'].get_coord())
        
        if len(ca_coords) == 0:
            return np.array([])
        
        ca_coords = np.array(ca_coords)
        
        # 计算距离矩阵
        n = len(ca_coords)
        contact_map = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(ca_coords[i] - ca_coords[j])
                if dist < threshold:
                    contact_map[i, j] = 1
                    contact_map[j, i] = 1
        
        return contact_map
    
    def extract_all_features(self, pdb_path: Path) -> Dict:
        """提取所有结构特征"""
        features = {}
        
        # 1. pLDDT分数
        plddt_scores = self.extract_plddt_scores(pdb_path)
        if len(plddt_scores) > 0:
            features['plddt'] = {
                'mean': float(np.mean(plddt_scores)),
                'std': float(np.std(plddt_scores)),
                'min': float(np.min(plddt_scores)),
                'max': float(np.max(plddt_scores)),
                'high_confidence_ratio': float(np.sum(plddt_scores > 70) / len(plddt_scores))
            }
        
        # 2. 二级结构
        ss_features = self.extract_secondary_structure(pdb_path)
        features['secondary_structure'] = ss_features
        
        # 3. SASA
        sasa_features = self.calculate_sasa(pdb_path)
        features['sasa'] = sasa_features
        
        # 4. 接触图特征
        contact_map = self.calculate_contact_map(pdb_path)
        if contact_map.size > 0:
            features['contact_map'] = {
                'density': float(np.sum(contact_map) / (contact_map.shape[0] * contact_map.shape[1])),
                'mean_contacts_per_residue': float(np.sum(contact_map, axis=1).mean())
            }
        
        return features


# ==============================================================================
# 第三部分：主程序
# ==============================================================================

def main():
    """主程序"""
    
    print("\n" + "="*80)
    print("🚀 AlphaFold结构特征提取流程")
    print("="*80)
    
    # 1. 加载序列数据
    print("\n1. 加载序列数据...")
    with open(DATA_DIR / "real_sequences.json", 'r') as f:
        sequences_data = json.load(f)
    
    uniprot_ids = list(sequences_data.keys())
    print(f"   需要处理 {len(uniprot_ids)} 个蛋白质")
    
    # 2. 初始化下载器和提取器
    print("\n2. 初始化下载器和特征提取器...")
    downloader = AlphaFoldDownloader()
    extractor = StructureFeatureExtractor()
    
    # 3. 下载PDB文件并提取特征
    print("\n3. 下载PDB文件并提取结构特征...")
    
    structure_features = {}
    failed_downloads = []
    
    for uniprot_id in tqdm(uniprot_ids, desc="处理蛋白质"):
        # 下载PDB
        pdb_path = downloader.download_pdb(uniprot_id)
        
        if pdb_path is None:
            failed_downloads.append(uniprot_id)
            continue
        
        # 提取特征
        features = extractor.extract_all_features(pdb_path)
        
        if features:
            structure_features[uniprot_id] = features
    
    print(f"\n   ✅ 成功提取 {len(structure_features)} 个蛋白质的结构特征")
    if failed_downloads:
        print(f"   ⚠️ 失败: {len(failed_downloads)} 个")
    
    # 4. 保存特征
    print("\n4. 保存结构特征...")
    
    with open(STRUCT_FEATURES_DIR / "structure_features.json", 'w') as f:
        json.dump(structure_features, f, indent=2)
    
    # 创建特征矩阵
    feature_list = []
    feature_ids = []
    
    for uniprot_id, features in structure_features.items():
        feat_vector = []
        
        # pLDDT特征
        if 'plddt' in features:
            feat_vector.extend([
                features['plddt'].get('mean', 0),
                features['plddt'].get('std', 0),
                features['plddt'].get('high_confidence_ratio', 0)
            ])
        else:
            feat_vector.extend([0, 0, 0])
        
        # 二级结构特征
        if 'secondary_structure' in features:
            feat_vector.extend([
                features['secondary_structure'].get('helix', 0),
                features['secondary_structure'].get('sheet', 0),
                features['secondary_structure'].get('turn', 0),
                features['secondary_structure'].get('coil', 0)
            ])
        else:
            feat_vector.extend([0, 0, 0, 0])
        
        # SASA特征
        if 'sasa' in features:
            feat_vector.extend([
                features['sasa'].get('mean_sasa', 0),
                features['sasa'].get('buried_residues', 0)
            ])
        else:
            feat_vector.extend([0, 0])
        
        # 接触图特征
        if 'contact_map' in features:
            feat_vector.extend([
                features['contact_map'].get('density', 0),
                features['contact_map'].get('mean_contacts_per_residue', 0)
            ])
        else:
            feat_vector.extend([0, 0])
        
        feature_list.append(feat_vector)
        feature_ids.append(uniprot_id)
    
    if feature_list:
        feature_matrix = np.array(feature_list)
        np.save(STRUCT_FEATURES_DIR / "structure_feature_matrix.npy", feature_matrix)
        
        with open(STRUCT_FEATURES_DIR / "structure_feature_ids.json", 'w') as f:
            json.dump(feature_ids, f, indent=2)
        
        print(f"   特征矩阵形状: {feature_matrix.shape}")
    
    print(f"\n   ✅ 结构特征保存完成!")
    print(f"   - 结构特征: {STRUCT_FEATURES_DIR / 'structure_features.json'}")
    print(f"   - 特征矩阵: {STRUCT_FEATURES_DIR / 'structure_feature_matrix.npy'}")
    
    # 统计信息
    print("\n" + "="*80)
    print("📊 结构特征统计")
    print("="*80)
    print(f"  成功处理的蛋白质: {len(structure_features)}")
    print(f"  结构特征维度: 11")
    
    # pLDDT统计
    plddt_means = [f['plddt']['mean'] for f in structure_features.values() if 'plddt' in f]
    if plddt_means:
        print(f"  平均pLDDT: {np.mean(plddt_means):.2f} ± {np.std(plddt_means):.2f}")
        print(f"  高质量结构 (>70): {sum(1 for p in plddt_means if p > 70)}/{len(plddt_means)}")
    
    print("\n" + "="*80)
    print("✅ AlphaFold结构特征提取完成!")
    print("="*80)
    
    return structure_features


if __name__ == "__main__":
    features = main()
