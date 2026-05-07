#!/usr/bin/env python3
"""
序列衍生结构特征提取 - 简化版
"""
import json
import numpy as np
from pathlib import Path
from collections import Counter

DATA_DIR = Path('/data/lgh/GPCR/extended_data')
FEATURE_DIR = Path('/data/lgh/GPCR/extended_data/features')
FEATURE_DIR.mkdir(exist_ok=True)

# 疏水性标度 (Kyte-Doolittle)
HYDRO_SCALE = {
    'I': 4.5, 'V': 4.2, 'L': 3.8, 'F': 2.8, 'C': 2.5,
    'M': 1.9, 'A': 1.8, 'G': -0.4, 'T': -0.7, 'S': -0.8,
    'W': -0.9, 'Y': -1.3, 'P': -1.6, 'H': -3.2, 'E': -3.5,
    'Q': -3.5, 'D': -3.5, 'N': -3.5, 'K': -3.9, 'R': -4.5
}

def extract_features(sequence):
    """提取结构特征"""
    seq = sequence.upper()
    L = len(seq)
    
    features = {}
    
    # 1. 疏水性特征
    hydro = [HYDRO_SCALE.get(aa, 0) for aa in seq]
    features['hydro_mean'] = float(np.mean(hydro))
    features['hydro_std'] = float(np.std(hydro))
    features['hydro_max'] = float(np.max(hydro))
    
    # 21残基窗口跨膜区预测
    tm_count = 0
    for i in range(L - 20):
        if np.mean(hydro[i:i+21]) > 1.5:
            tm_count += 1
    features['tm_regions'] = min(tm_count, 10)
    
    # 2. 电荷特征
    pos = sum(1 for aa in seq if aa in 'KR') / L
    neg = sum(1 for aa in seq if aa in 'DE') / L
    features['positive_frac'] = pos
    features['negative_frac'] = neg
    features['net_charge'] = pos - neg
    
    # 3. 组成特征
    aa_counts = Counter(seq)
    features['hydrophobic_frac'] = sum(aa_counts.get(aa, 0) for aa in 'AVLIMCWFY') / L
    features['polar_frac'] = sum(aa_counts.get(aa, 0) for aa in 'STNQ') / L
    features['aromatic_frac'] = sum(aa_counts.get(aa, 0) for aa in 'FWY') / L
    features['cys_frac'] = aa_counts.get('C', 0) / L
    
    # 4. 结构域特征
    features['n_term_hydro'] = float(np.mean([HYDRO_SCALE.get(aa, 0) for aa in seq[:30]]))
    features['c_term_hydro'] = float(np.mean([HYDRO_SCALE.get(aa, 0) for aa in seq[-30:]]))
    features['length'] = L
    
    # 5. 二级结构倾向
    helix_aa = set('ALMREQK')
    sheet_aa = set('VIFYWCST')
    features['helix_tendency'] = sum(1 for aa in seq if aa in helix_aa) / L
    features['sheet_tendency'] = sum(1 for aa in seq if aa in sheet_aa) / L
    
    return features

def main():
    print("=" * 60)
    print("结构特征提取 - 100样本")
    print("=" * 60)
    
    with open(DATA_DIR / 'extended_sequences.json', 'r') as f:
        sequences = json.load(f)
    
    print(f"[INFO] 处理 {len(sequences)} 个序列")
    
    features = {}
    for uid, data in sequences.items():
        seq = data['sequence'] if isinstance(data, dict) else data
        features[uid] = extract_features(seq)
    
    output_file = FEATURE_DIR / 'structure_features_100samples.json'
    with open(output_file, 'w') as f:
        json.dump(features, f)
    
    print(f"[OK] 特征保存: {output_file}")
    print(f"[OK] 特征维度: {len(features[list(features.keys())[0]])}")
    print("=" * 60)

if __name__ == "__main__":
    main()
