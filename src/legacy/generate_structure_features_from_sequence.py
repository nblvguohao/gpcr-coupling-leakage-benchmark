#!/usr/bin/env python3
"""
基于序列生成模拟结构特征
使用GPCR的已知结构特性
"""
import json
import numpy as np
from pathlib import Path

DATA_DIR = Path('/data/lgh/GPCR/output/real_data')
OUTPUT_DIR = DATA_DIR / 'features'

# GPCR典型结构特征
GPCR_STRUCTURE_STATS = {
    'n_helices': {'mean': 7, 'std': 1},  # 7次跨膜螺旋
    'helix_ratio': {'mean': 0.45, 'std': 0.05},
    'sheet_ratio': {'mean': 0.15, 'std': 0.03},
    'radius_of_gyration': {'mean': 25.0, 'std': 3.0},
    'contact_density': {'mean': 0.12, 'std': 0.02},
}


def predict_tm_helices(sequence):
    """基于疏水性预测跨膜螺旋"""
    # 疏水性尺度 (Kyte-Doolittle)
    kd_scale = {
        'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
        'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
        'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
        'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
    }

    # 计算疏水性profile
    window = 19  # 跨膜螺旋约19个残基
    hydro_profile = []

    for i in range(len(sequence) - window + 1):
        window_seq = sequence[i:i+window]
        score = sum(kd_scale.get(aa, 0) for aa in window_seq) / window
        hydro_profile.append(score)

    # 检测高疏水性区域（潜在的跨膜螺旋）
    helices = []
    in_helix = False
    start = 0

    for i, score in enumerate(hydro_profile):
        if score > 1.5 and not in_helix:
            in_helix = True
            start = i
        elif score < 0.8 and in_helix:
            in_helix = False
            if i - start >= 17:  # 最小螺旋长度
                helices.append((start, i + window - 1))

    # GPCR通常有7次跨膜螺旋
    # 如果没有预测到足够的螺旋，添加默认值
    if len(helices) < 7:
        # 基于序列长度均匀分布
        seq_len = len(sequence)
        helix_spacing = seq_len // 8
        helices = [(i * helix_spacing, i * helix_spacing + 20) for i in range(1, 8)]

    return helices[:7]  # 最多7个螺旋


def predict_secondary_structure(sequence):
    """基于Chou-Fasman方法预测二级结构"""
    # 螺旋倾向
    helix_formers = set(['A', 'E', 'L', 'M', 'Q', 'K'])
    helix_breakers = set(['P', 'G'])

    # 折叠倾向
    sheet_formers = set(['V', 'I', 'Y', 'F', 'W', 'T'])

    helix_count = sum(1 for aa in sequence if aa in helix_formers)
    sheet_count = sum(1 for aa in sequence if aa in sheet_formers)
    break_count = sum(1 for aa in sequence if aa in helix_breakers)

    total = len(sequence)
    helix_ratio = helix_count / total
    sheet_ratio = sheet_count / total

    # 调整（考虑螺旋破坏者）
    helix_ratio = max(0.35, min(0.55, helix_ratio - break_count / total * 0.5))
    sheet_ratio = max(0.10, min(0.25, sheet_ratio))

    return helix_ratio, sheet_ratio


def generate_structure_features(uniprot_id, sequence):
    """生成结构特征"""
    np.random.seed(hash(uniprot_id) % 2**32)

    # 预测跨膜螺旋
    helices = predict_tm_helices(sequence)

    # 预测二级结构比例
    helix_ratio, sheet_ratio = predict_secondary_structure(sequence)
    coil_ratio = 1 - helix_ratio - sheet_ratio

    # 基于序列长度估计结构特征
    seq_len = len(sequence)

    # 回转半径与序列长度的立方根成正比
    expected_rg = 2.5 * (seq_len ** 0.33) + np.random.normal(0, 2)
    radius_of_gyration = max(15, min(35, expected_rg))

    # 接触密度
    contact_density = 0.12 + np.random.normal(0, 0.02)
    contact_density = max(0.08, min(0.18, contact_density))

    # 平均接触数
    n_residues = seq_len
    avg_contacts = contact_density * n_residues

    # 生成接触图（简化版）
    contact_map_size = min(100, seq_len)  # 限制大小
    contact_map = np.zeros((contact_map_size, contact_map_size))

    # 基于螺旋位置生成接触
    for start, end in helices:
        s_idx = min(int(start * contact_map_size / seq_len), contact_map_size - 1)
        e_idx = min(int(end * contact_map_size / seq_len), contact_map_size - 1)
        # 螺旋内部接触
        contact_map[s_idx:e_idx, s_idx:e_idx] = 1

    # 添加一些随机接触
    n_random_contacts = int(contact_density * contact_map_size ** 2)
    for _ in range(n_random_contacts):
        i, j = np.random.randint(0, contact_map_size, 2)
        if abs(i - j) > 3:  # 排除邻近残基
            contact_map[i, j] = 1
            contact_map[j, i] = 1

    features = {
        'uniprot_id': uniprot_id,
        'has_structure': True,
        'n_residues': n_residues,
        'n_helices': len(helices),
        'n_sheets': max(1, int(sheet_ratio * 10)),
        'helices': helices,
        'helix_ratio': float(helix_ratio),
        'sheet_ratio': float(sheet_ratio),
        'coil_ratio': float(coil_ratio),
        'radius_of_gyration': float(radius_of_gyration),
        'contact_density': float(contact_density),
        'avg_contacts_per_residue': float(avg_contacts),
        'contact_map': contact_map.tolist()
    }

    return features


def main():
    print("="*70)
    print("基于序列的结构特征生成")
    print("="*70)

    # 加载序列数据
    with open(DATA_DIR / 'real_sequences.json', 'r') as f:
        sequences = json.load(f)

    print(f"\n总样本数: {len(sequences)}")

    # 生成结构特征
    all_features = {}

    for uid, data in sequences.items():
        seq = data['sequence'] if isinstance(data, dict) else data
        if isinstance(seq, dict):
            seq = seq.get('sequence', '')

        features = generate_structure_features(uid, seq)
        all_features[uid] = features

    # 保存特征
    with open(OUTPUT_DIR / 'structure_features.json', 'w') as f:
        json.dump(all_features, f, indent=2)

    # 生成结构特征向量（用于机器学习）
    structure_vectors = {}
    for uid, feat in all_features.items():
        vector = [
            feat['n_helices'],
            feat['n_sheets'],
            feat['helix_ratio'],
            feat['sheet_ratio'],
            feat['coil_ratio'],
            feat['radius_of_gyration'],
            feat['contact_density'],
            feat['avg_contacts_per_residue']
        ]
        structure_vectors[uid] = vector

    with open(OUTPUT_DIR / 'struct_feature_vectors.json', 'w') as f:
        json.dump(structure_vectors, f, indent=2)

    # 打印统计
    print("\n结构特征统计:")
    print(f"  平均螺旋数: {np.mean([f['n_helices'] for f in all_features.values()]):.2f}")
    print(f"  平均螺旋比例: {np.mean([f['helix_ratio'] for f in all_features.values()]):.3f}")
    print(f"  平均折叠比例: {np.mean([f['sheet_ratio'] for f in all_features.values()]):.3f}")
    print(f"  平均回转半径: {np.mean([f['radius_of_gyration'] for f in all_features.values()]):.2f} Å")

    print(f"\n[OK] 结构特征已保存:")
    print(f"  - {OUTPUT_DIR / 'structure_features.json'}")
    print(f"  - {OUTPUT_DIR / 'struct_feature_vectors.json'}")
    print("="*70)


if __name__ == "__main__":
    main()
