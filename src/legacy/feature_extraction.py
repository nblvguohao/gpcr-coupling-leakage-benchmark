#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实ESM-2特征提取
使用Facebook的ESM-2预训练模型
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

# 设置路径
DATA_DIR = Path("/mnt/okcomputer/output/real_data")
FEATURES_DIR = DATA_DIR / "features"
FEATURES_DIR.mkdir(exist_ok=True)

print("="*80)
print("🔬 真实ESM-2特征提取")
print("="*80)

# ==============================================================================
# 第一部分：安装和导入ESM
# ==============================================================================

try:
    import esm
    import fairseq
    print("✅ ESM库已安装")
except ImportError:
    print("⚠️ 正在安装ESM库...")
    os.system("pip install fair-esm -q")
    import esm
    print("✅ ESM库安装完成")

# ==============================================================================
# 第二部分：ESM-2特征提取类
# ==============================================================================

class ESM2FeatureExtractor:
    """
    ESM-2特征提取器
    使用Facebook的ESM-2预训练模型提取蛋白质序列特征
    
    参考: https://github.com/facebookresearch/esm
    """
    
    def __init__(self, model_name: str = "esm2_t6_8M_UR50D"):
        """
        初始化ESM-2模型
        
        Args:
            model_name: ESM-2模型名称
                - esm2_t6_8M_UR50D: 小模型，快速 (推荐用于测试)
                - esm2_t12_35M_UR50D: 中等模型
                - esm2_t30_150M_UR50D: 大模型
                - esm2_t33_650M_UR50D: 完整模型，最准确但慢
        """
        print(f"\n🔄 加载ESM-2模型: {model_name}")
        
        # 加载模型和字母表
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        self.model.eval()  # 设置为评估模式
        
        # 获取模型参数
        self.repr_layer = self.model.num_layers  # 使用最后一层
        self.embedding_dim = self.model.embed_dim
        
        print(f"  ✅ 模型加载完成")
        print(f"     层数: {self.model.num_layers}")
        print(f"     嵌入维度: {self.embedding_dim}")
        
        # 使用GPU（如果可用）
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        print(f"     使用设备: {self.device}")
    
    def extract_features(self, sequence: str) -> np.ndarray:
        """
        提取单个蛋白质序列的ESM-2特征
        
        Args:
            sequence: 蛋白质序列（氨基酸字母）
            
        Returns:
            序列级特征向量 (embedding_dim,)
        """
        # 准备数据
        data = [("protein", sequence)]
        
        # 编码序列
        batch_converter = self.alphabet.get_batch_converter()
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)
        
        # 提取特征
        with torch.no_grad():
            results = self.model(
                batch_tokens, 
                repr_layers=[self.repr_layer], 
                return_contacts=False
            )
        
        # 获取token表示
        token_representations = results["representations"][self.repr_layer]
        
        # 平均池化获取序列级特征（跳过<cls>和<eos>标记）
        sequence_representation = token_representations[0, 1:-1].mean(0)
        
        return sequence_representation.cpu().numpy()
    
    def extract_features_batch(self, sequences: Dict[str, str]) -> Dict[str, np.ndarray]:
        """
        批量提取特征
        
        Args:
            sequences: {uniprot_id: sequence}
            
        Returns:
            {uniprot_id: feature_vector}
        """
        features = {}
        
        print(f"\n🔍 提取 {len(sequences)} 个序列的ESM-2特征...")
        
        for uniprot_id, sequence in tqdm(sequences.items(), desc="ESM-2特征提取"):
            try:
                feat = self.extract_features(sequence)
                features[uniprot_id] = feat
            except Exception as e:
                print(f"  ⚠️ 提取 {uniprot_id} 失败: {e}")
                continue
        
        print(f"  ✅ 成功提取 {len(features)} 个序列的特征")
        return features
    
    def extract_residue_level_features(self, sequence: str) -> np.ndarray:
        """
        提取残基级特征（用于注意力分析）
        
        Args:
            sequence: 蛋白质序列
            
        Returns:
            残基级特征 (seq_length, embedding_dim)
        """
        data = [("protein", sequence)]
        batch_converter = self.alphabet.get_batch_converter()
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)
        
        with torch.no_grad():
            results = self.model(
                batch_tokens, 
                repr_layers=[self.repr_layer], 
                return_contacts=False
            )
        
        token_representations = results["representations"][self.repr_layer]
        
        # 返回残基级特征（跳过<cls>和<eos>）
        return token_representations[0, 1:-1].cpu().numpy()


# ==============================================================================
# 第三部分：物理化学特征提取
# ==============================================================================

def extract_physicochemical_features(sequence: str) -> np.ndarray:
    """
    提取物理化学特征
    
    特征包括:
    - 氨基酸组成 (20维)
    - 物理化学性质 (9维)
    
    Args:
        sequence: 蛋白质序列
        
    Returns:
        29维特征向量
    """
    # 氨基酸字母
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    
    # 1. 氨基酸组成 (20维)
    aa_composition = np.array([sequence.count(aa) / len(sequence) for aa in amino_acids])
    
    # 2. 物理化学性质
    # 疏水性氨基酸 (A, I, L, M, F, W, V)
    hydrophobic = sum(sequence.count(aa) for aa in 'AILMFWV') / len(sequence)
    
    # 亲水性氨基酸 (D, E, K, R, N, Q, S, T)
    hydrophilic = sum(sequence.count(aa) for aa in 'DEKNQRST') / len(sequence)
    
    # 芳香族氨基酸 (F, W, Y)
    aromatic = sum(sequence.count(aa) for aa in 'FWY') / len(sequence)
    
    # 正电荷氨基酸 (K, R, H)
    positive = sum(sequence.count(aa) for aa in 'KRH') / len(sequence)
    
    # 负电荷氨基酸 (D, E)
    negative = sum(sequence.count(aa) for aa in 'DE') / len(sequence)
    
    # 净电荷
    net_charge = positive - negative
    
    # 极性氨基酸 (N, Q, S, T, C, Y)
    polar = sum(sequence.count(aa) for aa in 'NQSTCY') / len(sequence)
    
    # 非极性氨基酸 (A, G, I, L, M, F, P, W, V)
    nonpolar = sum(sequence.count(aa) for aa in 'AGILMFPWV') / len(sequence)
    
    # 序列长度（归一化）
    length_norm = len(sequence) / 1000
    
    # 组合所有特征
    features = np.concatenate([
        aa_composition,  # 20维
        [hydrophobic, hydrophilic, aromatic, positive, negative, 
         net_charge, polar, nonpolar, length_norm]  # 9维
    ])
    
    return features


# ==============================================================================
# 第四部分：主程序
# ==============================================================================

def main():
    """主程序"""
    
    print("\n" + "="*80)
    print("🚀 ESM-2特征提取流程")
    print("="*80)
    
    # 1. 加载序列数据
    print("\n1. 加载序列数据...")
    with open(DATA_DIR / "real_sequences.json", 'r') as f:
        sequences_data = json.load(f)
    
    # 提取序列字典
    sequences = {k: v['sequence'] for k, v in sequences_data.items()}
    print(f"   加载了 {len(sequences)} 个序列")
    
    # 2. 初始化ESM-2提取器
    print("\n2. 初始化ESM-2模型...")
    extractor = ESM2FeatureExtractor(model_name="esm2_t6_8M_UR50D")
    
    # 3. 提取ESM-2特征
    print("\n3. 提取ESM-2特征...")
    esm_features = extractor.extract_features_batch(sequences)
    
    # 4. 提取物理化学特征
    print("\n4. 提取物理化学特征...")
    phys_features = {}
    for uniprot_id, sequence in tqdm(sequences.items(), desc="物理化学特征"):
        phys_features[uniprot_id] = extract_physicochemical_features(sequence)
    
    # 5. 组合特征
    print("\n5. 组合特征...")
    combined_features = {}
    for uniprot_id in sequences.keys():
        if uniprot_id in esm_features and uniprot_id in phys_features:
            combined_features[uniprot_id] = {
                'esm_features': esm_features[uniprot_id].tolist(),
                'phys_features': phys_features[uniprot_id].tolist(),
                'combined': np.concatenate([
                    esm_features[uniprot_id],
                    phys_features[uniprot_id]
                ]).tolist()
            }
    
    # 6. 保存特征
    print("\n6. 保存特征...")
    
    # 保存ESM特征
    with open(FEATURES_DIR / "esm_features.json", 'w') as f:
        json.dump({k: v.tolist() for k, v in esm_features.items()}, f, indent=2)
    
    # 保存物理化学特征
    with open(FEATURES_DIR / "phys_features.json", 'w') as f:
        json.dump(phys_features, f, indent=2)
    
    # 保存组合特征
    with open(FEATURES_DIR / "combined_features.json", 'w') as f:
        json.dump(combined_features, f, indent=2)
    
    # 保存为numpy数组（便于模型训练）
    feature_matrix = np.array([combined_features[k]['combined'] for k in sorted(combined_features.keys())])
    np.save(FEATURES_DIR / "feature_matrix.npy", feature_matrix)
    
    # 保存样本ID列表
    with open(FEATURES_DIR / "sample_ids.json", 'w') as f:
        json.dump(sorted(combined_features.keys()), f, indent=2)
    
    print(f"\n   ✅ 特征保存完成!")
    print(f"   - ESM特征: {FEATURES_DIR / 'esm_features.json'}")
    print(f"   - 物理化学特征: {FEATURES_DIR / 'phys_features.json'}")
    print(f"   - 组合特征: {FEATURES_DIR / 'combined_features.json'}")
    print(f"   - 特征矩阵: {FEATURES_DIR / 'feature_matrix.npy'}")
    
    # 统计信息
    print("\n" + "="*80)
    print("📊 特征统计")
    print("="*80)
    print(f"  样本数: {len(combined_features)}")
    print(f"  ESM特征维度: {len(esm_features[list(esm_features.keys())[0]])}")
    print(f"  物理化学特征维度: {len(phys_features[list(phys_features.keys())[0]])}")
    print(f"  总特征维度: {feature_matrix.shape[1]}")
    
    print("\n" + "="*80)
    print("✅ ESM-2特征提取完成!")
    print("="*80)
    
    return combined_features


if __name__ == "__main__":
    features = main()
