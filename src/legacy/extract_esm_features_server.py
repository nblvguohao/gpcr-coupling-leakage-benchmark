#!/usr/bin/env python3
"""
在服务器上为100个样本提取ESM-2特征
使用真实数据，非模拟
"""
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

# 导入ESM
import esm

# 路径设置
DATA_DIR = Path('/data/lgh/GPCR/extended_data')
OUTPUT_DIR = Path('/data/lgh/GPCR/extended_data/features')
OUTPUT_DIR.mkdir(exist_ok=True)

def load_esm_model():
    """加载ESM-2模型"""
    print("[INFO] 加载ESM-2模型...")
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model = model.eval().cuda()
    batch_converter = alphabet.get_batch_converter()
    return model, batch_converter

def extract_features(sequences_dict, model, batch_converter):
    """提取ESM-2特征"""
    features_dict = {}

    # 准备数据
    data = [(uid, seq['sequence'] if isinstance(seq, dict) else seq)
            for uid, seq in sequences_dict.items()]

    # 批量处理
    batch_size = 8
    for i in tqdm(range(0, len(data), batch_size), desc="提取特征"):
        batch = data[i:i+batch_size]
        _, _, batch_tokens = batch_converter(batch)
        batch_tokens = batch_tokens.cuda()

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[6], return_contacts=False)

        # 获取第6层表示 (320维)
        token_representations = results["representations"][6].cpu().numpy()

        # 为每个序列保存特征
        for j, (uid, _) in enumerate(batch):
            # 去除特殊token (BOS, EOS)
            seq_len = len(batch[j][1])
            features = token_representations[j, 1:seq_len+1, :]
            features_dict[uid] = features.tolist()

    return features_dict

def main():
    print("=" * 70)
    print("ESM-2特征提取 - 100样本扩展数据集")
    print("=" * 70)

    # 加载序列数据
    with open(DATA_DIR / 'extended_sequences.json', 'r') as f:
        sequences = json.load(f)

    print(f"[INFO] 加载 {len(sequences)} 个序列")

    # 加载模型
    model, batch_converter = load_esm_model()

    # 提取特征
    print("[INFO] 开始提取ESM-2特征...")
    features = extract_features(sequences, model, batch_converter)

    # 保存特征
    output_file = OUTPUT_DIR / 'esm_features_100samples.json'
    with open(output_file, 'w') as f:
        json.dump(features, f)

    print(f"\n[OK] 特征已保存到: {output_file}")
    print(f"[OK] 总样本数: {len(features)}")
    print(f"[OK] 特征维度: 320")

    # 验证
    sample_id = list(features.keys())[0]
    sample_feat = np.array(features[sample_id])
    print(f"[OK] 示例特征形状: {sample_feat.shape}")

    print("=" * 70)

if __name__ == "__main__":
    main()
