#!/usr/bin/env python3
"""
获取额外的3个负样本，使总样本数达到50个
选择Gi/o偶联的GPCR作为负样本
"""
from Bio import Entrez, SeqIO
from Bio import ExPASy
import json
import time
from pathlib import Path

# 配置
Entrez.email = "research@example.com"
OUTPUT_FILE = Path('E:/kimi/Kimi_Agent_批判者监督执行/additional_negative_samples.json')

# 选择的3个负样本 - Gi/o偶联GPCR
NEGATIVE_SAMPLES = {
    # ADRA2C - Alpha-2C adrenergic receptor (Gi/o coupled)
    'P18825': {
        'name': 'ADRA2C',
        'description': 'Alpha-2C adrenergic receptor',
        'g_protein': 'Gi/o',
        'function': 'ADRA2C is an alpha-2 adrenergic receptor that couples to Gi/o proteins to inhibit adenylate cyclase'
    },
    # HRH3 - Histamine H3 receptor (Gi/o coupled)
    'Q9Y5N1': {
        'name': 'HRH3',
        'description': 'Histamine H3 receptor',
        'g_protein': 'Gi/o',
        'function': 'HRH3 couples to Gi/o proteins and functions as a presynaptic histamine receptor'
    },
    # CNR1 - Cannabinoid receptor 1 (Gi/o coupled)
    'P21554': {
        'name': 'CNR1',
        'description': 'Cannabinoid receptor 1',
        'g_protein': 'Gi/o',
        'function': 'CNR1 is a cannabinoid receptor that couples to Gi/o proteins to inhibit adenylate cyclase'
    }
}

def fetch_uniprot_sequence(uniprot_id):
    """从UniProt获取序列"""
    try:
        handle = ExPASy.get_sprot_raw(uniprot_id)
        record = SeqIO.read(handle, "swiss")
        handle.close()
        return {
            'uniprot_id': uniprot_id,
            'id': record.id,
            'name': record.name,
            'description': record.description,
            'sequence': str(record.seq),
            'length': len(record.seq)
        }
    except Exception as e:
        print(f"Error fetching {uniprot_id}: {e}")
        return None

def main():
    print("="*70)
    print("获取额外的负样本 (3个 Gi/o偶联GPCR)")
    print("="*70)

    results = {}
    labels = {}

    for uniprot_id, info in NEGATIVE_SAMPLES.items():
        print(f"\n正在获取: {uniprot_id} ({info['name']})...")
        data = fetch_uniprot_sequence(uniprot_id)

        if data:
            results[uniprot_id] = {
                **data,
                'gpcr_info': info
            }
            labels[uniprot_id] = 0  # 负样本
            print(f"  [OK] 成功: {data['name']}")
            print(f"    序列长度: {data['length']} aa")
            print(f"    G蛋白偶联: {info['g_protein']}")
        else:
            print(f"  [FAIL] 失败: {uniprot_id}")

        time.sleep(0.5)  # 避免请求过快

    # 保存结果
    output = {
        'sequences': results,
        'labels': labels
    }

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*70}")
    print(f"成功获取 {len(results)} 个负样本")
    print(f"数据保存到: {OUTPUT_FILE}")
    print(f"{'='*70}")

    # 打印摘要
    print("\n样本摘要:")
    for uid, data in results.items():
        print(f"  {uid}: {data['gpcr_info']['name']} ({data['length']} aa) - {data['gpcr_info']['g_protein']}")

if __name__ == "__main__":
    main()
