#!/usr/bin/env python3
"""
获取额外GPCR样本以扩展数据集至100+
从GPCRdb和UniProt获取Gαq、Gi/o、Gs偶联GPCR
"""
import json
import requests
import time
from pathlib import Path

# GPCRdb API端点
GPCRDB_API = "https://gpcrdb.org/services"

# 已知Gαq偶联GPCR家族 (目标: +20)
Gq_COUPLING_FAMILIES = [
    # 已经确定的家族
    "Cysteinyl_leukotriene_receptor",
    "Acetylcholine_receptor_muscarinic",
    "Adrenoceptor",
    "Histamine_receptor_H1",
    "Oxytocin_receptor",
    "Vasopressin_receptor",
    "Angiotensin_receptor",
    "Prostanoid_receptor",
    "Platelet-activating_factor_receptor",
    "Tachykinin_receptor",
    "Cholecystokinin_receptor",
    "Neurotensin_receptor",
    "Ghrelin_receptor",
    "Melanocortin_receptor",
    "Orexin_receptor",
    "Proteinase-activated_receptor",
]

# Gi/o偶联GPCR家族 (已有16个，目标: +10)
Gi_COUPLING_FAMILIES = [
    "Adrenoceptor_alpha",
    "Dopamine_receptor_D2",
    "Dopamine_receptor_D4",
    "Histamine_receptor_H3",
    "Histamine_receptor_H4",
    "Muscarinic_acetylcholine_receptor_M2",
    "Muscarinic_acetylcholine_receptor_M4",
    "Opioid_receptor",
    "Somatostatin_receptor",
    "Galanin_receptor",
    "Cannabinoid_receptor",
    "Melatonin_receptor",
    "Adenosine_receptor",
    "Prostaglandin_receptor_EP1",
]

# Gs偶联GPCR家族 (已有8个，目标: +5)
Gs_COUPLING_FAMILIES = [
    "Beta-adrenergic_receptor",
    "Glucagon_receptor",
    "Secretin_receptor",
    "Vasoactive_intestinal_polypeptide_receptor",
    "Parathyroid_hormone_receptor",
    "Calcitonin_receptor",
    "Corticotropin_releasing_factor_receptor",
    "Follicle_stimulating_hormone_receptor",
]

def fetch_from_gpcrdb():
    """从GPCRdb获取受体列表"""
    try:
        url = f"{GPCRDB_API}/protein/"
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            return response.json()
        return []
    except Exception as e:
        print(f"[ERROR] GPCRdb请求失败: {e}")
        return []

def fetch_uniprot_sequence(uniprot_id):
    """从UniProt获取序列"""
    try:
        url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            data = response.json()
            seq_info = data.get('sequence', {})
            return seq_info.get('sequence', '')
        return None
    except Exception as e:
        print(f"[ERROR] UniProt请求失败 {uniprot_id}: {e}")
        return None

# 手动整理额外47个GPCR样本
ADDITIONAL_SAMPLES = {
    # === Gαq偶联 (+18个) ===
    "Gq_001": {"name": "TBXA2R", "uniprot": "P21731", "family": "Thromboxane_A2_receptor", "label": 1},
    "Gq_002": {"name": "F2R", "uniprot": "P25116", "family": "Proteinase-activated_receptor_1", "label": 1},
    "Gq_003": {"name": "F2RL1", "uniprot": "P55085", "family": "Proteinase-activated_receptor_2", "label": 1},
    "Gq_004": {"name": "F2RL3", "uniprot": "Q96RI0", "family": "Proteinase-activated_receptor_4", "label": 1},
    "Gq_005": {"name": "HTR2C", "uniprot": "P28335", "family": "Serotonin_receptor_2C", "label": 1},
    "Gq_006": {"name": "HTR2B", "uniprot": "P41595", "family": "Serotonin_receptor_2B", "label": 1},
    "Gq_007": {"name": "EDNRA", "uniprot": "P21453", "family": "Endothelin_receptor_A", "label": 1},
    "Gq_008": {"name": "EDNRB", "uniprot": "P24530", "family": "Endothelin_receptor_B", "label": 1},
    "Gq_009": {"name": "AGTR2", "uniprot": "P50052", "family": "Angiotensin_II_receptor_type_2", "label": 1},
    "Gq_010": {"name": "NMBR", "uniprot": "P28336", "family": "Neuromedin-B_receptor", "label": 1},
    "Gq_011": {"name": "GRPR", "uniprot": "P30550", "family": "Gastrin-releasing_peptide_receptor", "label": 1},
    "Gq_012": {"name": "BRS3", "uniprot": "P32247", "family": "Bombesin_receptor_subtype_3", "label": 1},
    "Gq_013": {"name": "NPY1R", "uniprot": "P25929", "family": "Neuropeptide_Y_receptor_Y1", "label": 1},
    "Gq_014": {"name": "NPY2R", "uniprot": "P49146", "family": "Neuropeptide_Y_receptor_Y2", "label": 1},
    "Gq_015": {"name": "NPY5R", "uniprot": "Q15761", "family": "Neuropeptide_Y_receptor_Y5", "label": 1},
    "Gq_016": {"name": "S1PR2", "uniprot": "O95136", "family": "Sphingosine-1-phosphate_receptor_2", "label": 1},
    "Gq_017": {"name": "S1PR3", "uniprot": "Q99500", "family": "Sphingosine-1-phosphate_receptor_3", "label": 1},
    "Gq_018": {"name": "LPAR1", "uniprot": "Q92633", "family": "Lysophosphatidic_acid_receptor_1", "label": 1},

    # === Gi/o偶联 (+17个) ===
    "Gi_017": {"name": "DRD1", "uniprot": "P21728", "family": "Dopamine_receptor_D1", "label": 0},
    "Gi_018": {"name": "DRD5", "uniprot": "P21918", "family": "Dopamine_receptor_D5", "label": 0},
    "Gi_019": {"name": "ADRA2A", "uniprot": "P08913", "family": "Alpha-2A_adrenergic_receptor", "label": 0},
    "Gi_020": {"name": "ADRA2B", "uniprot": "P18089", "family": "Alpha-2B_adrenergic_receptor", "label": 0},
    "Gi_021": {"name": "HRH3", "uniprot": "Q9Y5N1", "family": "Histamine_H3_receptor", "label": 0},
    "Gi_022": {"name": "HRH4", "uniprot": "Q9H3N8", "family": "Histamine_H4_receptor", "label": 0},
    "Gi_023": {"name": "CHRM2", "uniprot": "P08172", "family": "Muscarinic_acetylcholine_receptor_M2", "label": 0},
    "Gi_024": {"name": "CHRM4", "uniprot": "P08173", "family": "Muscarinic_acetylcholine_receptor_M4", "label": 0},
    "Gi_025": {"name": "OPRD1", "uniprot": "P41143", "family": "Delta-type_opioid_receptor", "label": 0},
    "Gi_026": {"name": "OPRK1", "uniprot": "P41145", "family": "Kappa-type_opioid_receptor", "label": 0},
    "Gi_027": {"name": "SSTR1", "uniprot": "P30872", "family": "Somatostatin_receptor_type_1", "label": 0},
    "Gi_028": {"name": "SSTR2", "uniprot": "P30874", "family": "Somatostatin_receptor_type_2", "label": 0},
    "Gi_029": {"name": "GALR1", "uniprot": "P47211", "family": "Galanin_receptor_type_1", "label": 0},
    "Gi_030": {"name": "GALR2", "uniprot": "O43603", "family": "Galanin_receptor_type_2", "label": 0},
    "Gi_031": {"name": "MTNR1A", "uniprot": "P48039", "family": "Melatonin_receptor_type_1A", "label": 0},
    "Gi_032": {"name": "MTNR1B", "uniprot": "P49286", "family": "Melatonin_receptor_type_1B", "label": 0},
    "Gi_033": {"name": "CNR2", "uniprot": "P34972", "family": "Cannabinoid_receptor_2", "label": 0},

    # === Gs偶联 (+12个) ===
    "Gs_009": {"name": "ADRB1", "uniprot": "P08588", "family": "Beta-1_adrenergic_receptor", "label": 0},
    "Gs_010": {"name": "ADRB2", "uniprot": "P07550", "family": "Beta-2_adrenergic_receptor", "label": 0},
    "Gs_011": {"name": "GLP1R", "uniprot": "P43220", "family": "Glucagon-like_peptide_1_receptor", "label": 0},
    "Gs_012": {"name": "GIPR", "uniprot": "P48546", "family": "Gastric_inhibitory_polypeptide_receptor", "label": 0},
    "Gs_013": {"name": "ADCYAP1R1", "uniprot": "P41586", "family": "PACAP_receptor_type_1", "label": 0},
    "Gs_014": {"name": "VIPR1", "uniprot": "P32241", "family": "VIP_receptor_type_1", "label": 0},
    "Gs_015": {"name": "PTHR1", "uniprot": "Q03431", "family": "Parathyroid_hormone_receptor_1", "label": 0},
    "Gs_016": {"name": "CRHR1", "uniprot": "P34998", "family": "Corticotropin-releasing_factor_receptor_1", "label": 0},
    "Gs_017": {"name": "CRHR2", "uniprot": "Q13324", "family": "Corticotropin-releasing_factor_receptor_2", "label": 0},
    "Gs_018": {"name": "CALCR", "uniprot": "P30988", "family": "Calcitonin_receptor", "label": 0},
    "Gs_019": {"name": "FSHR", "uniprot": "P23945", "family": "Follicle-stimulating_hormone_receptor", "label": 0},
    "Gs_020": {"name": "LHCGR", "uniprot": "P22888", "family": "Luteinizing_hormone_choriogonadotropin_receptor", "label": 0},
}

def main():
    print("=" * 70)
    print("GPCR数据集扩展 - 获取额外47个样本")
    print("=" * 70)

    # 统计
    print(f"\n计划获取样本:")
    print(f"  Gαq偶联 (正样本): 18个")
    print(f"  Gi/o偶联 (负样本): 17个")
    print(f"  Gs偶联 (负样本): 12个")
    print(f"  总计: 47个")

    # 获取序列
    sequences = {}
    labels = {}
    failed = []

    for sample_id, info in ADDITIONAL_SAMPLES.items():
        uniprot_id = info['uniprot']
        print(f"\n[{sample_id}] {info['name']} ({uniprot_id})...", end=' ')

        seq = fetch_uniprot_sequence(uniprot_id)
        time.sleep(0.5)  # 避免请求过快

        if seq and len(seq) > 200:  # 确保是完整的GPCR序列
            sequences[sample_id] = {
                'uniprot_id': uniprot_id,
                'name': info['name'],
                'family': info['family'],
                'sequence': seq,
                'label': info['label']
            }
            labels[sample_id] = info['label']
            print(f"[OK] 长度={len(seq)}")
        else:
            failed.append(sample_id)
            print(f"[FAIL]")

    # 保存结果
    OUTPUT_DIR = Path('E:/kimi/Kimi_Agent_批判者监督执行/extended_samples')
    OUTPUT_DIR.mkdir(exist_ok=True)

    with open(OUTPUT_DIR / 'additional_47_samples.json', 'w') as f:
        json.dump(sequences, f, indent=2)

    with open(OUTPUT_DIR / 'additional_47_labels.json', 'w') as f:
        json.dump(labels, f, indent=2)

    # 统计
    print("\n" + "=" * 70)
    print("获取结果统计:")
    print(f"  成功: {len(sequences)}个")
    print(f"  失败: {len(failed)}个")
    if failed:
        print(f"  失败列表: {failed}")

    # 类别分布
    pos_count = sum(1 for l in labels.values() if l == 1)
    neg_count = sum(1 for l in labels.values() if l == 0)
    print(f"\n类别分布:")
    print(f"  正样本(Gαq): {pos_count}个")
    print(f"  负样本(Gi/o+Gs): {neg_count}个")

    print(f"\n[OK] 结果保存到: {OUTPUT_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    main()
