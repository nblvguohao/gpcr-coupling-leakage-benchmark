#!/usr/bin/env python3
"""
使用BioPython从UniProt获取额外GPCR样本
"""
import json
import time
from pathlib import Path
from Bio import ExPASy
from Bio import SeqIO

# 额外47个GPCR样本 (UniProt ID列表)
UNIPROT_IDS = {
    # Gαq偶联 (18个)
    "P21731": {"name": "TBXA2R", "label": 1, "family": "Thromboxane_A2_receptor"},
    "P25116": {"name": "F2R", "label": 1, "family": "PAR1"},
    "P55085": {"name": "F2RL1", "label": 1, "family": "PAR2"},
    "Q96RI0": {"name": "F2RL3", "label": 1, "family": "PAR4"},
    "P28335": {"name": "HTR2C", "label": 1, "family": "5-HT2C"},
    "P41595": {"name": "HTR2B", "label": 1, "family": "5-HT2B"},
    "P21453": {"name": "EDNRA", "label": 1, "family": "Endothelin_A"},
    "P24530": {"name": "EDNRB", "label": 1, "family": "Endothelin_B"},
    "P50052": {"name": "AGTR2", "label": 1, "family": "AT2"},
    "P28336": {"name": "NMBR", "label": 1, "family": "NMB"},
    "P30550": {"name": "GRPR", "label": 1, "family": "GRP"},
    "P32247": {"name": "BRS3", "label": 1, "family": "Bombesin_3"},
    "P25929": {"name": "NPY1R", "label": 1, "family": "NPY1"},
    "P49146": {"name": "NPY2R", "label": 1, "family": "NPY2"},
    "Q15761": {"name": "NPY5R", "label": 1, "family": "NPY5"},
    "O95136": {"name": "S1PR2", "label": 1, "family": "S1P2"},
    "Q99500": {"name": "S1PR3", "label": 1, "family": "S1P3"},
    "Q92633": {"name": "LPAR1", "label": 1, "family": "LPA1"},

    # Gi/o偶联 (17个)
    "P21728": {"name": "DRD1", "label": 0, "family": "D1"},
    "P21918": {"name": "DRD5", "label": 0, "family": "D5"},
    "P08913": {"name": "ADRA2A", "label": 0, "family": "Alpha2A"},
    "P18089": {"name": "ADRA2B", "label": 0, "family": "Alpha2B"},
    "Q9Y5N1": {"name": "HRH3", "label": 0, "family": "H3"},
    "Q9H3N8": {"name": "HRH4", "label": 0, "family": "H4"},
    "P08172": {"name": "CHRM2", "label": 0, "family": "M2"},
    "P08173": {"name": "CHRM4", "label": 0, "family": "M4"},
    "P41143": {"name": "OPRD1", "label": 0, "family": "Delta_opioid"},
    "P41145": {"name": "OPRK1", "label": 0, "family": "Kappa_opioid"},
    "P30872": {"name": "SSTR1", "label": 0, "family": "SST1"},
    "P30874": {"name": "SSTR2", "label": 0, "family": "SST2"},
    "P47211": {"name": "GALR1", "label": 0, "family": "Galanin_1"},
    "O43603": {"name": "GALR2", "label": 0, "family": "Galanin_2"},
    "P48039": {"name": "MTNR1A", "label": 0, "family": "MT1"},
    "P49286": {"name": "MTNR1B", "label": 0, "family": "MT2"},
    "P34972": {"name": "CNR2", "label": 0, "family": "CB2"},

    # Gs偶联 (12个)
    "P08588": {"name": "ADRB1", "label": 0, "family": "Beta1"},
    "P07550": {"name": "ADRB2", "label": 0, "family": "Beta2"},
    "P43220": {"name": "GLP1R", "label": 0, "family": "GLP-1"},
    "P48546": {"name": "GIPR", "label": 0, "family": "GIP"},
    "P41586": {"name": "ADCYAP1R1", "label": 0, "family": "PAC1"},
    "P32241": {"name": "VIPR1", "label": 0, "family": "VPAC1"},
    "Q03431": {"name": "PTHR1", "label": 0, "family": "PTH1"},
    "P34998": {"name": "CRHR1", "label": 0, "family": "CRF1"},
    "Q13324": {"name": "CRHR2", "label": 0, "family": "CRF2"},
    "P30988": {"name": "CALCR", "label": 0, "family": "Calcitonin"},
    "P23945": {"name": "FSHR", "label": 0, "family": "FSH"},
    "P22888": {"name": "LHCGR", "label": 0, "family": "LH/CG"},
}

def fetch_sequence(uniprot_id):
    """从UniProt获取序列"""
    try:
        handle = ExPASy.get_sprot_raw(uniprot_id)
        record = SeqIO.read(handle, "swiss")
        handle.close()
        return str(record.seq)
    except Exception as e:
        print(f"  [FAIL] {e}")
        return None

def main():
    print("=" * 70)
    print("GPCR Dataset Extension - Fetching 47 Additional Samples")
    print("=" * 70)

    sequences = {}
    labels = {}
    failed = []
    success = []

    total = len(UNIPROT_IDS)
    for i, (uniprot_id, info) in enumerate(UNIPROT_IDS.items(), 1):
        print(f"\n[{i}/{total}] {info['name']} ({uniprot_id})...")

        seq = fetch_sequence(uniprot_id)
        time.sleep(0.3)  # 避免请求过快

        if seq and len(seq) > 200:
            sample_id = f"{list(info['family'])[0]}_{uniprot_id}"
            sequences[sample_id] = {
                'uniprot_id': uniprot_id,
                'name': info['name'],
                'family': info['family'],
                'sequence': seq,
                'label': info['label']
            }
            labels[sample_id] = info['label']
            success.append(uniprot_id)
            print(f"  [OK] Length={len(seq)}")
        else:
            failed.append(uniprot_id)
            print(f"  [FAIL]")

    # 保存结果
    OUTPUT_DIR = Path('E:/kimi/Kimi_Agent_批判者监督执行/extended_samples')
    OUTPUT_DIR.mkdir(exist_ok=True)

    with open(OUTPUT_DIR / 'additional_samples.json', 'w') as f:
        json.dump(sequences, f, indent=2)

    with open(OUTPUT_DIR / 'additional_labels.json', 'w') as f:
        json.dump(labels, f, indent=2)

    # 统计
    print("\n" + "=" * 70)
    print("Fetch Results:")
    print(f"  Success: {len(success)}/{total}")
    print(f"  Failed: {len(failed)}/{total}")

    if success:
        pos = sum(1 for s in success if UNIPROT_IDS[s]['label'] == 1)
        neg = sum(1 for s in success if UNIPROT_IDS[s]['label'] == 0)
        print(f"\nClass Distribution:")
        print(f"  Positive (Gq): {pos}")
        print(f"  Negative (Gi/o+Gs): {neg}")

    if failed:
        print(f"\nFailed IDs: {failed}")

    print(f"\n[OK] Results saved to: {OUTPUT_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    main()
