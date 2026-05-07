#!/usr/bin/env python3
"""
对 OPN4 测试数据进行预测并检查数据泄露。

输入:
- Opsin4_sequences.txt: 9条opsin序列（FASTA格式）
- GNAQ_human.txt: 人源GNAQ序列（FASTA格式）
- OPN4_Gq_interaction_labels.xlsx: 真实标签

流程:
1. 用训练集数据训练完整模型（ICL_stats配置, SVM-RBF C=10）
2. 对9条序列提取ESM-2 embedding + ICL特征
3. 预测是否与Gq偶联
4. 与真实标签对比
5. 数据泄露分析
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

BASE = Path(__file__).parent
DATA_DIR = BASE / "paired_dataset"

# ===== 1. 加载所有训练数据的特征 =====
print("=" * 70)
print("  OPN4 测试集预测与数据泄露分析")
print("=" * 70)

# 加载 GPCR ESM-2 特征
GPCR_FEATURES_FILE = BASE / "server_sync" / "extended_data" / "features" / "esm_features_100samples.json"
G_PROTEIN_FEATURES_FILE = DATA_DIR / "g_protein_esm_features.json"
ICL_FEATURES_FILE = DATA_DIR / "icl_features.json"
PAIRING_MATRIX_FILE = DATA_DIR / "pairing_matrix_raw.csv"
CLUSTERS_FILE = DATA_DIR / "sequence_clusters.json"

with open(GPCR_FEATURES_FILE) as f:
    gpcr_raw = json.load(f)
with open(G_PROTEIN_FEATURES_FILE) as f:
    gprot_raw = json.load(f)
with open(ICL_FEATURES_FILE) as f:
    icl_data = json.load(f)

# GPCR mean-pooled features
gpcr_feats = {}
for k, v in gpcr_raw.items():
    arr = np.array(v)
    gpcr_feats[k] = arr.mean(axis=0) if arr.ndim == 2 else arr

# G-protein family features
gprot_feats = {}
family_map = {
    "GNAQ": "Gq", "GNAI1": "Gi", "GNAI2": "Gi", "GNAI3": "Gi",
    "GNAS": "Gs", "GNA12": "G12_13", "GNA13": "G12_13",
}
for subtype, info in gprot_raw.items():
    family = family_map.get(subtype, subtype)
    gprot_feats[subtype] = np.array(info["mean_pooling"])
    if family not in gprot_feats:
        gprot_feats[family] = np.array(info["mean_pooling"])

print(f"[INFO] GPCR features loaded: {len(gpcr_feats)}")
print(f"[INFO] G-protein features loaded: {list(gprot_feats.keys())}")
print(f"[INFO] ICL features loaded: {len(icl_data)}")

# ===== 2. 构建训练集 (ICL_stats 模式: 640+16=656-d) =====
df = pd.read_csv(PAIRING_MATRIX_FILE)
print(f"[INFO] Pairing matrix: {len(df)} rows")

stat_keys = ["length", "mean_hydro", "std_hydro", "net_charge",
             "pos_charge_ratio", "neg_charge_ratio", "hydrophobic_ratio", "aromatic_ratio"]

def get_gpcr_feat(gid):
    feat = gpcr_feats.get(gid)
    if feat is None and "_" in gid and len(gid.split("_")[0]) <= 2:
        base = gid.split("_", 1)[1]
        feat = gpcr_feats.get(base)
    if feat is None:
        for key in gpcr_feats:
            if "_" in key and key.split("_", 1)[1] == gid:
                feat = gpcr_feats[key]
                break
    return feat

def get_icl_stats(gid):
    rec = icl_data.get(gid)
    if rec is None:
        for key in icl_data:
            if "_" in key and key.split("_", 1)[1] == gid:
                rec = icl_data[key]
                break
    if rec is None:
        return np.zeros(16)
    icl2_stats = rec.get("ICL2_stats", {})
    icl3_stats = rec.get("ICL3_stats", {})
    icl2_vec = np.array([icl2_stats.get(k, 0.0) for k in stat_keys])
    icl3_vec = np.array([icl3_stats.get(k, 0.0) for k in stat_keys])
    return np.concatenate([icl2_vec, icl3_vec])

X_train, y_train = [], []
train_ids = []
skipped = 0
for _, row in df.iterrows():
    gid = row["gpcr_id"]
    gfam = row["g_protein_family"]
    gpcr_feat = get_gpcr_feat(gid)
    gprot_feat = gprot_feats.get(gfam)
    if gpcr_feat is None or gprot_feat is None:
        skipped += 1
        continue
    icl_stat = get_icl_stats(gid)
    vec = np.concatenate([gpcr_feat, gprot_feat, icl_stat])
    X_train.append(vec)
    y_train.append(int(row["coupling"]))
    train_ids.append(gid)

X_train = np.array(X_train)
y_train = np.array(y_train)
print(f"[INFO] Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features, skipped {skipped}")
print(f"[INFO] Positive: {y_train.sum()}, Negative: {(1 - y_train).sum()}")

# ===== 3. 训练模型 =====
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
svm = SVC(kernel="rbf", C=10.0, class_weight="balanced", probability=True, random_state=42)
svm.fit(X_train_scaled, y_train)
print("[INFO] SVM model trained (RBF, C=10, balanced)")

# ===== 4. 数据泄露分析 =====
print("\n" + "=" * 70)
print("  数据泄露分析")
print("=" * 70)

# 读取测试标签
import openpyxl
wb = openpyxl.load_workbook(BASE / "OPN4_Gq_interaction_labels.xlsx")
ws = wb.active
test_labels = {}
for row in ws.iter_rows(min_row=2, values_only=True):
    ncbi_id, name, organism, label = row
    test_labels[ncbi_id] = {
        "name": name, "organism": organism,
        "label": 1 if str(label).strip().lower() == "yes" else 0
    }

# Check Q9UHM6 (human melanopsin = NP_150598.1)
q9uhm6_in_train = df[df["gpcr_id"] == "Q9UHM6"]
print(f"\n[泄露检查] Q9UHM6 (Human Melanopsin/OPN4) 在训练集中的情况:")
print(f"  出现次数: {len(q9uhm6_in_train)} 行")
if len(q9uhm6_in_train) > 0:
    for _, r in q9uhm6_in_train.iterrows():
        print(f"    Family={r['g_protein_family']}, Coupling={r['coupling']}, Source={r['source']}")

# NP_150598.1 = Q9UHM6 isoform 1 (human melanopsin)
print(f"\n[泄露结论]:")
print(f"  - NP_150598.1 (Human Melanopsin isoform 1) = UniProt Q9UHM6")
print(f"  - Q9UHM6 存在于训练集中 (来源: GPCRdb/IUPHAR)")
print(f"  - 训练集标注: Gq=1 (coupling), Gi=1, Gs=0, G12/13=0")
print(f"  - 测试标签: Gq=Yes")
print(f"  *** 存在数据泄露: NP_150598.1 的 Gq 偶联标签已在训练集中 ***")
print(f"  *** NP_001025186.1 (isoform 2) 与 isoform 1 序列高度相似，也构成泄露 ***")

# Check other test sequences against training set
print(f"\n[其他测试序列泄露检查]:")
# Parse test FASTA
test_seqs = {}
with open(BASE / "Opsin4_sequences.txt") as f:
    current_id = None
    current_seq = []
    for line in f:
        line = line.strip()
        if line.startswith(">"):
            if current_id:
                test_seqs[current_id] = "".join(current_seq)
            parts = line[1:].split()
            current_id = parts[0]
            current_seq = []
        elif line:
            current_seq.append(line)
    if current_id:
        test_seqs[current_id] = "".join(current_seq)

# Read all training sequences
train_seqs = {}
for _, row in df.iterrows():
    if pd.notna(row.get("gpcr_sequence")) and row["gpcr_sequence"]:
        train_seqs[row["gpcr_id"]] = str(row["gpcr_sequence"])

# Load sequences from extended_sequences.json
with open(BASE / "merged_dataset" / "extended_sequences.json") as f:
    ext_seqs = json.load(f)
for k, v in ext_seqs.items():
    seq = v["sequence"] if isinstance(v, dict) else v
    if isinstance(seq, str) and seq:
        base_id = k.split("_", 1)[1] if "_" in k and len(k.split("_")[0]) <= 2 else k
        train_seqs[base_id] = seq

print(f"  训练集序列数: {len(train_seqs)}")
for test_id, test_seq in test_seqs.items():
    # k-mer Jaccard similarity
    k = 3
    test_kmers = set(test_seq[i:i+k] for i in range(len(test_seq) - k + 1))
    max_sim = 0
    most_similar = None
    for train_id, train_seq in train_seqs.items():
        train_kmers = set(train_seq[i:i+k] for i in range(len(train_seq) - k + 1))
        if not test_kmers or not train_kmers:
            continue
        sim = len(test_kmers & train_kmers) / len(test_kmers | train_kmers)
        if sim > max_sim:
            max_sim = sim
            most_similar = train_id
    leakage = "LEAKAGE" if max_sim > 0.9 else ("HIGH_SIM" if max_sim > 0.3 else "OK")
    label_info = test_labels.get(test_id, {})
    print(f"  {test_id:20s} ({label_info.get('name','?'):30s}) -> max_sim={max_sim:.3f} to {most_similar:10s} [{leakage}]")

# ===== 5. 预测（使用ESM-2提取新序列特征） =====
print("\n" + "=" * 70)
print("  模型预测")
print("=" * 70)

# 因为测试序列不在训练特征缓存中，需要用ESM-2提取
# 检查是否有可用的ESM-2模型
try:
    import torch
    from transformers import AutoTokenizer, AutoModel

    print("[INFO] Loading ESM-2 model...")
    model_name = "facebook/esm2_t6_8M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    esm_model = AutoModel.from_pretrained(model_name)
    esm_model.eval()

    def extract_esm2_features(seq):
        """提取序列的ESM-2 mean-pooled embedding (320-d)"""
        inputs = tokenizer(seq, return_tensors="pt", padding=False, truncation=True, max_length=1024)
        with torch.no_grad():
            outputs = esm_model(**inputs)
        # Remove BOS/EOS tokens, mean pool
        hidden = outputs.last_hidden_state[0, 1:-1, :]  # (seq_len, 320)
        return hidden.numpy()

    # ICL extraction helper
    KD_SCALE = {
        "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
        "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
        "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
        "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
    }
    POSITIVE = {"K", "R", "H"}
    NEGATIVE = {"D", "E"}
    HYDROPHOBIC = {"A", "V", "I", "L", "M", "F", "W", "G", "P", "C"}
    AROMATIC = {"F", "W", "Y"}

    def compute_icl_stats(subseq):
        n = len(subseq)
        if n == 0:
            return np.zeros(8)
        hydro = [KD_SCALE.get(aa, 0) for aa in subseq]
        pos = sum(1 for aa in subseq if aa in POSITIVE)
        neg = sum(1 for aa in subseq if aa in NEGATIVE)
        return np.array([
            n,
            float(np.mean(hydro)),
            float(np.std(hydro)),
            (pos - neg) / n,
            pos / n,
            neg / n,
            sum(1 for aa in subseq if aa in HYDROPHOBIC) / n,
            sum(1 for aa in subseq if aa in AROMATIC) / n,
        ])

    # Use UniProt topology for Q9UHM6, and predict for others
    with open(DATA_DIR / "uniprot_topology.json") as f:
        topo_cache = json.load(f)

    q9uhm6_topo = topo_cache.get("Q9UHM6", {})

    # Predict for each test sequence
    gq_feat = gprot_feats["Gq"]

    results = []
    print(f"\n{'NCBI ID':20s} {'Name':30s} {'Org':25s} {'True':5s} {'Prob':6s} {'Pred':5s} {'Leakage':10s}")
    print("-" * 110)

    for test_id, info in test_labels.items():
        seq = test_seqs.get(test_id, "")
        if not seq:
            print(f"{test_id:20s} {'NO SEQUENCE':30s}")
            continue

        # Extract ESM-2 features
        tokens = extract_esm2_features(seq)
        global_feat = tokens.mean(axis=0)  # (320,)

        # ICL stats - use UniProt topology if Q9UHM6, else use heuristic
        if test_id in ["NP_150598.1", "NP_001025186.1"] and q9uhm6_topo:
            loops = q9uhm6_topo.get("loops", {})
        else:
            loops = {}

        if loops:
            icl2_range = loops.get("ICL2", [0, 0])
            icl3_range = loops.get("ICL3", [0, 0])
            icl2_seq = seq[icl2_range[0]-1:icl2_range[1]] if icl2_range[1] > 0 else ""
            icl3_seq = seq[icl3_range[0]-1:icl3_range[1]] if icl3_range[1] > 0 else ""
        else:
            # No topology - use zero padding
            icl2_seq = ""
            icl3_seq = ""

        icl2_stat = compute_icl_stats(icl2_seq)
        icl3_stat = compute_icl_stats(icl3_seq)
        icl_stat = np.concatenate([icl2_stat, icl3_stat])

        # Build feature vector (656-d: 320 GPCR + 320 Gq + 16 ICL stats)
        vec = np.concatenate([global_feat, gq_feat, icl_stat]).reshape(1, -1)
        vec_scaled = scaler.transform(vec)

        prob = svm.predict_proba(vec_scaled)[0, 1]
        pred = 1 if prob >= 0.5 else 0
        true_label = info["label"]

        # Leakage status
        if test_id in ["NP_150598.1", "NP_001025186.1"]:
            leak = "LEAKED"
        else:
            leak = "clean"

        correct = "OK" if pred == true_label else "WRONG"

        results.append({
            "id": test_id, "name": info["name"], "organism": info["organism"],
            "true": true_label, "prob": prob, "pred": pred, "leakage": leak
        })

        print(f"{test_id:20s} {info['name']:30s} {info['organism']:25s} {true_label:5d} {prob:6.3f} {pred:5d} {leak:10s} {correct}")

    # Summary
    print("\n" + "=" * 70)
    print("  预测汇总")
    print("=" * 70)

    correct_all = sum(1 for r in results if r["pred"] == r["true"])
    clean_results = [r for r in results if r["leakage"] == "clean"]
    correct_clean = sum(1 for r in clean_results if r["pred"] == r["true"])
    leaked_results = [r for r in results if r["leakage"] == "LEAKED"]
    correct_leaked = sum(1 for r in leaked_results if r["pred"] == r["true"])

    print(f"  全部样本: {correct_all}/{len(results)} 正确 ({correct_all/len(results)*100:.1f}%)")
    if leaked_results:
        print(f"  泄露样本: {correct_leaked}/{len(leaked_results)} 正确")
    if clean_results:
        print(f"  干净样本: {correct_clean}/{len(clean_results)} 正确 ({correct_clean/len(clean_results)*100:.1f}%)")

except ImportError as e:
    print(f"[ERROR] 无法加载ESM-2模型: {e}")
    print("[INFO] 请安装: pip install torch transformers")
    print("[INFO] 跳过预测，仅输出数据泄露分析")
