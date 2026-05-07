#!/usr/bin/env python3
"""
SHAP 特征重要性归因 — 修正数据 (86 独立样本 + 真实 GNAQ P50148)
使用 SVM-RBF (C=10, balanced) 和 PermutationExplainer。

输出:
- 全局 Top 20 GPCR ESM-2 维度
- 全局 Top 20 GNAQ ESM-2 维度
- 每个样本的残基级伪热图 (post-hoc projection)
- 与 ICL2/ICL3/TM 区域的关联统计
"""
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import shap
import warnings

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
FEATURES_FILE = BASE / "server_sync" / "extended_data" / "features" / "esm_features_100samples.json"
LABELS_FILE = BASE / "merged_dataset" / "extended_labels.json"
SEQUENCES_FILE = BASE / "merged_dataset" / "extended_sequences.json"
GNAQ_FEATURE_FILE = BASE / "server_sync" / "extended_data" / "features" / "gnaq_esm_features.json"
OUTPUT_DIR = BASE / "shap_results_corrected"
OUTPUT_DIR.mkdir(exist_ok=True)


def deduplicate_and_load(features_raw, labels, sequences):
    prefixed = set()
    for k in labels:
        if "_" in k and len(k.split("_")[0]) <= 2:
            base = k.split("_", 1)[1]
            if base in labels:
                prefixed.add(k)
    keep_ids = [k for k in labels if k not in prefixed]

    X_gpcr_list, y_list, id_list, seq_list = [], [], [], []
    for uid in keep_ids:
        feat = np.array(features_raw[uid])
        if feat.ndim == 2:
            feat_mean = feat.mean(axis=0)
            feat_residue = feat
        else:
            feat_mean = feat
            feat_residue = None
        X_gpcr_list.append(feat_mean)
        y_list.append(labels[uid])
        id_list.append(uid)
        seq_list.append(sequences[uid]["sequence"] if isinstance(sequences[uid], dict) else sequences[uid])
    return np.array(X_gpcr_list), np.array(y_list), id_list, seq_list, keep_ids


def approximate_tm_regions(seq: str):
    """使用 21 残基窗口 Kyte-Doolittle 近似预测跨膜区。"""
    HYDRO_SCALE = {
        'I': 4.5, 'V': 4.2, 'L': 3.8, 'F': 2.8, 'C': 2.5,
        'M': 1.9, 'A': 1.8, 'G': -0.4, 'T': -0.7, 'S': -0.8,
        'W': -0.9, 'Y': -1.3, 'P': -1.6, 'H': -3.2, 'E': -3.5,
        'Q': -3.5, 'D': -3.5, 'N': -3.5, 'K': -3.9, 'R': -4.5
    }
    seq = seq.upper()
    L = len(seq)
    hydro = [HYDRO_SCALE.get(aa, 0) for aa in seq]
    tm_regions = []
    in_tm = False
    start = 0
    for i in range(L - 20):
        window_mean = np.mean(hydro[i:i+21])
        if window_mean > 1.5 and not in_tm:
            in_tm = True
            start = i
        elif window_mean <= 1.5 and in_tm:
            in_tm = False
            tm_regions.append((start, i + 21))
    if in_tm:
        tm_regions.append((start, L))
    # 去重/合并重叠区域
    merged = []
    for s, e in tm_regions:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    # 只保留最长的7个（典型 GPCR 有7次跨膜）
    merged = sorted(merged, key=lambda x: x[1]-x[0], reverse=True)[:7]
    merged = sorted(merged, key=lambda x: x[0])
    return merged


def classify_residue_region(idx: int, tm_regions):
    """将残基索引分类到 ICL1/2/3, ECL1/2, TM, N-tail, C-tail。"""
    if not tm_regions:
        return "unknown"
    # 检查是否在 TM 中
    for i, (s, e) in enumerate(tm_regions):
        if s <= idx < e:
            return f"TM{i+1}"
    # 在 TM 之间
    for i in range(len(tm_regions) - 1):
        if tm_regions[i][1] <= idx < tm_regions[i+1][0]:
            gap_name = ["ICL1", "ECL1", "ICL2", "ECL2", "ICL3", "ECL3"]
            if i < len(gap_name):
                return gap_name[i]
            else:
                return f"loop_{i}"
    if idx < tm_regions[0][0]:
        return "N-tail"
    if idx >= tm_regions[-1][1]:
        return "C-tail"
    return "unknown"


def main():
    print("=" * 70)
    print("  SHAP 特征归因 — 修正数据集 (86 样本 + 真实 GNAQ)")
    print("=" * 70)

    # 加载数据
    with open(FEATURES_FILE) as f:
        features_raw = json.load(f)
    with open(LABELS_FILE) as f:
        labels = json.load(f)
    with open(SEQUENCES_FILE) as f:
        sequences = json.load(f)
    with open(GNAQ_FEATURE_FILE) as f:
        gnaq_esm = json.load(f)

    X_gpcr, y, id_list, seq_list, keep_ids = deduplicate_and_load(features_raw, labels, sequences)
    gq_feature = np.array(gnaq_esm["mean_pooling"])
    X = np.concatenate([X_gpcr, np.tile(gq_feature, (len(X_gpcr), 1))], axis=1)
    print(f"[INFO] 样本数: {len(y)} (正: {y.sum()}, 负: {len(y)-y.sum()})")
    print(f"[INFO] 特征维度: {X.shape[1]} (GPCR 0-319, GNAQ 320-639)")

    # 训练 SVM-RBF
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    svm = SVC(kernel="rbf", C=10.0, class_weight="balanced", probability=True, random_state=42)
    svm.fit(X_scaled, y)
    print(f"[INFO] SVM 训练完成。训练集 AUC 估计不可用（全数据训练）。")

    # SHAP
    print("[INFO] 开始计算 Permutation SHAP ...")
    masker = shap.maskers.Independent(X_scaled)
    explainer = shap.explainers.Permutation(svm.predict_proba, masker)
    shap_values = explainer(X_scaled, max_evals=700)  # 640 features, need >= 641

    # 提取 class 1 (Gαq coupling) 的 SHAP 值
    sv_class1 = shap_values.values[:, :, 1]  # (86, 640)

    # 全局平均绝对 SHAP
    mean_abs_sv = np.abs(sv_class1).mean(axis=0)
    gpcr_importance = mean_abs_sv[:320]
    gnaq_importance = mean_abs_sv[320:]

    top_gpcr_dims = np.argsort(gpcr_importance)[::-1][:20]
    top_gnaq_dims = np.argsort(gnaq_importance)[::-1][:20]

    print("\n--- Top 20 GPCR ESM-2 维度 (by |SHAP|) ---")
    for rank, d in enumerate(top_gpcr_dims, 1):
        print(f"  {rank:2d}. dim {d:3d} | SHAP={gpcr_importance[d]:.4f}")

    print("\n--- Top 20 GNAQ ESM-2 维度 (by |SHAP|) ---")
    for rank, d in enumerate(top_gnaq_dims, 1):
        print(f"  {rank:2d}. dim {d:3d} | SHAP={gnaq_importance[d]:.4f}")

    # ===============================
    # 残基级伪热图 (post-hoc projection)
    # ===============================
    print("\n[INFO] 生成残基级伪热图投影 ...")

    # 使用 Top 50 GPCR 维度生成投影权重（比 Top 20 更稳定）
    top_n = 50
    top_dims = np.argsort(gpcr_importance)[::-1][:top_n]
    weights = sv_class1[:, top_dims]  # (86, top_n) — 样本特定的 SHAP 值

    heatmap_data = []
    region_stats = defaultdict(lambda: {"count": 0, "sum_abs_score": 0.0})

    for i, uid in enumerate(keep_ids):
        seq = seq_list[i]
        feat_residue = np.array(features_raw[uid])  # (L, 320)
        if feat_residue.ndim == 1:
            continue
        L = feat_residue.shape[0]

        # 每个残基的投影分数 = sum_d( shap_i,d * feat_residue[pos, d] )
        # weights[i] shape: (top_n,)
        # feat_residue[:, top_dims] shape: (L, top_n)
        scores = feat_residue[:, top_dims] @ weights[i]  # (L,)

        # 归一化到 [-1, 1]
        if np.max(np.abs(scores)) > 0:
            scores_norm = scores / np.max(np.abs(scores))
        else:
            scores_norm = scores

        tm_regions = approximate_tm_regions(seq)
        regions = [classify_residue_region(pos, tm_regions) for pos in range(L)]

        for pos in range(L):
            region_stats[regions[pos]]["count"] += 1
            region_stats[regions[pos]]["sum_abs_score"] += abs(float(scores_norm[pos]))

        heatmap_data.append({
            "uid": uid,
            "label": int(y[i]),
            "sequence_length": L,
            "scores": [round(float(s), 4) for s in scores_norm],
            "regions": regions,
        })

    print(f"[OK] 生成热图数据: {len(heatmap_data)} 条序列")

    # 区域平均重要性
    print("\n--- 区域平均 |伪热图分数| (Top {} GPCR dims) ---".format(top_n))
    region_avg = []
    for region, stat in sorted(region_stats.items(), key=lambda x: -x[1]["sum_abs_score"]/max(x[1]["count"],1)):
        avg = stat["sum_abs_score"] / stat["count"]
        region_avg.append({"region": region, "avg_abs_score": round(avg, 4), "residue_count": stat["count"]})
        print(f"  {region:12s} | avg |score| = {avg:.4f} (n_residues={stat['count']})")

    # 保存结果
    results = {
        "n_samples": int(len(y)),
        "n_positive": int(y.sum()),
        "n_negative": int(len(y) - y.sum()),
        "feature_dim": int(X.shape[1]),
        "top_gpcr_dims": [int(d) for d in top_gpcr_dims],
        "top_gpcr_shap": [round(float(gpcr_importance[d]), 6) for d in top_gpcr_dims],
        "top_gnaq_dims": [int(d) for d in top_gnaq_dims],
        "top_gnaq_shap": [round(float(gnaq_importance[d]), 6) for d in top_gnaq_dims],
        "region_importance": region_avg,
        "note": "GPCR residue heatmap is a post-hoc projection: scores[pos] = sum_{d in top_dims} SHAP_i,d * ESM[pos,d]. It does NOT imply the SVM natively attends to specific residues; mean pooling removes spatial resolution.",
    }

    with open(OUTPUT_DIR / "global_shap_summary.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(OUTPUT_DIR / "per_sample_heatmap_projection.json", "w") as f:
        json.dump(heatmap_data, f, indent=2, ensure_ascii=False)

    # 额外保存 numpy 矩阵供 Python 可视化
    np.save(OUTPUT_DIR / "shap_values_class1.npy", sv_class1)
    np.save(OUTPUT_DIR / "mean_abs_shap.npy", mean_abs_sv)

    print(f"\n[OK] 结果已保存到: {OUTPUT_DIR}/")
    print("    - global_shap_summary.json")
    print("    - per_sample_heatmap_projection.json")
    print("    - shap_values_class1.npy")
    print("    - mean_abs_shap.npy")

    # 关键生物学洞察摘要
    print("\n" + "=" * 70)
    print("  生物学洞察摘要")
    print("=" * 70)
    # 检查 ICL2/ICL3 是否位于前列
    icl_scores = {r["region"]: r["avg_abs_score"] for r in region_avg if r["region"].startswith("ICL")}
    tm_scores = {r["region"]: r["avg_abs_score"] for r in region_avg if r["region"].startswith("TM")}
    ntail = next((r for r in region_avg if r["region"] == "N-tail"), None)
    ctail = next((r for r in region_avg if r["region"] == "C-tail"), None)

    print(f"ICL 区域平均重要性: {icl_scores}")
    print(f"TM  区域平均重要性: { {k:v for k,v in list(tm_scores.items())[:5]} }")
    if ntail:
        print(f"N-tail 平均重要性: {ntail['avg_abs_score']:.4f}")
    if ctail:
        print(f"C-tail 平均重要性: {ctail['avg_abs_score']:.4f}")

    # Top 3 区域
    top3 = [r["region"] for r in region_avg[:3]]
    print(f"\nTop-3 最重要区域: {', '.join(top3)}")
    if any("ICL" in r for r in top3):
        print("-> ICL 区域（特别是 ICL2/ICL3）在 SHAP 投影中显现为高重要性，与已知 G 蛋白结合界面相符。")
    else:
        print("-> 未观察到 ICL 区域在 SHAP 投影中显著突出；可能受限于 mean pooling 的空间信息丢失。")


if __name__ == "__main__":
    main()
