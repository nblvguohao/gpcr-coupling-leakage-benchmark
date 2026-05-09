#!/usr/bin/env python3
"""
增强版配对模型交叉验证 V2 (650M ESM-2):
1. 加入 GPCR 家族级 Gq 正样本率作为先验特征
2. 加入与人源同源蛋白的 ICL2/3 局部相似度
3. 使用 esm2_t33_650M_UR50D 1280-d 特征重新评估

对比分支:
- baseline_v2    -> 2560-d global concat
- icl_stats_v2   -> 2560 + ICL2/3 stats + family_pos_rate + icl_sim (2578-d)
- icl_full_v2    -> 2560 + ICL2/3 full + family_pos_rate + icl_sim (3618-d)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
import warnings
warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
DATA_DIR = Path(__file__).parent.parent / "data"
GPCR_FEATURES_FILE = BASE / "server_sync" / "extended_data" / "features" / "esm_features_650m_meanpool.json"
G_PROTEIN_FEATURES_FILE = DATA_DIR / "g_protein_esm_features.json"
ICL_FEATURES_FILE = DATA_DIR / "icl_features_650m.json"
ALPHA_FEATURES_FILE = DATA_DIR / "alphafold_icl_features.json"
PAIRING_MATRIX_FILE = DATA_DIR / "pairing_matrix_raw.csv"
CLUSTERS_FILE = DATA_DIR / "sequence_clusters.json"
OUTPUT_FILE = DATA_DIR / "cv_results.json"

# --- 加载家族映射 ---
IUPHAR_FILE = DATA_DIR / "iuphar_coupling_data.csv"


def load_family_map():
    df_iuphar = pd.read_csv(IUPHAR_FILE)
    df_iuphar.columns = [str(c).strip().lower().replace(" ", "_") for c in df_iuphar.columns]
    family_map = {}
    for _, row in df_iuphar.iterrows():
        uniprot = str(row.get("uniprot", "")).strip()
        family = str(row.get("family", "Unknown")).strip()
        if uniprot and family and family.lower() != "nan":
            family_map[uniprot] = family
    return family_map


def compute_family_pos_rate(pairing_df, family_map):
    """计算每个家族的 Gq 正样本率。"""
    gq_data = pairing_df[pairing_df["g_protein_family"] == "Gq"][["gpcr_id", "coupling"]].drop_duplicates()
    family_stats = defaultdict(lambda: [0, 0])
    for _, row in gq_data.iterrows():
        uid = row["gpcr_id"]
        fam = family_map.get(uid, "Unknown")
        family_stats[fam][1] += 1
        family_stats[fam][0] += int(row["coupling"])
    global_rate = gq_data["coupling"].mean()
    rates = {}
    for fam, (pos, total) in family_stats.items():
        rates[fam] = pos / total if total > 0 else global_rate
    rates["Unknown"] = global_rate
    return rates, global_rate


def compute_icl_similarity(gid, icl_data, human_references):
    """计算该 GPCR 的 ICL2/3 与训练集中最近人源同源蛋白的 ICL2/3 相似度。"""
    rec = icl_data.get(gid)
    if rec is None:
        for key in icl_data:
            if "_" in key and key.split("_", 1)[1] == gid:
                rec = icl_data[key]
                break
    if rec is None:
        return 0.0

    def get_icl_seq(r):
        s2 = r.get("ICL2_stats", {})
        s3 = r.get("ICL3_stats", {})
        vec = []
        for k in ["length", "mean_hydro", "std_hydro", "net_charge",
                  "pos_charge_ratio", "neg_charge_ratio", "hydrophobic_ratio", "aromatic_ratio"]:
            vec.append(s2.get(k, 0.0))
            vec.append(s3.get(k, 0.0))
        return np.array(vec)

    test_vec = get_icl_seq(rec)
    if np.linalg.norm(test_vec) == 0:
        return 0.0

    max_sim = 0.0
    for ref_id, ref_rec in human_references.items():
        if ref_id == gid:
            continue
        ref_vec = get_icl_seq(ref_rec)
        if np.linalg.norm(ref_vec) == 0:
            continue
        sim = np.dot(test_vec, ref_vec) / (np.linalg.norm(test_vec) * np.linalg.norm(ref_vec) + 1e-8)
        max_sim = max(max_sim, sim)
    return max_sim


def load_features():
    with open(GPCR_FEATURES_FILE) as f:
        gpcr_raw = json.load(f)
    with open(G_PROTEIN_FEATURES_FILE) as f:
        gprot_raw = json.load(f)

    gpcr_feats = {}
    for k, v in gpcr_raw.items():
        arr = np.array(v)
        gpcr_feats[k] = arr.mean(axis=0) if arr.ndim == 2 else arr

    gprot_feats = {}
    family_map_gprot = {
        "GNAQ": "Gq", "GNAI1": "Gi", "GNAI2": "Gi", "GNAI3": "Gi",
        "GNAS": "Gs", "GNA12": "G12_13", "GNA13": "G12_13",
    }
    for subtype, info in gprot_raw.items():
        family = family_map_gprot.get(subtype, subtype)
        gprot_feats[subtype] = np.array(info["mean_pooling"])
        if family not in gprot_feats:
            gprot_feats[family] = np.array(info["mean_pooling"])

    return gpcr_feats, gprot_feats


def load_icl_features():
    if not ICL_FEATURES_FILE.exists():
        return {}
    with open(ICL_FEATURES_FILE) as f:
        return json.load(f)


def load_alpha_features():
    if not ALPHA_FEATURES_FILE.exists():
        return {}
    with open(ALPHA_FEATURES_FILE) as f:
        return json.load(f)


def get_alpha_vector(alpha_data, gid):
    rec = alpha_data.get(gid)
    if rec is None:
        for key in alpha_data:
            if "_" in key and key.split("_", 1)[1] == gid:
                rec = alpha_data[key]
                break
    keys = [
        "icl2_plddt_mean", "icl2_plddt_std", "icl3_plddt_mean", "icl3_plddt_std",
        "ntail_plddt_mean", "ntail_plddt_std", "ctail_plddt_mean", "ctail_plddt_std",
        "tm_mean_plddt",
        "global_plddt_mean", "global_plddt_std",
        "high_confidence_ratio_70", "high_confidence_ratio_90",
        "sasa_mean", "sasa_buried_ratio",
        "contact_density", "mean_contacts_per_residue",
        # Geometric features
        "tm5_tm6_cyto_ca_distance", "icl2_end_to_end_ca_distance",
        "icl3_end_to_end_ca_distance", "tm5_tm6_cyto_dihedral_angle",
        "icl2_aromatic_centroid_depth", "interface_patch_sasa",
        "interface_patch_sasa_ratio", "icl2_helix_ratio", "icl2_sheet_ratio",
        "icl2_coil_ratio", "icl3_helix_ratio", "icl3_sheet_ratio",
        "icl3_coil_ratio",
        # PAE features
        "icl2_mean_pae", "icl2_intra_pae",
        "icl3_mean_pae", "icl3_intra_pae",
        "icl2_tm5_pae", "icl2_tm6_pae",
        "icl3_tm5_pae", "icl3_tm6_pae",
    ]
    if rec is None:
        return np.zeros(len(keys))
    return np.array([rec.get(k, 0.0) for k in keys])


def get_gpcr_feat(gpcr_feats, gid):
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


def get_icl_vector(icl_data, gid, gpcr_feat_dim=1280):
    rec = icl_data.get(gid)
    if rec is None:
        for key in icl_data:
            if "_" in key and key.split("_", 1)[1] == gid:
                rec = icl_data[key]
                break

    icl2_esm = np.array(rec.get("ICL2_esm", [])) if rec else np.array([])
    icl3_esm = np.array(rec.get("ICL3_esm", [])) if rec else np.array([])
    icl2_stats = rec.get("ICL2_stats", {}) if rec else {}
    icl3_stats = rec.get("ICL3_stats", {}) if rec else {}

    if icl2_esm.size == 0:
        icl2_esm = np.zeros(gpcr_feat_dim)
    if icl3_esm.size == 0:
        icl3_esm = np.zeros(gpcr_feat_dim)

    stat_keys = ["length", "mean_hydro", "std_hydro", "net_charge",
                 "pos_charge_ratio", "neg_charge_ratio", "hydrophobic_ratio", "aromatic_ratio"]
    icl2_stat_vec = np.array([icl2_stats.get(k, 0.0) for k in stat_keys])
    icl3_stat_vec = np.array([icl3_stats.get(k, 0.0) for k in stat_keys])

    return icl2_esm, icl2_stat_vec, icl3_esm, icl3_stat_vec


def build_vectors(df, gpcr_feats, gprot_feats, icl_data, alpha_data, family_map, family_rates, global_rate,
                  human_icl_refs, mode="baseline"):
    X_list, y_list, meta = [], [], []
    missing = defaultdict(int)
    for _, row in df.iterrows():
        gid = row["gpcr_id"]
        gpcr_feat = get_gpcr_feat(gpcr_feats, gid)
        gfam = row["g_protein_family"]
        gprot_feat = gprot_feats.get(gfam)
        if gprot_feat is None:
            gprot_feat = gprot_feats.get(gfam.capitalize()) or gprot_feats.get(gfam.upper())

        if gpcr_feat is None:
            missing[f"gpcr_missing_{gid}"] += 1
            continue
        if gprot_feat is None:
            missing[f"gprot_missing_{gfam}"] += 1
            continue

        base_vec = np.concatenate([gpcr_feat, gprot_feat])
        vec_parts = [base_vec]

        # ICL features
        if mode in ("icl_full", "icl_stats", "icl_only", "icl_full_v2", "icl_stats_v2", "icl_only_v2", "alpha", "alpha_stats"):
            icl2_esm, icl2_stat, icl3_esm, icl3_stat = get_icl_vector(icl_data, gid, gpcr_feat_dim=len(gpcr_feat))
            if mode in ("icl_full", "icl_full_v2", "alpha"):
                icl_vec = np.concatenate([icl2_esm, icl2_stat, icl3_esm, icl3_stat])
                vec_parts.append(icl_vec)
            elif mode in ("icl_stats", "icl_stats_v2", "alpha_stats"):
                icl_vec = np.concatenate([icl2_stat, icl3_stat])
                vec_parts.append(icl_vec)
            else:
                icl_vec = np.concatenate([icl2_esm, icl2_stat, icl3_esm, icl3_stat])
                vec_parts = [icl_vec]

        # AlphaFold features
        if mode in ("alpha", "alpha_stats"):
            vec_parts.append(get_alpha_vector(alpha_data, gid))

        # 新增 V2 特征
        if mode.endswith("_v2"):
            fam = family_map.get(gid, "Unknown")
            fam_rate = family_rates.get(fam, global_rate)
            icl_sim = compute_icl_similarity(gid, icl_data, human_icl_refs)
            vec_parts.append(np.array([fam_rate, icl_sim]))

        vec = np.concatenate(vec_parts)
        X_list.append(vec)
        y_list.append(int(row["coupling"]))
        meta.append({
            "gpcr_id": gid,
            "g_protein_family": gfam,
            "cluster_id": int(row["cluster_id"]),
        })

    if missing:
        print(f"[WARN] 缺失特征统计 (去重后): {dict(missing)}")
    return np.array(X_list), np.array(y_list), meta


def evaluate_svm(X_train, y_train, X_test, y_test, kernel="rbf", C=10.0,
                 class_weight="balanced", tune_threshold=False,
                 threshold_val_size=0.15, random_state=42):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)
    svm = SVC(kernel=kernel, C=C, class_weight=class_weight,
              probability=True, random_state=random_state)
    svm.fit(Xtr, y_train)

    if tune_threshold:
        X_tr_sub, X_val, y_tr_sub, y_val = train_test_split(
            Xtr, y_train, test_size=threshold_val_size,
            random_state=random_state, stratify=y_train)
        svm_val = SVC(kernel=kernel, C=C, class_weight=class_weight,
                      probability=True, random_state=random_state)
        svm_val.fit(X_tr_sub, y_tr_sub)
        y_val_proba = svm_val.predict_proba(X_val)[:, 1]
        thresholds = np.sort(np.unique(y_val_proba))
        if len(thresholds) == 0:
            thresholds = np.array([0.5])
        else:
            thresholds = np.concatenate([[0.0], thresholds, [1.0]])
        best_f1 = -1.0
        best_thresh = 0.5
        for thr in thresholds:
            y_val_pred = (y_val_proba >= thr).astype(int)
            if np.sum(y_val_pred) == 0 or np.sum(y_val_pred) == len(y_val_pred):
                continue
            f1 = f1_score(y_val, y_val_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thr
        y_proba = svm.predict_proba(Xte)[:, 1]
        y_pred = (y_proba >= best_thresh).astype(int)
    else:
        y_proba = svm.predict_proba(Xte)[:, 1]
        y_pred = svm.predict(Xte)

    metrics = {
        "auc": roc_auc_score(y_test, y_proba) if len(set(y_test)) >= 2 else float("nan"),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }
    return metrics, svm, scaler


def experiment_random_cv(X, y, n_splits=5):
    results = {}
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_aucs = []
    fold_accs = []
    for train_idx, test_idx in skf.split(X, y):
        m, _, _ = evaluate_svm(X[train_idx], y[train_idx], X[test_idx], y[test_idx])
        fold_aucs.append(m["auc"])
        fold_accs.append(m["accuracy"])
    results["SVM-RBF C=10 balanced"] = {
        "auc_mean": round(float(np.nanmean(fold_aucs)), 4),
        "auc_std": round(float(np.nanstd(fold_aucs)), 4),
        "acc_mean": round(float(np.nanmean(fold_accs)), 4),
    }
    return results


def experiment_cluster_cv(X, y, meta, cluster_list, n_folds=5):
    n = len(y)
    sample_to_cluster = {i: meta[i]["cluster_id"] for i in range(n)}

    cluster_sizes = defaultdict(int)
    cluster_pos = defaultdict(int)
    for i in range(n):
        cid = sample_to_cluster[i]
        cluster_sizes[cid] += 1
        cluster_pos[cid] += y[i]

    # Greedy bin-packing
    folds = [[] for _ in range(n_folds)]
    fold_sizes = [0] * n_folds
    fold_pos = [0] * n_folds
    sorted_clusters = sorted(cluster_list, key=lambda c: len(c["members"]), reverse=True)

    for c in sorted_clusters:
        cid = c["cluster_id"]
        idx = min(range(n_folds), key=lambda i: fold_sizes[i])
        folds[idx].append(cid)
        fold_sizes[idx] += cluster_sizes[cid]
        fold_pos[idx] += cluster_pos[cid]

    fold_aucs = []
    fold_accs = []
    for f_idx in range(n_folds):
        test_clusters = set(folds[f_idx])
        train_idx = [i for i in range(n) if sample_to_cluster[i] not in test_clusters]
        test_idx = [i for i in range(n) if sample_to_cluster[i] in test_clusters]
        m, _, _ = evaluate_svm(X[train_idx], y[train_idx], X[test_idx], y[test_idx])
        fold_aucs.append(m["auc"])
        fold_accs.append(m["accuracy"])

    return {
        "auc_mean": round(float(np.nanmean(fold_aucs)), 4),
        "auc_std": round(float(np.nanstd(fold_aucs)), 4),
        "acc_mean": round(float(np.nanmean(fold_accs)), 4),
    }


def experiment_logpso(X, y, meta, tune_threshold=False):
    families = ["Gq", "Gi", "Gs", "G12_13"]
    results = {}
    for fam in families:
        train_idx = [i for i in range(len(y)) if meta[i]["g_protein_family"] != fam]
        test_idx = [i for i in range(len(y)) if meta[i]["g_protein_family"] == fam]
        m, _, _ = evaluate_svm(X[train_idx], y[train_idx], X[test_idx], y[test_idx],
                               tune_threshold=tune_threshold)
        results[fam] = m
    return results


def main():
    print("=" * 70)
    print("  增强版配对模型交叉验证 V2 (650M ESM-2)")
    print("=" * 70)

    gpcr_feats, gprot_feats = load_features()
    icl_data = load_icl_features()
    alpha_data = load_alpha_features()
    df = pd.read_csv(PAIRING_MATRIX_FILE)
    family_map = load_family_map()
    family_rates, global_rate = compute_family_pos_rate(df, family_map)

    with open(CLUSTERS_FILE) as f:
        clusters_data = json.load(f)
    cluster_list = clusters_data["clusters"]

    # 构建人源ICL参考集
    human_icl_refs = {}
    for gid, rec in icl_data.items():
        base = gid.split("_", 1)[1] if "_" in gid and len(gid.split("_")[0]) <= 2 else gid
        df_match = pd.read_csv(IUPHAR_FILE)
        df_match.columns = [str(c).strip().lower().replace(" ", "_") for c in df_match.columns]
        match = df_match[(df_match["uniprot"] == base) & (df_match["entryname"].str.endswith("_HUMAN", na=False))]
        if len(match) > 0 or base in family_map:
            human_icl_refs[gid] = rec

    print(f"[INFO] 人源ICL参考集大小: {len(human_icl_refs)}")
    print(f"[INFO] 全局Gq正样本率: {global_rate:.3f}")
    print(f"[INFO] AlphaFold 特征条目: {len(alpha_data)}")

    all_results = {}
    modes = ["baseline", "icl_stats", "icl_full", "icl_stats_v2", "icl_full_v2"]
    if alpha_data:
        modes += ["alpha", "alpha_stats"]

    for mode in modes:
        print(f"\n{'='*70}")
        print(f"  模式: {mode}")
        print(f"{'='*70}")
        X, y, meta = build_vectors(df, gpcr_feats, gprot_feats, icl_data, alpha_data,
                                   family_map, family_rates, global_rate,
                                   human_icl_refs, mode=mode)
        print(f"[INFO] 特征维度: {X.shape[1]}, 样本数: {X.shape[0]}")

        rand_cv = experiment_random_cv(X, y)
        print(f"[Random CV]     AUC={rand_cv['SVM-RBF C=10 balanced']['auc_mean']} ± {rand_cv['SVM-RBF C=10 balanced']['auc_std']}")

        cluster_cv = experiment_cluster_cv(X, y, meta, cluster_list)
        print(f"[Cluster CV]    AUC={cluster_cv['auc_mean']} ± {cluster_cv['auc_std']}")

        logpso = experiment_logpso(X, y, meta, tune_threshold=False)
        logpso_aucs = [logpso[f]["auc"] for f in ["Gq", "Gi", "Gs", "G12_13"]]
        print(f"[LOGPSO]        AUC={np.mean(logpso_aucs):.3f} (Gq={logpso['Gq']['auc']:.3f}, Gi={logpso['Gi']['auc']:.3f}, Gs={logpso['Gs']['auc']:.3f}, G12_13={logpso['G12_13']['auc']:.3f})")

        all_results[mode] = {
            "random_cv": rand_cv,
            "cluster_cv": cluster_cv,
            "logpso": logpso,
        }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[OK] 结果保存至 {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
