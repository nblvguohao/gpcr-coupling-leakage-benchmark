#!/usr/bin/env python3
"""
Classical baselines for GPCR-G protein coupling prediction.
- k-NN: nearest-neighbor classifier based on GPCR sequence identity
- AAC+SVM: amino acid composition + RBF SVM (traditional baseline)
- Global ESM-2 k-NN: nearest-neighbor using ESM-2 embedding cosine similarity

Answers reviewer concern #7: missing classical GPCR coupling prediction baselines.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
DATA_DIR = BASE.parent / "data"
PAIRING_MATRIX_FILE = DATA_DIR / "pairing_matrix_raw.csv"
GPCR_FEATURES_FILE = DATA_DIR / "gpcr_esm_features_650m.json"
CLUSTERS_FILE = DATA_DIR / "sequence_clusters.json"

N_FOLDS = 5
RANDOM_SEED = 42

# Standard 20 amino acids
AA_LETTERS = "ACDEFGHIKLMNPQRSTVWY"


def aa_composition(seq):
    """Compute 20-d amino acid composition vector."""
    if not isinstance(seq, str) or len(seq) == 0:
        return np.zeros(20)
    counts = np.zeros(20)
    for aa in seq:
        idx = AA_LETTERS.find(aa.upper())
        if idx >= 0:
            counts[idx] += 1
    total = counts.sum()
    return counts / total if total > 0 else counts


def load_data():
    df = pd.read_csv(PAIRING_MATRIX_FILE)

    with open(GPCR_FEATURES_FILE) as f:
        gpcr_raw = json.load(f)
    gpcr_feats = {}
    for k, v in gpcr_raw.items():
        arr = np.array(v)
        gpcr_feats[k] = arr.mean(axis=0) if arr.ndim == 2 else arr

    with open(CLUSTERS_FILE) as f:
        clusters_data = json.load(f)
    cluster_list = clusters_data["clusters"]

    return df, gpcr_feats, cluster_list


def get_gpcr_base(gid):
    if "_" in gid and len(gid.split("_")[0]) <= 2:
        return gid.split("_", 1)[1]
    return gid


def get_cluster_folds(df, cluster_list, n_folds=5):
    """Create cluster-aware fold assignments."""
    sample_clusters = []
    for _, row in df.iterrows():
        try:
            cid = int(row["cluster_id"])
        except (ValueError, TypeError):
            cid = -1  # unassigned — put in its own singleton group
        sample_clusters.append(cid)

    cluster_sizes = defaultdict(int)
    for c in sample_clusters:
        cluster_sizes[c] += 1

    # Greedy bin-packing (same as original)
    fold_cids = [[] for _ in range(n_folds)]
    fold_size = [0] * n_folds
    sorted_cids = sorted(cluster_list, key=lambda c: len(c["members"]), reverse=True)
    for c in sorted_cids:
        cid = c["cluster_id"]
        csize = cluster_sizes.get(cid, 0)
        if csize == 0:
            continue
        target = int(np.argmin(fold_size))
        fold_cids[target].append(cid)
        fold_size[target] += csize

    # Assign unclustered (-1) samples round-robin
    if cluster_sizes.get(-1, 0) > 0:
        unclustered_count = cluster_sizes[-1]
        for i in range(unclustered_count):
            target = int(np.argmin(fold_size))
            fold_cids[target].append(-(i+1000))  # unique ID
            fold_size[target] += 1

    # Map sample indices to folds
    fold_idx = []
    for c in sample_clusters:
        for fi, cids in enumerate(fold_cids):
            if c in cids:
                fold_idx.append(fi)
                break
        else:
            fold_idx.append(-1)

    return fold_idx, fold_cids


def evaluate_baseline(X, y, fold_idx, model, model_name, scale=True):
    """Cluster-aware 5-fold CV for a given model."""
    X = np.array(X)
    y = np.array(y)
    fold_aucs = []
    all_probs = np.full(len(y), np.nan)

    for fi in range(N_FOLDS):
        test_mask = np.array([f == fi for f in fold_idx])
        train_mask = ~test_mask

        if scale:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[train_mask])
            X_te = scaler.transform(X[test_mask])
        else:
            X_tr, X_te = X[train_mask], X[test_mask]

        model_clone = model if hasattr(model, 'fit') else None
        if isinstance(model, type):  # sklearn class
            model_instance = model()
        else:
            from copy import deepcopy
            model_instance = deepcopy(model)

        model_instance.fit(X_tr, y[train_mask])

        if hasattr(model_instance, 'predict_proba'):
            y_prob = model_instance.predict_proba(X_te)[:, 1]
        else:
            y_prob = model_instance.predict(X_te)

        if len(set(y[test_mask])) >= 2:
            auc = roc_auc_score(y[test_mask], y_prob)
            fold_aucs.append(auc)
        all_probs[test_mask] = y_prob

    mean_auc = np.mean(fold_aucs) if fold_aucs else float("nan")
    std_auc = np.std(fold_aucs) if fold_aucs else float("nan")
    return mean_auc, std_auc, all_probs


def main():
    print("=" * 70)
    print("  Classical Baselines for GPCR Coupling Prediction")
    print("=" * 70)

    df, gpcr_feats, cluster_list = load_data()
    df = df.dropna(subset=["cluster_id"]).copy()
    print(f"  Dataset: {len(df)} pairs, {len(gpcr_feats)} GPCRs")

    # Create fold assignments
    fold_idx, fold_cids = get_cluster_folds(df, cluster_list, N_FOLDS)
    for fi in range(N_FOLDS):
        n_test = sum(1 for f in fold_idx if f == fi)
        n_train = len(df) - n_test
        print(f"  Fold {fi+1}: train={n_train}, test={n_test}")

    # Build feature matrices
    # 1) Amino Acid Composition (AAC): 20-d per GPCR
    print("\n--- AAC Features ---")
    aac_features = {}
    for _, row in df.iterrows():
        gid = row["gpcr_id"]
        base = get_gpcr_base(gid)
        if base not in aac_features:
            seq = str(row.get("gpcr_sequence", ""))
            aac_features[base] = aa_composition(seq)
        if gid not in aac_features:
            aac_features[gid] = aac_features[base]

    X_aac, y_aac = [], []
    for _, row in df.iterrows():
        gid = row["gpcr_id"]
        gf = row["g_protein_family"]
        gf_onehot = np.array([1.0 if gf == f else 0.0 for f in ["Gq", "Gi", "Gs", "G12_13"]])
        aac = aac_features.get(gid, np.zeros(20))
        X_aac.append(np.concatenate([aac, gf_onehot]))
        y_aac.append(int(row["coupling"]))

    X_aac = np.array(X_aac)
    y_aac = np.array(y_aac)
    print(f"  AAC feature dim: {X_aac.shape[1]}")

    # 2) ESM-2 embedding features for k-NN
    print("\n--- ESM-2 k-NN ---")
    X_esm, y_esm = [], []
    for _, row in df.iterrows():
        gid = row["gpcr_id"]
        gf = row["g_protein_family"]
        gf_onehot = np.array([1.0 if gf == f else 0.0 for f in ["Gq", "Gi", "Gs", "G12_13"]])
        feat = gpcr_feats.get(gid)
        if feat is None:
            base = get_gpcr_base(gid)
            feat = gpcr_feats.get(base)
        if feat is None:
            feat = np.zeros(1280)
        X_esm.append(np.concatenate([feat, gf_onehot]))
        y_esm.append(int(row["coupling"]))
    X_esm = np.array(X_esm)
    y_esm = np.array(y_esm)
    print(f"  ESM-2 feature dim: {X_esm.shape[1]}")

    # ===================================================================
    # Run baselines
    # ===================================================================
    results = {}

    print("\n" + "=" * 70)
    print("  Results (Cluster-CV)")
    print("=" * 70)
    print(f"  {'Method':<30s} {'AUC':>10s} {'Std':>10s}")
    print(f"  {'-'*50}")

    # Baseline 1: AAC + SVM (RBF)
    print("  Running AAC + SVM (RBF)...")
    svm = SVC(kernel="rbf", C=10.0, class_weight="balanced", probability=True,
              random_state=RANDOM_SEED)
    auc, std, _ = evaluate_baseline(X_aac, y_aac, fold_idx, svm, "AAC+SVM")
    results["AAC + RBF SVM"] = {"auc": auc, "std": std}
    print(f"  {'AAC + RBF SVM':<30s} {auc:>10.4f} {std:>10.4f}")

    # Baseline 2: AAC + k-NN (k=5)
    print("  Running AAC + k-NN (k=5)...")
    knn = KNeighborsClassifier(n_neighbors=5)
    auc, std, _ = evaluate_baseline(X_aac, y_aac, fold_idx, knn, "AAC+kNN")
    results["AAC + k-NN (k=5)"] = {"auc": auc, "std": std}
    print(f"  {'AAC + k-NN (k=5)':<30s} {auc:>10.4f} {std:>10.4f}")

    # Baseline 3: ESM-2 + k-NN (k=5)
    print("  Running ESM-2 + k-NN (k=5)...")
    knn_esm = KNeighborsClassifier(n_neighbors=5, metric="cosine")
    auc, std, _ = evaluate_baseline(X_esm, y_esm, fold_idx, knn_esm, "ESM2+kNN")
    results["ESM-2 + k-NN (k=5, cosine)"] = {"auc": auc, "std": std}
    print(f"  {'ESM-2 + k-NN (k=5)':<30s} {auc:>10.4f} {std:>10.4f}")

    # Baseline 4: ESM-2 + k-NN (k=1)
    print("  Running ESM-2 + k-NN (k=1)...")
    knn1 = KNeighborsClassifier(n_neighbors=1, metric="cosine")
    auc, std, _ = evaluate_baseline(X_esm, y_esm, fold_idx, knn1, "ESM2+kNN(k=1)")
    results["ESM-2 + k-NN (k=1, cosine)"] = {"auc": auc, "std": std}
    print(f"  {'ESM-2 + k-NN (k=1)':<30s} {auc:>10.4f} {std:>10.4f}")

    # Baseline 5: AAC + k-NN with varied k
    for k in [3, 7, 10]:
        print(f"  Running AAC + k-NN (k={k})...")
        knn_k = KNeighborsClassifier(n_neighbors=k)
        auc, std, _ = evaluate_baseline(X_aac, y_aac, fold_idx, knn_k, f"AAC+kNN(k={k})")
        results[f"AAC + k-NN (k={k})"] = {"auc": auc, "std": std}
        print(f"  {'AAC + k-NN (k=' + str(k) + ')':<30s} {auc:>10.4f} {std:>10.4f}")

    # Comparison: SVM baseline from paper (for context)
    print(f"\n  {'Paper SVM (8M baseline)':<30s} {'0.8188':>10s} {'0.0167':>10s}")
    print(f"  {'Paper CA (650M + ICL)':<30s} {'0.8619':>10s} {'0.0249':>10s}")

    # Save results
    out = {
        "evaluation": "Cluster-aware 5-fold CV",
        "n_folds": N_FOLDS,
        "results": {k: {"auc_mean": round(v["auc"], 4),
                         "auc_std": round(v["std"], 4)} for k, v in results.items()},
        "reference_svm_8m_baseline": 0.8188,
        "reference_ca_650m_icl": 0.8619,
    }
    with open(DATA_DIR / "classical_baselines.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {DATA_DIR / 'classical_baselines.json'}")


if __name__ == "__main__":
    main()
