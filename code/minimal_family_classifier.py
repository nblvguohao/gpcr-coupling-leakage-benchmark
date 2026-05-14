#!/usr/bin/env python3
"""
Minimal family-conditioned classifier: GPCR ESM-2 650M + ICL + 4-d family onehot.

This is the simplest possible classifier that uses explicit G protein family identity
as the sole G-protein-side signal. It serves as a critical baseline for judging
whether model complexity (cross-attention, deep FFNs) contributes meaningfully
beyond a simple linear/nonlinear classifier conditioned on family identity.

Models:
  - Logistic Regression (L2, C tuned via inner 3-fold CV)
  - MLP (3-layer, 256-128-1, GELU/BN/Dropout)
  - SVM (RBF kernel, C=10, class_weight='balanced')

All evaluated under cluster-aware 5-fold CV (matching main protocol).
"""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from collections import defaultdict
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              brier_score_loss)
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
DATA_DIR = BASE.parent / "data"

GPCR_FEATURES_FILE = DATA_DIR / "gpcr_esm_features_650m.json"
ICL_FEATURES_FILE = DATA_DIR / "icl_features_650m.json"
PAIRING_MATRIX_FILE = DATA_DIR / "pairing_matrix_raw.csv"
CLUSTERS_FILE = DATA_DIR / "sequence_clusters.json"
OUTPUT_FILE = DATA_DIR / "minimal_family_classifier_results.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_FOLDS = 5
N_REPEATS = 5  # repeated CV with different fold assignments
RANDOM_SEED = 42
FAMILIES = ["Gq", "Gi", "Gs", "G12_13"]


# ==========================================================================
# Feature loading (reuse patterns from gprot_onehot_ablation.py)
# ==========================================================================

def load_gpcr_features():
    with open(GPCR_FEATURES_FILE) as f:
        return {k: np.array(v) for k, v in json.load(f).items()}


def load_icl_features():
    if not ICL_FEATURES_FILE.exists():
        return {}
    with open(ICL_FEATURES_FILE) as f:
        return json.load(f)


def get_gpcr_feat(gpcr_feats, gid):
    feat = gpcr_feats.get(gid)
    if feat is None and "_" in gid and len(gid.split("_")[0]) <= 2:
        feat = gpcr_feats.get(gid.split("_", 1)[1])
    if feat is None:
        for key in gpcr_feats:
            if "_" in key and key.split("_", 1)[1] == gid:
                return gpcr_feats[key]
    return feat


def get_icl_vector(icl_data, gid, dim=1280):
    rec = icl_data.get(gid)
    if rec is None:
        for key in icl_data:
            if "_" in key and key.split("_", 1)[1] == gid:
                rec = icl_data[key]; break
    icl2_esm = np.array(rec.get("ICL2_esm", [])) if rec else np.array([])
    icl3_esm = np.array(rec.get("ICL3_esm", [])) if rec else np.array([])
    icl2_stats = rec.get("ICL2_stats", {}) if rec else {}
    icl3_stats = rec.get("ICL3_stats", {}) if rec else {}
    if icl2_esm.size == 0: icl2_esm = np.zeros(dim)
    if icl3_esm.size == 0: icl3_esm = np.zeros(dim)
    sk = ["length","mean_hydro","std_hydro","net_charge","pos_charge_ratio",
          "neg_charge_ratio","hydrophobic_ratio","aromatic_ratio"]
    s2 = np.array([icl2_stats.get(k, 0.0) for k in sk])
    s3 = np.array([icl3_stats.get(k, 0.0) for k in sk])
    return icl2_esm, s2, icl3_esm, s3


def build_vectors(df, gpcr_feats, icl_data):
    """
    Build feature vectors: [GPCR global 1280 | ICL full 2576 | family onehot 4] = 3860-d
    G protein side is a 4-d family one-hot vector (NOT the 1280-d embedding).
    """
    X_list, y_list, metas = [], [], []
    for _, row in df.iterrows():
        gid = row["gpcr_id"]
        gpcr_f = get_gpcr_feat(gpcr_feats, gid)
        gf = row["g_protein_family"]
        gprot_f = np.array([1.0 if gf == f else 0.0 for f in FAMILIES], dtype=np.float64)

        if gpcr_f is None: continue

        i2_e, i2_s, i3_e, i3_s = get_icl_vector(icl_data, gid, 1280)
        parts = [gpcr_f, gprot_f, i2_e, i2_s, i3_e, i3_s]
        X_list.append(np.concatenate(parts))
        y_list.append(int(row["coupling"]))
        metas.append({
            "gpcr_id": gid, "family": gf,
            "cluster_id": int(row["cluster_id"]),
        })
    return np.array(X_list), np.array(y_list), metas


# ==========================================================================
# Cluster-aware fold assignment (same as other scripts)
# ==========================================================================

def get_cluster_folds(metas, cluster_list, n_folds=5, seed=42):
    """Greedy bin-packing of clusters into folds."""
    np.random.seed(seed)
    n = len(metas)
    s2c = {i: metas[i]["cluster_id"] for i in range(n)}
    c_sizes = defaultdict(int)
    for i in range(n): c_sizes[s2c[i]] += 1

    # Shuffle clusters before bin-packing to vary fold assignments
    shuffled = sorted(cluster_list, key=lambda c: (len(c["members"]), c["cluster_id"]),
                      reverse=True)
    fold_cids = [[] for _ in range(n_folds)]
    fold_size = [0.0] * n_folds
    for c in shuffled:
        cid = c["cluster_id"]
        if c_sizes.get(cid, 0) == 0: continue
        target = int(np.argmin(fold_size))
        fold_cids[target].append(cid)
        fold_size[target] += c_sizes[cid]
    return s2c, fold_cids


# ==========================================================================
# MLP Model (same architecture as main paper)
# ==========================================================================

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=(256, 128), dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h), nn.BatchNorm1d(h), nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class VecDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_epoch(model, loader, optim, crit):
    model.train()
    total = 0.0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optim.zero_grad()
        loss = crit(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        total += loss.item() * len(y)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate_mlp(model, loader):
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        x = x.to(DEVICE)
        probs.append(torch.sigmoid(model(x)).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


# ==========================================================================
# Evaluation helpers
# ==========================================================================

def compute_metrics(y_true, y_prob):
    """AUC, PRAUC, Brier score."""
    auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) >= 2 else float("nan")
    prauc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    return {"auc": auc, "prauc": prauc, "brier": brier}


def per_family_metrics(y_true, y_prob, families_arr):
    """Compute metrics per G protein family."""
    results = {}
    for fam in FAMILIES:
        mask = np.array([f == fam for f in families_arr])
        if mask.sum() == 0 or len(set(y_true[mask])) < 2:
            results[fam] = {"n": int(mask.sum()), "auc": float("nan"),
                            "prauc": float("nan"), "brier": float("nan")}
        else:
            m = compute_metrics(y_true[mask], y_prob[mask])
            m["n"] = int(mask.sum())
            results[fam] = m
    return results


# ==========================================================================
# Logistic Regression with inner CV
# ==========================================================================

def run_logistic_cv(X, y, s2c, fold_cids, Cs=None):
    """Logistic regression with L2 penalty, C tuned via inner 3-fold CV."""
    if Cs is None:
        Cs = [0.01, 0.1, 1.0, 10.0, 100.0]
    fold_results = []
    for fi in range(N_FOLDS):
        test_cids = set(fold_cids[fi])
        te_idx = [i for i in range(len(y)) if s2c[i] in test_cids]
        tr_idx = [i for i in range(len(y)) if s2c[i] not in test_cids]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr_idx]); X_te = scaler.transform(X[te_idx])

        # Inner CV for C tuning on training set only
        try:
            lr = LogisticRegressionCV(Cs=Cs, cv=3, scoring="roc_auc",
                                       class_weight="balanced",
                                       max_iter=2000, random_state=RANDOM_SEED,
                                       n_jobs=1)
            lr.fit(X_tr, y[tr_idx])
        except Exception:
            lr = LogisticRegression(C=1.0, class_weight="balanced",
                                     max_iter=2000, random_state=RANDOM_SEED)
            lr.fit(X_tr, y[tr_idx])

        p = lr.predict_proba(X_te)[:, 1]
        fold_results.append({"probs": p, "labels": y[te_idx],
                              "te_idx": te_idx, "best_C": float(lr.C_[0]) if hasattr(lr, "C_") else 1.0})
    return fold_results


def run_svm_cv(X, y, s2c, fold_cids):
    """SVM RBF with fixed C=10 (matching main protocol)."""
    fold_results = []
    for fi in range(N_FOLDS):
        test_cids = set(fold_cids[fi])
        te_idx = [i for i in range(len(y)) if s2c[i] in test_cids]
        tr_idx = [i for i in range(len(y)) if s2c[i] not in test_cids]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr_idx]); X_te = scaler.transform(X[te_idx])

        svm = SVC(kernel="rbf", C=10.0, class_weight="balanced",
                  probability=True, random_state=RANDOM_SEED)
        svm.fit(X_tr, y[tr_idx])
        p = svm.predict_proba(X_te)[:, 1]
        fold_results.append({"probs": p, "labels": y[te_idx], "te_idx": te_idx})
    return fold_results


def run_mlp_cv(X, y, s2c, fold_cids):
    """3-layer MLP (same architecture as paper)."""
    fold_results = []
    for fi in range(N_FOLDS):
        test_cids = set(fold_cids[fi])
        te_idx = [i for i in range(len(y)) if s2c[i] in test_cids]
        tr_idx = [i for i in range(len(y)) if s2c[i] not in test_cids]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr_idx]); X_te = scaler.transform(X[te_idx])

        tr_ds = VecDataset(X_tr, y[tr_idx])
        te_ds = VecDataset(X_te, y[te_idx])
        tr_ld = DataLoader(tr_ds, batch_size=32, shuffle=True)
        te_ld = DataLoader(te_ds, batch_size=32)

        model = MLPClassifier(input_dim=X.shape[1]).to(DEVICE)
        pw = len(y[tr_idx]) / max(1, y[tr_idx].sum())
        optim = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        crit = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([pw]).to(DEVICE))

        best_auc, best_p, patience = -1.0, None, 0
        for ep in range(200):
            train_epoch(model, tr_ld, optim, crit)
            p, lbl = evaluate_mlp(model, te_ld)
            auc = roc_auc_score(lbl, p) if len(set(lbl)) >= 2 else -1
            if auc > best_auc:
                best_auc, best_p, patience = auc, p, 0
            else:
                patience += 1
            if patience >= 20: break
        fold_results.append({"probs": best_p, "labels": y[te_idx], "te_idx": te_idx})
    return fold_results


def aggregate_fold_results(fold_results, metas):
    """Compute aggregated metrics from fold results."""
    all_probs = np.concatenate([fr["probs"] for fr in fold_results])
    all_labels = np.concatenate([fr["labels"] for fr in fold_results])
    all_te_idx = np.concatenate([fr["te_idx"] for fr in fold_results])

    # Reorder to original index order
    sort_idx = np.argsort(all_te_idx)
    all_probs = all_probs[sort_idx]
    all_labels = all_labels[sort_idx]

    fold_aucs = []
    fold_praucs = []
    for fr in fold_results:
        if len(set(fr["labels"])) >= 2:
            fold_aucs.append(roc_auc_score(fr["labels"], fr["probs"]))
            fold_praucs.append(average_precision_score(fr["labels"], fr["probs"]))

    # Per-family metrics (from reordered predictions)
    families_arr = [metas[i]["family"] for i in all_te_idx]
    families_arr = [families_arr[i] for i in sort_idx]
    pf = per_family_metrics(all_labels, all_probs, families_arr)

    overall = compute_metrics(all_labels, all_probs)
    overall["auc_mean"] = float(np.mean(fold_aucs))
    overall["auc_std"] = float(np.std(fold_aucs))
    overall["prauc_mean"] = float(np.mean(fold_praucs))
    overall["prauc_std"] = float(np.std(fold_praucs))
    overall["fold_aucs"] = [float(a) for a in fold_aucs]

    return overall, pf


# ==========================================================================
# Repeated CV with bootstrap CI
# ==========================================================================

def run_repeated_cv(X, y, metas, cluster_list, model_fn, model_name, n_repeats=5):
    """Run repeated cluster-aware CV with different fold assignments per repeat."""
    all_repeat_aucs = []
    all_repeat_praucs = []
    all_repeat_briers = []

    for rep in range(n_repeats):
        seed = RANDOM_SEED + rep * 100
        s2c, fold_cids = get_cluster_folds(metas, cluster_list, n_folds=N_FOLDS, seed=seed)
        fold_results = model_fn(X, y, s2c, fold_cids)

        all_probs = np.concatenate([fr["probs"] for fr in fold_results])
        all_labels = np.concatenate([fr["labels"] for fr in fold_results])

        rep_aucs = [roc_auc_score(fr["labels"], fr["probs"])
                     for fr in fold_results if len(set(fr["labels"])) >= 2]
        rep_praucs = [average_precision_score(fr["labels"], fr["probs"])
                       for fr in fold_results if len(set(fr["labels"])) >= 2]
        rep_brier = brier_score_loss(all_labels, all_probs)

        all_repeat_aucs.extend(rep_aucs)
        all_repeat_praucs.extend(rep_praucs)
        all_repeat_briers.append(rep_brier)

    return {
        "auc_mean": float(np.mean(all_repeat_aucs)),
        "auc_std": float(np.std(all_repeat_aucs)),
        "auc_95ci_low": float(np.percentile(all_repeat_aucs, 2.5)),
        "auc_95ci_high": float(np.percentile(all_repeat_aucs, 97.5)),
        "prauc_mean": float(np.mean(all_repeat_praucs)),
        "prauc_std": float(np.std(all_repeat_praucs)),
        "brier_mean": float(np.mean(all_repeat_briers)),
        "brier_std": float(np.std(all_repeat_briers)),
        "n_repeats": n_repeats,
        "n_total_folds": n_repeats * N_FOLDS,
    }


# ==========================================================================
# Main
# ==========================================================================

def main():
    print("=" * 70)
    print("  Minimal Family-Conditioned Classifier Benchmark")
    print("  LR / MLP / SVM with GPCR ESM-2 650M + ICL + Family One-Hot")
    print("=" * 70)

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)

    # Load data
    gpcr_feats = load_gpcr_features()
    icl_data = load_icl_features()
    df = pd.read_csv(PAIRING_MATRIX_FILE)
    df = df.dropna(subset=["cluster_id"]).copy()

    with open(CLUSTERS_FILE) as f:
        cluster_list = json.load(f)["clusters"]

    print(f"  GPCR features: {len(gpcr_feats)}")
    print(f"  ICL features: {len(icl_data)}")
    print(f"  Pairs: {len(df)}")

    # Build feature vectors (family one-hot, not G protein embedding)
    X, y, metas = build_vectors(df, gpcr_feats, icl_data)
    print(f"  Feature dim: {X.shape[1]} (1280 GPCR + 2576 ICL + 4 family onehot)")
    print(f"  Samples: {len(y)}, Pos ratio: {y.mean():.4f}")

    # Reference values from paper
    print(f"\n  Reference: CA 650M + ICL (full Gprot embedding) = 0.862 +/- 0.025")
    print(f"  Reference: CA 650M + ICL (Gprot onehot)     = 0.855 +/- 0.018")
    print(f"  Reference: MLP 650M + ICL (full Gprot)       = 0.861 +/- 0.023")
    print(f"  Reference: SVM 650M + ICL (full Gprot)       = 0.832 +/- 0.014")

    results = {}

    # ---------- Logistic Regression ----------
    print(f"\n{'='*70}")
    print(f"  Logistic Regression (L2, C tuned via inner 3-fold CV)")
    print(f"{'='*70}")
    lr_res = run_repeated_cv(X, y, metas, cluster_list, run_logistic_cv,
                              "LogisticRegression", n_repeats=N_REPEATS)
    results["logistic_regression"] = lr_res
    print(f"  AUC = {lr_res['auc_mean']:.4f} [{lr_res['auc_95ci_low']:.4f}, {lr_res['auc_95ci_high']:.4f}]")
    print(f"  PRAUC = {lr_res['prauc_mean']:.4f}, Brier = {lr_res['brier_mean']:.4f}")

    # ---------- SVM ----------
    print(f"\n{'='*70}")
    print(f"  SVM (RBF, C=10, class_weight=balanced)")
    print(f"{'='*70}")
    svm_res = run_repeated_cv(X, y, metas, cluster_list, run_svm_cv,
                               "SVM", n_repeats=N_REPEATS)
    results["svm_rbf"] = svm_res
    print(f"  AUC = {svm_res['auc_mean']:.4f} [{svm_res['auc_95ci_low']:.4f}, {svm_res['auc_95ci_high']:.4f}]")
    print(f"  PRAUC = {svm_res['prauc_mean']:.4f}, Brier = {svm_res['brier_mean']:.4f}")

    # ---------- MLP ----------
    print(f"\n{'='*70}")
    print(f"  MLP (3-layer, 256-128-1, GELU/BN/Dropout)")
    print(f"{'='*70}")
    mlp_res = run_repeated_cv(X, y, metas, cluster_list, run_mlp_cv,
                               "MLP", n_repeats=N_REPEATS)
    results["mlp"] = mlp_res
    print(f"  AUC = {mlp_res['auc_mean']:.4f} [{mlp_res['auc_95ci_low']:.4f}, {mlp_res['auc_95ci_high']:.4f}]")
    print(f"  PRAUC = {mlp_res['prauc_mean']:.4f}, Brier = {mlp_res['brier_mean']:.4f}")

    # ---------- Summary ----------
    print(f"\n{'='*70}")
    print(f"  SUMMARY: Minimal Family-Conditioned Classifiers")
    print(f"{'='*70}")
    print(f"  {'Model':<25s} {'AUC':>10s} {'95% CI':>20s} {'PRAUC':>10s} {'Brier':>10s}")
    print(f"  {'-'*75}")
    for name, r in results.items():
        print(f"  {name:<25s} {r['auc_mean']:>10.4f} [{r['auc_95ci_low']:.4f}, {r['auc_95ci_high']:.4f}] "
              f"{r['prauc_mean']:>10.4f} {r['brier_mean']:>10.4f}")

    print(f"\n  Comparison with paper's paired models (with full Gprot embedding):")
    print(f"  {'Model':<25s} {'AUC':>10s}")
    print(f"  {'CA 650M + ICL (full Gprot)':<25s} {'0.862':>10s}")
    print(f"  {'MLP 650M + ICL (full Gprot)':<25s} {'0.861':>10s}")
    print(f"  {'SVM 650M + ICL (full Gprot)':<25s} {'0.832':>10s}")

    # Save
    out = {
        "description": (
            "Minimal family-conditioned classifiers: GPCR ESM-2 650M (1280-d) "
            "+ ICL full (2576-d) + 4-d family onehot. "
            "These are the simplest models that use explicit family identity "
            "as the sole G-protein-side signal."
        ),
        "feature_dim": int(X.shape[1]),
        "n_samples": int(len(y)),
        "positive_ratio": float(y.mean()),
        "evaluation": f"{N_REPEATS}x repeated cluster-aware 5-fold CV",
        "reference_ca_650m_icl_full_gprot": 0.862,
        "reference_mlp_650m_icl_full_gprot": 0.861,
        "reference_svm_650m_icl_full_gprot": 0.832,
        "reference_ca_650m_icl_onehot_gprot": 0.855,
        "results": results,
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
