#!/usr/bin/env python3
"""
Ablation and Control Experiments Suite
=======================================
Unified entry point for four control experiments and quality checks supporting
the GPCR-G protein coupling prediction paper.

Subcommands
-----------
onehot     : G Protein One-Hot Ablation
             Tests whether full G protein sequence embedding (1280-d) versus
             4-dimensional family one-hot identity vector matters for
             prediction performance.
             Compares three conditions:
               (a) GPCR features only (no G protein info)
               (b) GPCR + 4-d family one-hot (paired with identity only)
               (c) GPCR + 1280-d G protein embedding (full sequence info)
             Models: Cross-Attention network, SVM (RBF).
             Output: ../data/gprot_onehot_ablation.json

dimension  : Dimension Alignment Control Experiments (Point 4 of revision)
             Four minimal control groups testing whether feature dimension
             mismatch degradation stems from feature-scale/representation-space
             incompatibility or genuine dimension alignment requirements.
             Controls (SVM RBF, cluster-aware 5-fold CV):
               C1: PCA projection to 320-d for both GPCR and ICL
               C2: Learned linear projection 320-d ICL to 1280-d
               C3: Block-wise z-score normalization before concatenation
               C4: Zero-padding 320-d ICL to 1280-d
             Output: ../data/dimension_alignment_controls.json

minimal    : Minimal Family-Conditioned Classifier Benchmark
             Trains Logistic Regression / MLP / SVM with GPCR ESM-2 650M
             embedding + ICL features + 4-d family one-hot (no G protein
             embedding) as a baseline. Serves as lower bound for judging
             whether model complexity contributes beyond a simple classifier
             conditioned on family identity.
             Evaluation: 5x repeated cluster-aware 5-fold CV with bootstrap CI.
             Output: ../data/minimal_family_classifier_results.json

label      : Label Audit and Tiered Evaluation
             Annotates each (GPCR, G protein) pair with metadata (species,
             evidence strength, zero-positive GPCR flag) and evaluates SVM
             performance across three label-quality tiers:
               Tier 1: Full dataset (1647 pairs)
               Tier 2: Remove zero-positive GPCRs
               Tier 3: High-confidence only (direct assay + curated negative)
             Output: ../data/label_audit.json

all        : Run all four experiments sequentially.

Usage
-----
    python ablation.py onehot
    python ablation.py dimension
    python ablation.py minimal
    python ablation.py label
    python ablation.py all

All original output file paths, experimental parameters, models, and
evaluation logic are preserved from the individual scripts.
"""

import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from collections import defaultdict
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              brier_score_loss)
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings("ignore")


# ==========================================================================
# Shared Constants
# ==========================================================================

BASE = Path(__file__).parent
DATA_DIR = BASE.parent / "data"

# Shared data files
GPCR_FEATURES_FILE = DATA_DIR / "gpcr_esm_features_650m.json"
GPROT_FEATURES_FILE = DATA_DIR / "g_protein_esm_features_650m.json"
ICL_FEATURES_FILE = DATA_DIR / "icl_features_650m.json"
ICL_320_FILE = DATA_DIR / "icl_features.json"          # 320-d (8M model)
ICL_1280_FILE = DATA_DIR / "icl_features_650m.json"    # 1280-d (650M model)
PAIRING_MATRIX_FILE = DATA_DIR / "pairing_matrix_raw.csv"
CLUSTERS_FILE = DATA_DIR / "sequence_clusters.json"

# Output files (preserved from original scripts)
ONEOUT_OUTPUT = DATA_DIR / "gprot_onehot_ablation.json"
DIMENSION_OUTPUT = DATA_DIR / "dimension_alignment_controls.json"
MINIMAL_OUTPUT = DATA_DIR / "minimal_family_classifier_results.json"
LABEL_OUTPUT = DATA_DIR / "label_audit.json"

# Common parameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_FOLDS = 5
RANDOM_SEED = 42
FAMILIES = ["Gq", "Gi", "Gs", "G12_13"]
FAMILY_MAP_GPROT = {
    "GNAQ": "Gq", "GNAI1": "Gi", "GNAI2": "Gi", "GNAI3": "Gi",
    "GNAS": "Gs", "GNA12": "G12_13", "GNA13": "G12_13",
}


# ==========================================================================
# Shared Utility Functions
# ==========================================================================

def load_gpcr_features():
    """Load GPCR ESM-2 650M features (1280-d global embeddings)."""
    with open(GPCR_FEATURES_FILE) as f:
        return {k: np.array(v) for k, v in json.load(f).items()}


def load_icl_features(path=None):
    """Load ICL features JSON.  Returns {} if file missing."""
    p = path or ICL_FEATURES_FILE
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def get_gpcr_feat(gpcr_feats, gid):
    """Resolve GPCR feature vector with prefix/suffix fallback matching."""
    feat = gpcr_feats.get(gid)
    if feat is None and "_" in gid and len(gid.split("_")[0]) <= 2:
        feat = gpcr_feats.get(gid.split("_", 1)[1])
    if feat is None:
        for key in gpcr_feats:
            if "_" in key and key.split("_", 1)[1] == gid:
                return gpcr_feats[key]
    return feat


def get_icl_vector(icl_data, gid, dim=1280):
    """Extract ICL2/ICL3 embeddings + physicochemical stats for a GPCR."""
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
        icl2_esm = np.zeros(dim)
    if icl3_esm.size == 0:
        icl3_esm = np.zeros(dim)
    sk = ["length", "mean_hydro", "std_hydro", "net_charge",
          "pos_charge_ratio", "neg_charge_ratio",
          "hydrophobic_ratio", "aromatic_ratio"]
    s2 = np.array([icl2_stats.get(k, 0.0) for k in sk])
    s3 = np.array([icl3_stats.get(k, 0.0) for k in sk])
    return icl2_esm, s2, icl3_esm, s3


def get_cluster_folds(metas, cluster_list, n_folds=5, seed=None,
                      tiebreak_by_id=False):
    """
    Greedy bin-packing of sequence clusters into folds for cluster-aware CV.

    Parameters
    ----------
    metas : list of dict
        Must contain 'cluster_id' key per entry.
    cluster_list : list of dict
        Cluster definitions with 'cluster_id' and 'members'.
    n_folds : int
        Number of CV folds.
    seed : int or None
        If provided, seeds numpy RNG (used by minimal-family variant).
    tiebreak_by_id : bool
        If True, sort clusters by (size, cluster_id) to break ties
        deterministically (minimal-family variant).  Otherwise sort by size only.
    """
    if seed is not None:
        np.random.seed(seed)
    n = len(metas)
    s2c = {i: metas[i]["cluster_id"] for i in range(n)}
    c_sizes = defaultdict(int)
    for i in range(n):
        c_sizes[s2c[i]] += 1

    if tiebreak_by_id:
        sort_key = lambda c: (len(c["members"]), c["cluster_id"])
    else:
        sort_key = lambda c: len(c["members"])
    sorted_c = sorted(cluster_list, key=sort_key, reverse=True)

    fold_cids = [[] for _ in range(n_folds)]
    fold_size = [0.0] * n_folds
    for c in sorted_c:
        cid = c["cluster_id"]
        if c_sizes.get(cid, 0) == 0:
            continue
        target = int(np.argmin(fold_size))
        fold_cids[target].append(cid)
        fold_size[target] += c_sizes[cid]
    return s2c, fold_cids


# ==========================================================================
# Subcommand: onehot  --  G Protein One-Hot Ablation
# ==========================================================================

def _onehot_load_gprot_features():
    """Load G protein ESM-2 650M features (1280-d) keyed by subtype + family."""
    with open(GPROT_FEATURES_FILE) as f:
        raw = json.load(f)
    feats = {}
    for subtype, info in raw.items():
        family = FAMILY_MAP_GPROT.get(subtype, subtype)
        vec = np.array(info["mean_pooling"])
        feats[subtype] = vec
        feats[family] = vec
    return feats


def _onehot_build_vectors(df, gpcr_feats, gprot_feats, icl_data,
                          gprot_mode="embedding"):
    """
    Build feature matrices.

    gprot_mode:
        'embedding' -> 1280-d protein embedding
        'onehot'     -> 4-d family one-hot
        'none'       -> no G protein info (multi-task style)
    """
    X_list, y_list, metas = [], [], []
    for _, row in df.iterrows():
        gid = row["gpcr_id"]
        gpcr_f = get_gpcr_feat(gpcr_feats, gid)
        gf = row["g_protein_family"]

        if gprot_mode == "embedding":
            gprot_f = gprot_feats.get(gf)
            if gprot_f is None:
                gprot_f = (gprot_feats.get(gf.capitalize())
                           or gprot_feats.get(gf.upper()))
            if gprot_f is None:
                continue
        elif gprot_mode == "onehot":
            gprot_f = np.array([1.0 if gf == f else 0.0 for f in FAMILIES],
                               dtype=np.float64)
        else:  # "none"
            gprot_f = np.array([])

        if gpcr_f is None:
            continue

        i2_e, i2_s, i3_e, i3_s = get_icl_vector(icl_data, gid, 1280)
        parts = [gpcr_f, gprot_f, i2_e, i2_s, i3_e, i3_s]
        X_list.append(np.concatenate(parts))
        y_list.append(int(row["coupling"]))
        metas.append({"gpcr_id": gid, "family": gf,
                      "cluster_id": int(row["cluster_id"])})

    return np.array(X_list), np.array(y_list), metas


class PairedCANet(nn.Module):
    """Cross-attention network for paired GPCR-G protein prediction."""

    def __init__(self, gpcr_dim=3856, gprot_dim=1280, hidden_dim=256,
                 num_heads=4, dropout=0.3):
        super().__init__()
        self.gpcr_proj = nn.Sequential(
            nn.Linear(gpcr_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.gprot_proj = nn.Sequential(
            nn.Linear(gprot_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, gpcr_feat, gprot_feat):
        q = self.gpcr_proj(gpcr_feat).unsqueeze(1)
        kv = self.gprot_proj(gprot_feat).unsqueeze(1)
        attn_out, _ = self.cross_attn(q, kv, kv)
        x = torch.cat([attn_out.squeeze(1), self.gpcr_proj(gpcr_feat)], dim=-1)
        return self.ffn(x).squeeze(-1)


class PairDataset(Dataset):
    """Dataset that splits concatenated vector into GPCR side and G protein side."""

    def __init__(self, X, y, gprot_dim=1280):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.gprot_dim = gprot_dim

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        gpcr_side = torch.cat([x[:1280], x[1280 + self.gprot_dim:]])
        gprot_vec = x[1280:1280 + self.gprot_dim]
        return gpcr_side, gprot_vec, self.y[idx]


def _onehot_train_epoch(model, loader, optim, crit):
    model.train()
    total = 0.0
    for gpcr, gprot, y in loader:
        gpcr, gprot, y = gpcr.to(DEVICE), gprot.to(DEVICE), y.to(DEVICE)
        optim.zero_grad()
        loss = crit(model(gpcr, gprot), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        total += loss.item() * len(y)
    return total / len(loader.dataset)


@torch.no_grad()
def _onehot_evaluate_ca(model, loader):
    model.eval()
    probs, labels = [], []
    for gpcr, gprot, y in loader:
        gpcr, gprot = gpcr.to(DEVICE), gprot.to(DEVICE)
        probs.append(torch.sigmoid(model(gpcr, gprot)).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def _onehot_run_svm_cv(X, y, s2c, fold_cids):
    fold_aucs = []
    for fi in range(N_FOLDS):
        test_cids = set(fold_cids[fi])
        te_idx = [i for i in range(len(y)) if s2c[i] in test_cids]
        tr_idx = [i for i in range(len(y)) if s2c[i] not in test_cids]
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr_idx])
        X_te = scaler.transform(X[te_idx])
        svm = SVC(kernel="rbf", C=10.0, class_weight="balanced",
                  probability=True, random_state=RANDOM_SEED)
        svm.fit(X_tr, y[tr_idx])
        p = svm.predict_proba(X_te)[:, 1]
        if len(set(y[te_idx])) >= 2:
            fold_aucs.append(roc_auc_score(y[te_idx], p))
    return np.mean(fold_aucs), np.std(fold_aucs)


def _onehot_run_ca_cv(X, y, s2c, fold_cids, gprot_dim):
    fold_aucs = []
    for fi in range(N_FOLDS):
        test_cids = set(fold_cids[fi])
        te_idx = [i for i in range(len(y)) if s2c[i] in test_cids]
        tr_idx = [i for i in range(len(y)) if s2c[i] not in test_cids]
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr_idx])
        X_te = scaler.transform(X[te_idx])

        tr_ds = PairDataset(X_tr, y[tr_idx], gprot_dim)
        te_ds = PairDataset(X_te, y[te_idx], gprot_dim)
        tr_ld = DataLoader(tr_ds, batch_size=32, shuffle=True)
        te_ld = DataLoader(te_ds, batch_size=32)

        gpcr_dim = 1280 + 2576
        model = PairedCANet(gpcr_dim=gpcr_dim, gprot_dim=gprot_dim).to(DEVICE)
        pw = len(y[tr_idx]) / max(1, y[tr_idx].sum())
        optim = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        crit = nn.BCEWithLogitsLoss(
            pos_weight=torch.FloatTensor([pw]).to(DEVICE))

        best_auc, best_p, patience = -1.0, None, 0
        for ep in range(200):
            _onehot_train_epoch(model, tr_ld, optim, crit)
            p, lbl = _onehot_evaluate_ca(model, te_ld)
            auc = roc_auc_score(lbl, p) if len(set(lbl)) >= 2 else -1
            if auc > best_auc:
                best_auc, best_p, patience = auc, p, 0
            else:
                patience += 1
            if patience >= 20:
                break
        fold_aucs.append(best_auc)
    return np.mean(fold_aucs), np.std(fold_aucs)


def run_onehot():
    """G protein one-hot ablation: embedding vs onehot vs no G protein info."""
    print("=" * 70)
    print("  G Protein One-Hot Ablation Experiment")
    print("  Tests: embedding vs onehot vs no G protein info")
    print("=" * 70)

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)

    gpcr_feats = load_gpcr_features()
    gprot_feats = _onehot_load_gprot_features()
    icl_data = load_icl_features()
    df = pd.read_csv(PAIRING_MATRIX_FILE)
    df = df.dropna(subset=["cluster_id"]).copy()

    with open(CLUSTERS_FILE) as f:
        cluster_list = json.load(f)["clusters"]

    print(f"  GPCR features: {len(gpcr_feats)}, "
          f"G protein: {len(gprot_feats)}, Pairs: {len(df)}")

    configs = [
        ("embedding (1280-d)", "embedding", 1280),
        ("onehot (4-d)", "onehot", 4),
    ]

    results = {}
    for name, mode, gdim in configs:
        print(f"\n{'='*70}")
        print(f"  Config: {name}")
        print(f"{'='*70}")
        X, y, metas = _onehot_build_vectors(
            df, gpcr_feats, gprot_feats, icl_data, gprot_mode=mode)
        print(f"  Feature dim: {X.shape[1]}, Samples: {len(y)}, "
              f"Pos ratio: {y.mean():.3f}")

        s2c, fold_cids = get_cluster_folds(metas, cluster_list)

        svm_auc, svm_std = _onehot_run_svm_cv(X, y, s2c, fold_cids)
        print(f"  SVM (RBF):     AUC = {svm_auc:.4f} +/- {svm_std:.4f}")

        ca_auc, ca_std = _onehot_run_ca_cv(X, y, s2c, fold_cids, gdim)
        print(f"  Cross-Attn:   AUC = {ca_auc:.4f} +/- {ca_std:.4f}")

        results[name] = {"svm_auc": svm_auc, "svm_std": svm_std,
                         "ca_auc": ca_auc, "ca_std": ca_std}

    print(f"\n{'='*70}")
    print(f"  Reference (from paper):")
    print(f"{'='*70}")
    print(f"  Multi-task CA (no G protein info): AUC = 0.802 +/- 0.019")
    print(f"  Paired CA 650M + ICL (embedding):  AUC = 0.862 +/- 0.025")

    out = {
        "description": "G protein one-hot vs embedding ablation",
        "multi_task_reference_auc": 0.802,
        "paired_embedding_reference_auc": 0.862,
        "results": {k: {"svm_auc": round(v["svm_auc"], 4),
                        "svm_std": round(v["svm_std"], 4),
                        "ca_auc": round(v["ca_auc"], 4),
                        "ca_std": round(v["ca_std"], 4)}
                    for k, v in results.items()},
    }
    with open(ONEOUT_OUTPUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {ONEOUT_OUTPUT}")


# ==========================================================================
# Subcommand: dimension  --  Dimension Alignment Controls
# ==========================================================================

def _dimension_load_json(path):
    """Load JSON; return {} if file missing."""
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _dimension_load_gprot_embed():
    """Load G protein embeddings keyed by family only."""
    raw = _dimension_load_json(GPROT_FEATURES_FILE)
    feats = {}
    for s, info in raw.items():
        fam = FAMILY_MAP_GPROT.get(s, s)
        vec = np.array(info["mean_pooling"])
        feats[fam] = vec
    return feats


def _dimension_build_df(gpcr_feats, gprot_feats, icl_data, df,
                        gpcr_dim, icl_dim):
    """
    Build feature matrix for dimension controls.
    Note: gpcr_dim is passed but unused; GPCR global dim is always 1280.
    """
    Xl, yl, metas = [], [], []
    for _, row in df.iterrows():
        gid, gf = row["gpcr_id"], row["g_protein_family"]
        gpcr_f = get_gpcr_feat(gpcr_feats, gid)
        gprot_f = gprot_feats.get(gf)
        if gprot_f is None:
            gprot_f = (gprot_feats.get(gf.capitalize())
                       or gprot_feats.get(gf.upper()))
        if gpcr_f is None or gprot_f is None:
            continue
        i2e, i2s, i3e, i3s = get_icl_vector(icl_data, gid, icl_dim)
        Xl.append(np.concatenate([gpcr_f, gprot_f, i2e, i2s, i3e, i3s]))
        yl.append(int(row["coupling"]))
        metas.append({"gpcr_id": gid, "cluster_id": int(row["cluster_id"])})
    return np.array(Xl), np.array(yl), metas


def _dimension_run_svm_cv(X, y, metas, cluster_list):
    """Cluster-aware SVM CV returning (mean_auc, std_auc, fold_aucs_list)."""
    s2c, fold_cids = get_cluster_folds(metas, cluster_list)
    aucs = []
    for fi in range(N_FOLDS):
        tc = set(fold_cids[fi])
        te = [i for i in range(len(y)) if s2c[i] in tc]
        tr = [i for i in range(len(y)) if s2c[i] not in tc]
        sc = StandardScaler()
        X_tr = sc.fit_transform(X[tr])
        X_te = sc.transform(X[te])
        svm = SVC(kernel="rbf", C=10.0, class_weight="balanced",
                  probability=True, random_state=RANDOM_SEED)
        svm.fit(X_tr, y[tr])
        if len(set(y[te])) >= 2:
            aucs.append(roc_auc_score(y[te], svm.predict_proba(X_te)[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs)), aucs


def _dimension_zscore_block(block):
    """Block-wise z-score normalization (zero-mean, unit-variance per column)."""
    mean = block.mean(axis=0)
    std = block.std(axis=0)
    std[std == 0] = 1.0
    return (block - mean) / std


def run_dimension():
    """Dimension alignment control experiments (C1-C4)."""
    print("=" * 70)
    print("  Dimension Alignment Control Experiments")
    print("=" * 70)
    np.random.seed(RANDOM_SEED)

    # Load data
    gpcr_650m_raw = _dimension_load_json(GPCR_FEATURES_FILE)
    gpcr_650m = {k: np.array(v) for k, v in gpcr_650m_raw.items()}
    icl_320 = _dimension_load_json(ICL_320_FILE)
    icl_1280 = _dimension_load_json(ICL_1280_FILE)
    gprot_feats = _dimension_load_gprot_embed()
    df = pd.read_csv(PAIRING_MATRIX_FILE).dropna(subset=["cluster_id"])
    with open(CLUSTERS_FILE) as f:
        cluster_list = json.load(f)["clusters"]

    print(f"  650M GPCR: {len(gpcr_650m)}")
    print(f"  320-d ICL: {len(icl_320)}, 1280-d ICL: {len(icl_1280)}")
    print(f"  Pairs: {len(df)}")

    # Build feature matrices
    X_1280, y_1280, m_1280 = _dimension_build_df(
        gpcr_650m, gprot_feats, icl_1280, df, 1280, 1280)
    X_mis, y_mis, m_mis = _dimension_build_df(
        gpcr_650m, gprot_feats, icl_320, df, 1280, 320)
    print(f"  1280+1280 matched: {X_1280.shape}")
    print(f"  1280+320 mismatched: {X_mis.shape}")

    # Reference baselines
    ref_1280_auc, ref_1280_std, _ = _dimension_run_svm_cv(
        X_1280, y_1280, m_1280, cluster_list)
    ref_mis_auc, ref_mis_std, _ = _dimension_run_svm_cv(
        X_mis, y_mis, m_mis, cluster_list)
    ref_320_paper = 0.8301

    print(f"\n  SVM 1280+1280 matched:     {ref_1280_auc:.4f} +/- {ref_1280_std:.4f}")
    print(f"  SVM 1280+320 mismatched:   {ref_mis_auc:.4f} +/- {ref_mis_std:.4f}")
    print(f"  SVM 320+320 matched (paper): {ref_320_paper:.4f}")
    print(f"  Mismatch penalty: {ref_1280_auc - ref_mis_auc:.4f}")

    results = {
        "reference": {
            "1280_matched": {"auc": ref_1280_auc, "std": ref_1280_std},
            "mismatched_1280_320": {"auc": ref_mis_auc, "std": ref_mis_std},
            "320_matched_paper": ref_320_paper,
        },
        "controls": {},
    }

    # Extract ICL embedding blocks from feature matrices.
    # X_1280: [gpcr(1280) | gprot(1280) | icl2_e(1280) | icl2_s(8) | icl3_e(1280) | icl3_s(8)]
    # X_mis:  [gpcr(1280) | gprot(1280) | icl2_e(320)  | icl2_s(8) | icl3_e(320)  | icl3_s(8)]

    gpcr_global_1280 = X_1280[:, :1280]
    gprot_1280 = X_1280[:, 1280:2560]
    icl2_e_1280 = X_1280[:, 2560:2560 + 1280]
    icl2_s = X_1280[:, 2560 + 1280:2560 + 1280 + 8]
    icl3_e_1280 = X_1280[:, 2560 + 1280 + 8:2560 + 1280 + 8 + 1280]
    icl3_s = X_1280[:, 2560 + 1280 + 8 + 1280:]

    gpcr_global_1280_mis = X_mis[:, :1280]
    icl2_e_320 = X_mis[:, 2560:2560 + 320]
    icl2_s_mis = X_mis[:, 2560 + 320:2560 + 320 + 8]
    icl3_e_320 = X_mis[:, 2560 + 320 + 8:2560 + 320 + 8 + 320]
    icl3_s_mis = X_mis[:, 2560 + 320 + 8 + 320:]

    # ------------------------------------------------------------------
    # C1: PCA projection to 320-d for both GPCR global and ICL
    # ------------------------------------------------------------------
    print("\n--- C1: PCA projection to 320-d for both GPCR global and ICL ---")
    pca_gpcr = PCA(n_components=320, random_state=RANDOM_SEED)
    pca_icl2 = PCA(n_components=320, random_state=RANDOM_SEED)
    pca_icl3 = PCA(n_components=320, random_state=RANDOM_SEED)

    gpcr_320_proj = pca_gpcr.fit_transform(gpcr_global_1280)
    icl2_320_proj = pca_icl2.fit_transform(icl2_e_1280)
    icl3_320_proj = pca_icl3.fit_transform(icl3_e_1280)

    print(f"  PCA GPCR 1280->320: "
          f"explained var = {pca_gpcr.explained_variance_ratio_.sum():.4f}")
    print(f"  PCA ICL2 1280->320: "
          f"explained var = {pca_icl2.explained_variance_ratio_.sum():.4f}")
    print(f"  PCA ICL3 1280->320: "
          f"explained var = {pca_icl3.explained_variance_ratio_.sum():.4f}")

    X_c1 = np.concatenate([
        gpcr_320_proj, gprot_1280,
        icl2_320_proj, icl2_s,
        icl3_320_proj, icl3_s,
    ], axis=1)
    c1_auc, c1_std, c1_folds = _dimension_run_svm_cv(
        X_c1, y_1280, m_1280, cluster_list)
    results["controls"]["C1_PCA_all_to_320"] = {
        "description": (
            "GPCR 1280->320 PCA + ICL 1280->320 PCA + Gprot 1280 "
            "(all projected to 320-d except Gprot)"),
        "auc": c1_auc, "std": c1_std,
        "fold_aucs": [float(a) for a in c1_folds],
        "gpcr_explained_var": float(pca_gpcr.explained_variance_ratio_.sum()),
        "icl2_explained_var": float(pca_icl2.explained_variance_ratio_.sum()),
        "icl3_explained_var": float(pca_icl3.explained_variance_ratio_.sum()),
    }
    print(f"  C1 AUC: {c1_auc:.4f} +/- {c1_std:.4f}")
    print(f"  delta vs 1280 matched: {c1_auc - ref_1280_auc:+.4f}")
    print(f"  delta vs 320 matched (paper {ref_320_paper}): "
          f"{c1_auc - ref_320_paper:+.4f}")

    # ------------------------------------------------------------------
    # C2: Learned linear projection 320-d ICL -> 1280-d
    # ------------------------------------------------------------------
    print("\n--- C2: Learned linear projection 320-d ICL -> 1280-d ---")
    gids_1280 = [m["gpcr_id"] for m in m_1280]
    gids_mis = [m["gpcr_id"] for m in m_mis]
    gid_to_idx_1280 = {g: i for i, g in enumerate(gids_1280)}
    gid_to_idx_mis = {g: i for i, g in enumerate(gids_mis)}
    common = sorted(set(gid_to_idx_1280) & set(gid_to_idx_mis))
    idx_1280_sub = [gid_to_idx_1280[g] for g in common]
    idx_mis_sub = [gid_to_idx_mis[g] for g in common]
    print(f"  Common GPCRs: {len(common)}")

    icl2_320_sub = icl2_e_320[idx_mis_sub]
    icl3_320_sub = icl3_e_320[idx_mis_sub]
    icl2_1280_sub = icl2_e_1280[idx_1280_sub]
    icl3_1280_sub = icl3_e_1280[idx_1280_sub]

    lr2 = LinearRegression()
    lr3 = LinearRegression()
    lr2.fit(icl2_320_sub, icl2_1280_sub)
    lr3.fit(icl3_320_sub, icl3_1280_sub)
    print(f"  LR ICL2 320->1280: R^2 = "
          f"{lr2.score(icl2_320_sub, icl2_1280_sub):.4f}")
    print(f"  LR ICL3 320->1280: R^2 = "
          f"{lr3.score(icl3_320_sub, icl3_1280_sub):.4f}")

    icl2_1280_proj = lr2.predict(icl2_e_320)
    icl3_1280_proj = lr3.predict(icl3_e_320)

    X_c2 = np.concatenate([
        gpcr_global_1280_mis, gprot_1280,
        icl2_1280_proj, icl2_s_mis,
        icl3_1280_proj, icl3_s_mis,
    ], axis=1)
    c2_auc, c2_std, c2_folds = _dimension_run_svm_cv(
        X_c2, y_mis, m_mis, cluster_list)
    results["controls"]["C2_learned_proj_320icl_to_1280"] = {
        "description": (
            "320-d ICL projected to 1280-d via linear regression, "
            "combined with 1280-d global"),
        "auc": c2_auc, "std": c2_std,
        "fold_aucs": [float(a) for a in c2_folds],
        "icl2_r2": float(lr2.score(icl2_320_sub, icl2_1280_sub)),
        "icl3_r2": float(lr3.score(icl3_320_sub, icl3_1280_sub)),
    }
    print(f"  C2 AUC: {c2_auc:.4f} +/- {c2_std:.4f}")
    print(f"  delta vs mismatched: {c2_auc - ref_mis_auc:+.4f}")
    print(f"  delta vs 1280 matched: {c2_auc - ref_1280_auc:+.4f}")

    # ------------------------------------------------------------------
    # C3: Block-wise z-score normalization
    # ------------------------------------------------------------------
    print("\n--- C3: Block-wise z-score normalization ---")
    X_c3 = np.concatenate([
        _dimension_zscore_block(gpcr_global_1280_mis),
        gprot_1280,
        _dimension_zscore_block(icl2_e_320), icl2_s_mis,
        _dimension_zscore_block(icl3_e_320), icl3_s_mis,
    ], axis=1)
    c3_auc, c3_std, c3_folds = _dimension_run_svm_cv(
        X_c3, y_mis, m_mis, cluster_list)
    results["controls"]["C3_block_zscore"] = {
        "description": (
            "GPCR global and ICL blocks z-score normalized separately "
            "before concat (mismatched dims)"),
        "auc": c3_auc, "std": c3_std,
        "fold_aucs": [float(a) for a in c3_folds],
    }
    print(f"  C3 AUC: {c3_auc:.4f} +/- {c3_std:.4f}")
    print(f"  delta vs mismatched (no zscore): {c3_auc - ref_mis_auc:+.4f}")

    # ------------------------------------------------------------------
    # C4: Zero-padding 320-d ICL -> 1280-d
    # ------------------------------------------------------------------
    print("\n--- C4: Zero-padding 320-d ICL -> 1280-d ---")
    icl2_e_padded = np.pad(icl2_e_320, ((0, 0), (0, 960)), mode='constant')
    icl3_e_padded = np.pad(icl3_e_320, ((0, 0), (0, 960)), mode='constant')

    X_c4 = np.concatenate([
        gpcr_global_1280_mis, gprot_1280,
        icl2_e_padded, icl2_s_mis,
        icl3_e_padded, icl3_s_mis,
    ], axis=1)
    c4_auc, c4_std, c4_folds = _dimension_run_svm_cv(
        X_c4, y_mis, m_mis, cluster_list)
    results["controls"]["C4_zero_pad_320icl_to_1280"] = {
        "description": (
            "320-d ICL zero-padded to 1280-d, combined with 1280-d global"),
        "auc": c4_auc, "std": c4_std,
        "fold_aucs": [float(a) for a in c4_folds],
    }
    print(f"  C4 AUC: {c4_auc:.4f} +/- {c4_std:.4f}")
    print(f"  delta vs mismatched: {c4_auc - ref_mis_auc:+.4f}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  SUMMARY: Dimension Alignment Controls")
    print(f"{'='*70}")
    print(f"  {'Configuration':<55s} {'AUC':>8s} {'Std':>8s}")
    print(f"  {'-'*77}")
    print(f"  {'1280+1280 matched (reference)':<55s} "
          f"{ref_1280_auc:>8.4f} {ref_1280_std:>8.4f}")
    print(f"  {'1280+320 mismatched (reference)':<55s} "
          f"{ref_mis_auc:>8.4f} {ref_mis_std:>8.4f}")
    print(f"  {'320+320 matched (paper reference)':<55s} "
          f"{ref_320_paper:>8.4f} {'--':>8s}")
    print(f"  {'C1: PCA all->320 (GPCR 1280->320 + ICL 1280->320)':<55s} "
          f"{c1_auc:>8.4f} {c1_std:>8.4f}")
    print(f"  {'C2: LR ICL 320->1280 + GPCR 1280':<55s} "
          f"{c2_auc:>8.4f} {c2_std:>8.4f}")
    print(f"  {'C3: Block-wise z-score (mismatched dims)':<55s} "
          f"{c3_auc:>8.4f} {c3_std:>8.4f}")
    print(f"  {'C4: Zero-pad ICL 320->1280 + GPCR 1280':<55s} "
          f"{c4_auc:>8.4f} {c4_std:>8.4f}")

    print(f"\n  Interpretation:")
    mm_penalty = ref_1280_auc - ref_mis_auc
    print(f"    Mismatch penalty (1280-1280 vs 1280-320): {mm_penalty:+.4f} AUC")
    for label, delta in [
        ("C1 PCA->320", c1_auc - ref_320_paper),
        ("C2 LR->1280", c2_auc - ref_1280_auc),
        ("C3 zscore ", c3_auc - ref_mis_auc),
        ("C4 pad    ", c4_auc - ref_mis_auc),
    ]:
        if abs(delta) < 0.005:
            interp = "RECOVERS -- mismatch is feature-scale issue"
        elif delta > 0.002:
            interp = "partially recovers"
        else:
            interp = ("does NOT recover -- "
                      "dimension alignment requirement supported")
        print(f"    {label}: deltaAUC={delta:+.4f} -> {interp}")

    with open(DIMENSION_OUTPUT, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {DIMENSION_OUTPUT}")


# ==========================================================================
# Subcommand: minimal  --  Minimal Family-Conditioned Classifier
# ==========================================================================

class MLPClassifier(nn.Module):
    """3-layer MLP with GELU/BN/Dropout (same architecture as paper)."""

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
    """Simple vector-label dataset."""

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def _minimal_build_vectors(df, gpcr_feats, icl_data):
    """
    Build feature vectors: [GPCR global 1280 | ICL full 2576 | family onehot 4]
    G protein side is a 4-d family one-hot vector (NOT the 1280-d embedding).
    """
    X_list, y_list, metas = [], [], []
    for _, row in df.iterrows():
        gid = row["gpcr_id"]
        gpcr_f = get_gpcr_feat(gpcr_feats, gid)
        gf = row["g_protein_family"]
        gprot_f = np.array([1.0 if gf == f else 0.0 for f in FAMILIES],
                           dtype=np.float64)
        if gpcr_f is None:
            continue
        i2_e, i2_s, i3_e, i3_s = get_icl_vector(icl_data, gid, 1280)
        parts = [gpcr_f, gprot_f, i2_e, i2_s, i3_e, i3_s]
        X_list.append(np.concatenate(parts))
        y_list.append(int(row["coupling"]))
        metas.append({"gpcr_id": gid, "family": gf,
                      "cluster_id": int(row["cluster_id"])})
    return np.array(X_list), np.array(y_list), metas


def _minimal_train_epoch(model, loader, optim, crit):
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
def _minimal_evaluate_mlp(model, loader):
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        x = x.to(DEVICE)
        probs.append(torch.sigmoid(model(x)).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def _minimal_compute_metrics(y_true, y_prob):
    """Return AUC, PRAUC, Brier score dict."""
    auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) >= 2 else float("nan")
    prauc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    return {"auc": auc, "prauc": prauc, "brier": brier}


def _minimal_per_family_metrics(y_true, y_prob, families_arr):
    """Compute metrics per G protein family."""
    results = {}
    for fam in FAMILIES:
        mask = np.array([f == fam for f in families_arr])
        if mask.sum() == 0 or len(set(y_true[mask])) < 2:
            results[fam] = {"n": int(mask.sum()), "auc": float("nan"),
                            "prauc": float("nan"), "brier": float("nan")}
        else:
            m = _minimal_compute_metrics(y_true[mask], y_prob[mask])
            m["n"] = int(mask.sum())
            results[fam] = m
    return results


# ----- Logistic Regression -----

def _minimal_run_logistic_cv(X, y, s2c, fold_cids, Cs=None):
    """Logistic regression with L2 penalty, C tuned via inner 3-fold CV."""
    if Cs is None:
        Cs = [0.01, 0.1, 1.0, 10.0, 100.0]
    fold_results = []
    for fi in range(N_FOLDS):
        test_cids = set(fold_cids[fi])
        te_idx = [i for i in range(len(y)) if s2c[i] in test_cids]
        tr_idx = [i for i in range(len(y)) if s2c[i] not in test_cids]
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr_idx])
        X_te = scaler.transform(X[te_idx])
        try:
            lr = LogisticRegressionCV(
                Cs=Cs, cv=3, scoring="roc_auc", class_weight="balanced",
                max_iter=2000, random_state=RANDOM_SEED, n_jobs=1)
            lr.fit(X_tr, y[tr_idx])
        except Exception:
            lr = LogisticRegression(
                C=1.0, class_weight="balanced", max_iter=2000,
                random_state=RANDOM_SEED)
            lr.fit(X_tr, y[tr_idx])
        p = lr.predict_proba(X_te)[:, 1]
        best_c = float(lr.C_[0]) if hasattr(lr, "C_") else 1.0
        fold_results.append({"probs": p, "labels": y[te_idx],
                             "te_idx": te_idx, "best_C": best_c})
    return fold_results


# ----- SVM -----

def _minimal_run_svm_cv(X, y, s2c, fold_cids):
    """SVM RBF with fixed C=10 (matching main protocol)."""
    fold_results = []
    for fi in range(N_FOLDS):
        test_cids = set(fold_cids[fi])
        te_idx = [i for i in range(len(y)) if s2c[i] in test_cids]
        tr_idx = [i for i in range(len(y)) if s2c[i] not in test_cids]
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr_idx])
        X_te = scaler.transform(X[te_idx])
        svm = SVC(kernel="rbf", C=10.0, class_weight="balanced",
                  probability=True, random_state=RANDOM_SEED)
        svm.fit(X_tr, y[tr_idx])
        p = svm.predict_proba(X_te)[:, 1]
        fold_results.append({"probs": p, "labels": y[te_idx], "te_idx": te_idx})
    return fold_results


# ----- MLP -----

def _minimal_run_mlp_cv(X, y, s2c, fold_cids):
    """3-layer MLP (same architecture as paper)."""
    fold_results = []
    for fi in range(N_FOLDS):
        test_cids = set(fold_cids[fi])
        te_idx = [i for i in range(len(y)) if s2c[i] in test_cids]
        tr_idx = [i for i in range(len(y)) if s2c[i] not in test_cids]
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr_idx])
        X_te = scaler.transform(X[te_idx])

        tr_ds = VecDataset(X_tr, y[tr_idx])
        te_ds = VecDataset(X_te, y[te_idx])
        tr_ld = DataLoader(tr_ds, batch_size=32, shuffle=True)
        te_ld = DataLoader(te_ds, batch_size=32)

        model = MLPClassifier(input_dim=X.shape[1]).to(DEVICE)
        pw = len(y[tr_idx]) / max(1, y[tr_idx].sum())
        optim = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        crit = nn.BCEWithLogitsLoss(
            pos_weight=torch.FloatTensor([pw]).to(DEVICE))

        best_auc, best_p, patience = -1.0, None, 0
        for ep in range(200):
            _minimal_train_epoch(model, tr_ld, optim, crit)
            p, lbl = _minimal_evaluate_mlp(model, te_ld)
            auc = roc_auc_score(lbl, p) if len(set(lbl)) >= 2 else -1
            if auc > best_auc:
                best_auc, best_p, patience = auc, p, 0
            else:
                patience += 1
            if patience >= 20:
                break
        fold_results.append(
            {"probs": best_p, "labels": y[te_idx], "te_idx": te_idx})
    return fold_results


def _minimal_aggregate_fold_results(fold_results, metas):
    """Compute aggregated metrics from fold results."""
    all_probs = np.concatenate([fr["probs"] for fr in fold_results])
    all_labels = np.concatenate([fr["labels"] for fr in fold_results])
    all_te_idx = np.concatenate([fr["te_idx"] for fr in fold_results])
    sort_idx = np.argsort(all_te_idx)
    all_probs = all_probs[sort_idx]
    all_labels = all_labels[sort_idx]

    fold_aucs, fold_praucs = [], []
    for fr in fold_results:
        if len(set(fr["labels"])) >= 2:
            fold_aucs.append(roc_auc_score(fr["labels"], fr["probs"]))
            fold_praucs.append(average_precision_score(fr["labels"], fr["probs"]))

    families_arr = [metas[i]["family"] for i in all_te_idx]
    families_arr = [families_arr[i] for i in sort_idx]
    pf = _minimal_per_family_metrics(all_labels, all_probs, families_arr)

    overall = _minimal_compute_metrics(all_labels, all_probs)
    overall["auc_mean"] = float(np.mean(fold_aucs))
    overall["auc_std"] = float(np.std(fold_aucs))
    overall["prauc_mean"] = float(np.mean(fold_praucs))
    overall["prauc_std"] = float(np.std(fold_praucs))
    overall["fold_aucs"] = [float(a) for a in fold_aucs]
    return overall, pf


def _minimal_run_repeated_cv(X, y, metas, cluster_list, model_fn, n_repeats=5):
    """Run repeated cluster-aware CV with different fold assignments."""
    all_repeat_aucs, all_repeat_praucs, all_repeat_briers = [], [], []
    for rep in range(n_repeats):
        seed = RANDOM_SEED + rep * 100
        s2c, fold_cids = get_cluster_folds(
            metas, cluster_list, n_folds=N_FOLDS, seed=seed,
            tiebreak_by_id=True)
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


def run_minimal():
    """Minimal family-conditioned classifier benchmark (LR / MLP / SVM)."""
    N_REPEATS = 5  # local constant from original script

    print("=" * 70)
    print("  Minimal Family-Conditioned Classifier Benchmark")
    print("  LR / MLP / SVM with GPCR ESM-2 650M + ICL + Family One-Hot")
    print("=" * 70)

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)

    gpcr_feats = load_gpcr_features()
    icl_data = load_icl_features()
    df = pd.read_csv(PAIRING_MATRIX_FILE)
    df = df.dropna(subset=["cluster_id"]).copy()

    with open(CLUSTERS_FILE) as f:
        cluster_list = json.load(f)["clusters"]

    print(f"  GPCR features: {len(gpcr_feats)}")
    print(f"  ICL features: {len(icl_data)}")
    print(f"  Pairs: {len(df)}")

    X, y, metas = _minimal_build_vectors(df, gpcr_feats, icl_data)
    print(f"  Feature dim: {X.shape[1]} "
          f"(1280 GPCR + 2576 ICL + 4 family onehot)")
    print(f"  Samples: {len(y)}, Pos ratio: {y.mean():.4f}")

    print(f"\n  Reference: CA 650M + ICL (full Gprot embedding) = 0.862 +/- 0.025")
    print(f"  Reference: CA 650M + ICL (Gprot onehot)     = 0.855 +/- 0.018")
    print(f"  Reference: MLP 650M + ICL (full Gprot)       = 0.861 +/- 0.023")
    print(f"  Reference: SVM 650M + ICL (full Gprot)       = 0.832 +/- 0.014")

    results = {}

    # Logistic Regression
    print(f"\n{'='*70}")
    print(f"  Logistic Regression (L2, C tuned via inner 3-fold CV)")
    print(f"{'='*70}")
    lr_res = _minimal_run_repeated_cv(
        X, y, metas, cluster_list, _minimal_run_logistic_cv,
        n_repeats=N_REPEATS)
    results["logistic_regression"] = lr_res
    print(f"  AUC = {lr_res['auc_mean']:.4f} "
          f"[{lr_res['auc_95ci_low']:.4f}, {lr_res['auc_95ci_high']:.4f}]")
    print(f"  PRAUC = {lr_res['prauc_mean']:.4f}, "
          f"Brier = {lr_res['brier_mean']:.4f}")

    # SVM
    print(f"\n{'='*70}")
    print(f"  SVM (RBF, C=10, class_weight=balanced)")
    print(f"{'='*70}")
    svm_res = _minimal_run_repeated_cv(
        X, y, metas, cluster_list, _minimal_run_svm_cv,
        n_repeats=N_REPEATS)
    results["svm_rbf"] = svm_res
    print(f"  AUC = {svm_res['auc_mean']:.4f} "
          f"[{svm_res['auc_95ci_low']:.4f}, {svm_res['auc_95ci_high']:.4f}]")
    print(f"  PRAUC = {svm_res['prauc_mean']:.4f}, "
          f"Brier = {svm_res['brier_mean']:.4f}")

    # MLP
    print(f"\n{'='*70}")
    print(f"  MLP (3-layer, 256-128-1, GELU/BN/Dropout)")
    print(f"{'='*70}")
    mlp_res = _minimal_run_repeated_cv(
        X, y, metas, cluster_list, _minimal_run_mlp_cv,
        n_repeats=N_REPEATS)
    results["mlp"] = mlp_res
    print(f"  AUC = {mlp_res['auc_mean']:.4f} "
          f"[{mlp_res['auc_95ci_low']:.4f}, {mlp_res['auc_95ci_high']:.4f}]")
    print(f"  PRAUC = {mlp_res['prauc_mean']:.4f}, "
          f"Brier = {mlp_res['brier_mean']:.4f}")

    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY: Minimal Family-Conditioned Classifiers")
    print(f"{'='*70}")
    print(f"  {'Model':<25s} {'AUC':>10s} {'95% CI':>20s} "
          f"{'PRAUC':>10s} {'Brier':>10s}")
    print(f"  {'-'*75}")
    for name, r in results.items():
        print(f"  {name:<25s} {r['auc_mean']:>10.4f} "
              f"[{r['auc_95ci_low']:.4f}, {r['auc_95ci_high']:.4f}] "
              f"{r['prauc_mean']:>10.4f} {r['brier_mean']:>10.4f}")

    print(f"\n  Comparison with paper's paired models (with full Gprot embedding):")
    print(f"  {'Model':<25s} {'AUC':>10s}")
    print(f"  {'CA 650M + ICL (full Gprot)':<25s} {'0.862':>10s}")
    print(f"  {'MLP 650M + ICL (full Gprot)':<25s} {'0.861':>10s}")
    print(f"  {'SVM 650M + ICL (full Gprot)':<25s} {'0.832':>10s}")

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
    with open(MINIMAL_OUTPUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {MINIMAL_OUTPUT}")


# ==========================================================================
# Subcommand: label  --  Label Audit and Tiered Evaluation
# ==========================================================================

def _label_detect_species(gpcr_id):
    """Detect species from GPCR ID prefix convention."""
    if "_" in gpcr_id:
        prefix = gpcr_id.split("_")[0]
    else:
        prefix = ""

    MOUSE_PREFIXES = {"MOUSE", "MUS", "M_"}
    RAT_PREFIXES = {"RAT", "R_"}

    if prefix.upper() in MOUSE_PREFIXES:
        return "mouse"
    elif prefix.upper() in RAT_PREFIXES:
        return "rat"
    elif prefix == "":
        return "human"
    elif len(prefix) <= 2 and prefix[0].isalpha() and prefix[0].isupper():
        return "human"
    else:
        return "human"


def _label_classify_evidence(source, coupling, gpcr_pos_families):
    """
    Classify evidence strength for a pair.

    Positive labels: direct_assay (experimentally validated).
    Negative labels from local_seed: curated_negative (manually curated).
    Negative labels from gpcrdb_iuphar with >=1 positive for that GPCR:
        curated_negative (tested negative / selective coupling).
    Negative labels from gpcrdb_iuphar with zero positives for that GPCR:
        inferred_negative (cannot distinguish orphan from untested).
    """
    if coupling == 1:
        return "direct_assay"
    if source == "local_seed":
        return "curated_negative"
    if gpcr_pos_families > 0:
        return "curated_negative"
    else:
        return "inferred_negative"


def _label_build_vectors(df, gpcr_feats, icl_data):
    """Build feature vectors: GPCR 1280 + ICL 2576 + family onehot 4."""
    X_list, y_list, metas = [], [], []
    for _, row in df.iterrows():
        gid = row["gpcr_id"]
        gpcr_f = get_gpcr_feat(gpcr_feats, gid)
        gf = row["g_protein_family"]
        gprot_f = np.array([1.0 if gf == f else 0.0 for f in FAMILIES],
                           dtype=np.float64)
        if gpcr_f is None:
            continue
        i2_e, i2_s, i3_e, i3_s = get_icl_vector(icl_data, gid, 1280)
        parts = [gpcr_f, gprot_f, i2_e, i2_s, i3_e, i3_s]
        X_list.append(np.concatenate(parts))
        y_list.append(int(row["coupling"]))
        metas.append({"gpcr_id": gid, "family": gf,
                      "cluster_id": int(row["cluster_id"])})
    return np.array(X_list), np.array(y_list), metas


def _label_evaluate_subset(X, y, metas, cluster_list):
    """Run cluster-aware SVM CV on a data subset."""
    if len(y) < 20:
        return {"auc": float("nan"), "auc_std": float("nan"),
                "prauc": float("nan"), "n": int(len(y)),
                "error": "too few samples"}

    s2c, fold_cids = get_cluster_folds(metas, cluster_list)
    fold_aucs, fold_praucs = [], []
    all_probs, all_labels = [], []

    for fi in range(N_FOLDS):
        test_cids = set(fold_cids[fi])
        te_idx = [i for i in range(len(y)) if s2c[i] in test_cids]
        tr_idx = [i for i in range(len(y)) if s2c[i] not in test_cids]

        if len(te_idx) < 5 or len(tr_idx) < 10:
            continue

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr_idx])
        X_te = scaler.transform(X[te_idx])

        svm = SVC(kernel="rbf", C=10.0, class_weight="balanced",
                  probability=True, random_state=RANDOM_SEED)
        svm.fit(X_tr, y[tr_idx])
        p = svm.predict_proba(X_te)[:, 1]

        all_probs.extend(p)
        all_labels.extend(y[te_idx])
        if len(set(y[te_idx])) >= 2:
            fold_aucs.append(roc_auc_score(y[te_idx], p))
            fold_praucs.append(average_precision_score(y[te_idx], p))

    if not fold_aucs:
        return {"auc": float("nan"), "n": int(len(y)),
                "error": "no valid folds"}

    return {
        "auc_mean": float(np.mean(fold_aucs)),
        "auc_std": float(np.std(fold_aucs)),
        "prauc_mean": (float(np.mean(fold_praucs))
                       if fold_praucs else float("nan")),
        "prauc_std": (float(np.std(fold_praucs))
                      if fold_praucs else float("nan")),
        "brier": float(brier_score_loss(all_labels, all_probs)),
        "n": int(len(y)),
        "n_pos": int(sum(y)),
        "pos_ratio": float(np.mean(y)),
        "fold_aucs": [float(a) for a in fold_aucs],
    }


def run_label():
    """Label audit: metadata annotation and tiered SVM evaluation."""
    print("=" * 70)
    print("  Label Audit: Metadata Annotation & Tiered Evaluation")
    print("=" * 70)

    np.random.seed(RANDOM_SEED)

    df = pd.read_csv(PAIRING_MATRIX_FILE)
    df = df.dropna(subset=["cluster_id"]).copy()
    gpcr_feats = load_gpcr_features()
    icl_data = load_icl_features()

    with open(CLUSTERS_FILE) as f:
        cluster_list = json.load(f)["clusters"]

    print(f"  Dataset: {len(df)} pairs, {df['gpcr_id'].nunique()} GPCRs")

    # ------------------------------------------------------------------
    # Phase 1: Metadata Annotation
    # ------------------------------------------------------------------
    print("\n--- Phase 1: Metadata Annotation ---")

    gpcr_pos_count = defaultdict(int)
    gpcr_neg_count = defaultdict(int)
    gpcr_families_tested = defaultdict(set)
    gpcr_sources = defaultdict(set)

    for _, row in df.iterrows():
        gid = row["gpcr_id"]
        gf = row["g_protein_family"]
        if row["coupling"] == 1:
            gpcr_pos_count[gid] += 1
        else:
            gpcr_neg_count[gid] += 1
        gpcr_families_tested[gid].add(gf)
        gpcr_sources[gid].add(row["source"])

    annotations = []
    for _, row in df.iterrows():
        gid = row["gpcr_id"]
        species = _label_detect_species(gid)
        evidence = _label_classify_evidence(
            row["source"], int(row["coupling"]),
            gpcr_pos_count[gid])
        annotations.append({
            "gpcr_id": gid,
            "g_protein_family": row["g_protein_family"],
            "coupling": int(row["coupling"]),
            "source": row["source"],
            "species": species,
            "evidence_strength": evidence,
            "zero_positive_gpcr": gpcr_pos_count[gid] == 0,
            "n_families_tested": len(gpcr_families_tested[gid]),
            "gpcr_pos_count": gpcr_pos_count[gid],
        })

    species_counts = defaultdict(int)
    evidence_counts = defaultdict(int)
    zero_pos_count = sum(1 for a in annotations if a["zero_positive_gpcr"])
    n_zero_pos_gpcrs = sum(
        1 for gid in set(a["gpcr_id"] for a in annotations)
        if gpcr_pos_count[gid] == 0)

    for a in annotations:
        species_counts[a["species"]] += 1
        evidence_counts[a["evidence_strength"]] += 1

    print(f"  Species: {dict(species_counts)}")
    print(f"  Evidence: {dict(evidence_counts)}")
    print(f"  Zero-positive GPCRs: {n_zero_pos_gpcrs} ({zero_pos_count} pairs)")
    print(f"  GPCRs with >=1 positive: "
          f"{df['gpcr_id'].nunique() - n_zero_pos_gpcrs}")

    # ------------------------------------------------------------------
    # Phase 2: Tiered Evaluation
    # ------------------------------------------------------------------
    print("\n--- Phase 2: Tiered Evaluation ---")

    X_full, y_full, metas_full = _label_build_vectors(df, gpcr_feats, icl_data)
    print(f"  Full feature matrix: {X_full.shape}")

    # Tier 1: Full dataset
    print("\n  Tier 1: Full dataset")
    tier1 = _label_evaluate_subset(X_full, y_full, metas_full, cluster_list)
    t1_auc = tier1.get("auc_mean", float("nan"))
    t1_std = tier1.get("auc_std", 0)
    t1_prauc = tier1.get("prauc_mean", float("nan"))
    print(f"    N={tier1['n']}, AUC={t1_auc:.4f} +/- {t1_std:.4f}, "
          f"PRAUC={t1_prauc:.4f}")

    # Tier 2: Remove zero-positive GPCRs
    print("\n  Tier 2: Remove zero-positive GPCRs")
    tier2_mask = np.array([
        gpcr_pos_count[metas_full[i]["gpcr_id"]] > 0
        for i in range(len(y_full))
    ])
    X_t2, y_t2 = X_full[tier2_mask], y_full[tier2_mask]
    metas_t2 = [metas_full[i] for i in range(len(metas_full))
                if tier2_mask[i]]
    tier2 = _label_evaluate_subset(X_t2, y_t2, metas_t2, cluster_list)
    t2_auc = tier2.get("auc_mean", float("nan"))
    t2_std = tier2.get("auc_std", 0)
    t2_prauc = tier2.get("prauc_mean", float("nan"))
    print(f"    N={tier2['n']}, AUC={t2_auc:.4f} +/- {t2_std:.4f}, "
          f"PRAUC={t2_prauc:.4f}")

    # Tier 3: High-confidence only
    print("\n  Tier 3: High-confidence only (direct_assay + curated_negative)")
    tier3_mask = np.array([
        annotations[i]["evidence_strength"] != "inferred_negative"
        for i in range(len(y_full))
    ])
    X_t3, y_t3 = X_full[tier3_mask], y_full[tier3_mask]
    metas_t3 = [metas_full[i] for i in range(len(metas_full))
                if tier3_mask[i]]
    tier3 = _label_evaluate_subset(X_t3, y_t3, metas_t3, cluster_list)
    t3_auc = tier3.get("auc_mean", float("nan"))
    t3_std = tier3.get("auc_std", 0)
    t3_prauc = tier3.get("prauc_mean", float("nan"))
    print(f"    N={tier3['n']}, AUC={t3_auc:.4f} +/- {t3_std:.4f}, "
          f"PRAUC={t3_prauc:.4f}")

    # ------------------------------------------------------------------
    # Phase 3: Per-Family Evidence Breakdown
    # ------------------------------------------------------------------
    print("\n--- Phase 3: Per-Family Evidence Breakdown ---")
    family_breakdown = {}
    for fam in FAMILIES:
        fam_pairs = [a for a in annotations
                     if a["g_protein_family"] == fam]
        fam_evidence = defaultdict(lambda: {"pos": 0, "neg": 0})
        for a in fam_pairs:
            lbl = "pos" if a["coupling"] == 1 else "neg"
            fam_evidence[a["evidence_strength"]][lbl] += 1
        family_breakdown[fam] = {
            "total": len(fam_pairs),
            "positive": sum(1 for a in fam_pairs if a["coupling"] == 1),
            "by_evidence": {k: dict(v) for k, v in fam_evidence.items()},
        }
        print(f"    {fam}: {family_breakdown[fam]}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out = {
        "description": "Label audit: tiered evaluation by label quality",
        "dataset_summary": {
            "n_pairs": len(annotations),
            "n_gpcrs": df["gpcr_id"].nunique(),
            "n_zero_positive_gpcrs": n_zero_pos_gpcrs,
            "species_distribution": dict(species_counts),
            "evidence_distribution": dict(evidence_counts),
        },
        "tiered_results": {
            "tier1_full": tier1,
            "tier2_remove_zero_positive": tier2,
            "tier3_high_confidence": tier3,
        },
        "per_family_breakdown": family_breakdown,
        "reference_svm_650m_icl_auc": 0.832,
    }

    with open(LABEL_OUTPUT, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved to {LABEL_OUTPUT}")


# ==========================================================================
# Main entry point
# ==========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Ablation and control experiments suite for GPCR-G protein "
                    "coupling prediction. Select a subcommand to run a specific "
                    "experiment, or 'all' to run all four sequentially.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ablation.py onehot       # G protein one-hot ablation
  python ablation.py dimension    # Dimension alignment controls
  python ablation.py minimal      # Minimal family classifier benchmark
  python ablation.py label        # Label audit and tiered evaluation
  python ablation.py all          # Run all four experiments
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="Experiment to run")

    subparsers.add_parser(
        "onehot",
        help="G protein one-hot ablation: embedding vs 4-d onehot vs none")
    subparsers.add_parser(
        "dimension",
        help="Dimension alignment controls: PCA, linear proj, z-score, zero-pad")
    subparsers.add_parser(
        "minimal",
        help="Minimal family-conditioned classifier: LR / MLP / SVM with onehot")
    subparsers.add_parser(
        "label",
        help="Label audit: metadata annotation and tiered SVM evaluation")
    subparsers.add_parser(
        "all",
        help="Run all four experiments sequentially")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    commands = (["onehot", "dimension", "minimal", "label"]
                if args.command == "all" else [args.command])

    for cmd in commands:
        if cmd == "onehot":
            run_onehot()
        elif cmd == "dimension":
            run_dimension()
        elif cmd == "minimal":
            run_minimal()
        elif cmd == "label":
            run_label()

    print("\n" + "=" * 70)
    print("  All requested experiments complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
