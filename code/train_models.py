#!/usr/bin/env python3
"""
Unified training script for GPCR-G protein coupling prediction models.

Trains one or more models under cluster-aware 5-fold cross-validation:
  - ca  : PairedCrossAttentionNet (4-head, 256-d hidden, GELU, layer norm, dropout 0.3)
         using 650M ESM-2 embeddings across three feature modes (baseline, icl_full, alpha).
  - mlp : Multi-layer perceptron (256 -> 128 -> 1, GELU, batch norm, dropout 0.3)
         on the icl_full feature set.
  - rf  : Random Forest (200 estimators, class_weight='balanced') on icl_full.
  - xgb : XGBoost (300 estimators, max_depth=6, lr=0.05) on icl_full.
  - all : Train every model above and save results to their respective output files.

All models share the same data loading and cluster-to-fold assignment logic.
Cluster-aware splits ensure that all samples belonging to the same sequence
cluster are assigned to the same fold, preventing data leakage through
homologous sequences.

Output files (paths preserved from original scripts):
  - data/paired_cross_attention_650m_results.json  (cross-attention results)
  - data/baseline_results.json                      (MLP, RF, XGB results)

Usage:
  python train_models.py --model ca       # cross-attention only
  python train_models.py --model mlp      # MLP only
  python train_models.py --model rf       # random forest only
  python train_models.py --model xgb      # XGBoost only
  python train_models.py --model all      # all models
"""

import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE.parent / "data"

GPCR_FEATURES_FILE = BASE / "server_sync" / "extended_data" / "features" / "esm_features_650m_meanpool.json"
G_PROTEIN_FEATURES_FILE = DATA_DIR / "g_protein_esm_features.json"
ICL_FEATURES_FILE = DATA_DIR / "icl_features_650m.json"
ALPHA_FEATURES_FILE = DATA_DIR / "alphafold_icl_features.json"
PAIRING_MATRIX_FILE = DATA_DIR / "pairing_matrix_raw.csv"
CLUSTERS_FILE = DATA_DIR / "sequence_clusters.json"

CA_OUTPUT_FILE = DATA_DIR / "paired_cross_attention_650m_results.json"
BASELINE_OUTPUT_FILE = DATA_DIR / "baseline_results.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===================================================================
#  Feature loading utilities (shared by all models)
# ===================================================================

def load_features():
    """Load GPCR and G-protein ESM-2 650M mean-pooled features."""
    with open(GPCR_FEATURES_FILE) as f:
        gpcr_raw = json.load(f)
    with open(G_PROTEIN_FEATURES_FILE) as f:
        gprot_raw = json.load(f)

    gpcr_feats = {}
    for k, v in gpcr_raw.items():
        arr = np.array(v)
        gpcr_feats[k] = arr.mean(axis=0) if arr.ndim == 2 else arr

    gprot_feats = {}
    for subtype, info in gprot_raw.items():
        family_map = {
            "GNAQ": "Gq", "GNAI1": "Gi", "GNAI2": "Gi", "GNAI3": "Gi",
            "GNAS": "Gs", "GNA12": "G12_13", "GNA13": "G12_13",
        }
        family = family_map.get(subtype, subtype)
        gprot_feats[subtype] = np.array(info["mean_pooling"])
        if family not in gprot_feats:
            gprot_feats[family] = np.array(info["mean_pooling"])
    return gpcr_feats, gprot_feats


def load_icl_features():
    """Load ICL (intracellular loop) ESM and biophysical features."""
    if not ICL_FEATURES_FILE.exists():
        return {}
    with open(ICL_FEATURES_FILE) as f:
        return json.load(f)


def load_alpha_features():
    """Load AlphaFold-derived structural features for ICL regions."""
    if not ALPHA_FEATURES_FILE.exists():
        return {}
    with open(ALPHA_FEATURES_FILE) as f:
        return json.load(f)


def get_gpcr_feat(gpcr_feats, gid):
    """Retrieve GPCR feature vector with fallback name resolution."""
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
    """Extract concatenated ICL2/ICL3 ESM embeddings and biophysical stats."""
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
    stat_keys = ["length", "mean_hydro", "std_hydro", "net_charge", "pos_charge_ratio",
                 "neg_charge_ratio", "hydrophobic_ratio", "aromatic_ratio"]
    icl2_stat_vec = np.array([icl2_stats.get(k, 0.0) for k in stat_keys])
    icl3_stat_vec = np.array([icl3_stats.get(k, 0.0) for k in stat_keys])
    return icl2_esm, icl2_stat_vec, icl3_esm, icl3_stat_vec


def get_alpha_vector(alpha_data, gid):
    """Extract AlphaFold-derived structural features for a GPCR."""
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
        "tm5_tm6_cyto_ca_distance", "icl2_end_to_end_ca_distance",
        "icl3_end_to_end_ca_distance", "tm5_tm6_cyto_dihedral_angle",
        "icl2_aromatic_centroid_depth", "interface_patch_sasa",
        "interface_patch_sasa_ratio", "icl2_helix_ratio", "icl2_sheet_ratio",
        "icl2_coil_ratio", "icl3_helix_ratio", "icl3_sheet_ratio",
        "icl3_coil_ratio",
        "icl2_mean_pae", "icl2_intra_pae",
        "icl3_mean_pae", "icl3_intra_pae",
        "icl2_tm5_pae", "icl2_tm6_pae",
        "icl3_tm5_pae", "icl3_tm6_pae",
    ]
    if rec is None:
        return np.zeros(len(keys))
    return np.array([rec.get(k, 0.0) for k in keys])


def build_vectors(df, gpcr_feats, gprot_feats, icl_data, alpha_data, mode="baseline"):
    """Assemble feature matrix X and label vector y from the pairing matrix.

    Parameters
    ----------
    mode : str, one of {"baseline", "icl_full", "alpha"}
        Feature set to construct:
        - "baseline":  GPCR (1280) + G-protein (320) global ESM embeddings
        - "icl_full":   baseline + ICL2/ICL3 ESM (1280 each) + 8 biophysical stats each
        - "alpha":      icl_full + 38 AlphaFold structural features
    """
    X_list, y_list, meta = [], [], []
    missing = defaultdict(int)
    for _, row in df.iterrows():
        gid = row["gpcr_id"]
        gpcr_feat = get_gpcr_feat(gpcr_feats, gid)
        gfam = row["g_protein_family"]
        gprot_feat = gprot_feats.get(gfam)
        if gprot_feat is None:
            gprot_feat = gprot_feats.get(gfam.capitalize()) or gprot_feats.get(gfam.upper())
        if gpcr_feat is None or gprot_feat is None:
            missing[f"missing_{gid}_{gfam}"] += 1
            continue

        vec_parts = [np.concatenate([gpcr_feat, gprot_feat])]
        if mode in ("icl_full", "alpha"):
            icl2_esm, icl2_stat, icl3_esm, icl3_stat = get_icl_vector(icl_data, gid, gpcr_feat_dim=1280)
            vec_parts.append(np.concatenate([icl2_esm, icl2_stat, icl3_esm, icl3_stat]))
        if mode == "alpha":
            vec_parts.append(get_alpha_vector(alpha_data, gid))

        X_list.append(np.concatenate(vec_parts))
        y_list.append(int(row["coupling"]))
        meta.append({"gpcr_id": gid, "g_protein_family": gfam, "cluster_id": int(row["cluster_id"])})
    return np.array(X_list), np.array(y_list), meta


# ===================================================================
#  Neural network models
# ===================================================================

class PairedCrossAttentionNet(nn.Module):
    """Cross-attention network for paired GPCR-G protein coupling prediction.

    Architecture:
      - GPCR features projected to hidden_dim via Linear -> LayerNorm -> GELU
      - G-protein features projected to hidden_dim via Linear -> LayerNorm -> GELU
      - Multi-head cross-attention: GPCR as query, G-protein as key/value
      - Fused representation passed through a feed-forward network to scalar logit

    Hyperparameters: hidden_dim=256, num_heads=4, dropout=0.3
    """

    def __init__(self, gpcr_dim=1280, gprot_dim=320, hidden_dim=256, num_heads=4, dropout=0.3):
        super().__init__()
        self.gpcr_proj = nn.Sequential(
            nn.Linear(gpcr_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.gprot_proj = nn.Sequential(
            nn.Linear(gprot_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, gpcr_feat, gprot_feat):
        q = self.gpcr_proj(gpcr_feat).unsqueeze(1)   # (B, 1, hidden)
        kv = self.gprot_proj(gprot_feat).unsqueeze(1)  # (B, 1, hidden)
        attn_out, _ = self.cross_attn(q, kv, kv)       # (B, 1, hidden)
        x = torch.cat([attn_out.squeeze(1), self.gpcr_proj(gpcr_feat)], dim=-1)
        return self.ffn(x).squeeze(-1)


class MLP(nn.Module):
    """Multi-layer perceptron for baseline coupling prediction.

    Architecture: input -> 256 -> 128 -> 1
    Each hidden layer: Linear -> BatchNorm1d -> GELU -> Dropout(0.3)
    """

    def __init__(self, input_dim, hidden_dims=(256, 128), dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ===================================================================
#  PyTorch Datasets
# ===================================================================

class PairDataset(Dataset):
    """Dataset for cross-attention model: splits features into GPCR and G-protein parts."""

    def __init__(self, X, y, gpcr_total_dim):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.gpcr_total_dim = gpcr_total_dim

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        gpcr_feat = x[:self.gpcr_total_dim]
        gprot_feat = x[1280:1600]   # G-protein ESM is always 320-d at fixed offset
        return gpcr_feat, gprot_feat, self.y[idx]


class NumpyDataset(Dataset):
    """Generic dataset returning (features, label) pairs."""

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ===================================================================
#  Training / evaluation helpers
# ===================================================================

def train_ca_epoch(model, loader, optimizer, criterion):
    """Single training epoch for the cross-attention model."""
    model.train()
    total_loss = 0.0
    for gpcr, gprot, y in loader:
        gpcr, gprot, y = gpcr.to(DEVICE), gprot.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(gpcr, gprot)
        loss = criterion(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_ca(model, loader):
    """Evaluate cross-attention model, returning classification metrics."""
    model.eval()
    probs, labels = [], []
    for gpcr, gprot, y in loader:
        gpcr, gprot = gpcr.to(DEVICE), gprot.to(DEVICE)
        out = torch.sigmoid(model(gpcr, gprot))
        probs.append(out.cpu().numpy())
        labels.append(y.numpy())
    probs = np.concatenate(probs)
    labels = np.concatenate(labels)
    preds = (probs >= 0.5).astype(int)
    return {
        "auc": roc_auc_score(labels, probs) if len(set(labels)) >= 2 else float("nan"),
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }


def train_mlp_epoch(model, loader, optimizer, criterion):
    """Single training epoch for the MLP model."""
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(yb)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_mlp(model, loader):
    """Evaluate MLP model, returning classification metrics."""
    model.eval()
    probs, labels = [], []
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        out = torch.sigmoid(model(xb))
        probs.append(out.cpu().numpy())
        labels.append(yb.numpy())
    probs = np.concatenate(probs)
    labels = np.concatenate(labels)
    preds = (probs >= 0.5).astype(int)
    return {
        "auc": roc_auc_score(labels, probs) if len(set(labels)) >= 2 else float("nan"),
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }


# ===================================================================
#  Cluster-aware fold assignment (shared by all CV runners)
# ===================================================================

def assign_cluster_folds(meta, cluster_list, n_folds=5):
    """Partition sequence clusters into *n_folds* folds.

    Greedy assignment: clusters are sorted by descending size and assigned
    to the currently smallest fold, minimising fold-size imbalance.

    Returns
    -------
    fold_clusters : list of list of int
        fold_clusters[f] lists the cluster IDs assigned to fold *f*.
    sample_to_cluster : dict
        Maps sample index (0..n-1) to its cluster ID.
    """
    n = len(meta)
    sample_to_cluster = {i: meta[i]["cluster_id"] for i in range(n)}
    cluster_sizes = defaultdict(int)
    for i in range(n):
        cluster_sizes[sample_to_cluster[i]] += 1

    fold_clusters = [[] for _ in range(n_folds)]
    fold_size = [0] * n_folds
    sorted_cids = sorted(range(len(cluster_list)), key=lambda c: cluster_sizes[c], reverse=True)
    for cid in sorted_cids:
        if cluster_sizes[cid] == 0:
            continue
        target = int(np.argmin(fold_size))
        fold_clusters[target].append(cid)
        fold_size[target] += cluster_sizes[cid]

    return fold_clusters, sample_to_cluster


# ===================================================================
#  Cluster-aware CV runners
# ===================================================================

def _make_train_test_splits(n, sample_to_cluster, fold_clusters, fold_idx):
    """Return train and test sample indices for a given fold."""
    test_clusters = set(fold_clusters[fold_idx])
    test_idx = [i for i in range(n) if sample_to_cluster[i] in test_clusters]
    train_idx = [i for i in range(n) if sample_to_cluster[i] not in test_clusters]
    return train_idx, test_idx


def run_cv_cross_attention(X, y, meta, cluster_list, mode_name, n_folds=5,
                           epochs=80, lr=1e-4, batch_size=32):
    """Cluster-aware 5-fold CV for PairedCrossAttentionNet."""

    n = len(y)
    fold_clusters, sample_to_cluster = assign_cluster_folds(meta, cluster_list, n_folds)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Determine feature dimensions per mode
    if mode_name == "baseline":
        gpcr_total_dim, gprot_dim = 1280, 320
    elif mode_name == "icl_full":
        gpcr_total_dim, gprot_dim = 1280 + 2576, 320
    elif mode_name == "alpha":
        gpcr_total_dim, gprot_dim = 1280 + 2576 + 38, 320
    else:
        raise ValueError(f"Unknown mode: {mode_name}")

    fold_aucs = []
    for f in range(n_folds):
        train_idx, test_idx = _make_train_test_splits(n, sample_to_cluster, fold_clusters, f)
        if len(set(y[test_idx])) < 2:
            fold_aucs.append(float("nan"))
            continue

        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        train_ds = PairDataset(X_train, y_train, gpcr_total_dim)
        test_ds = PairDataset(X_test, y_test, gpcr_total_dim)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size)

        model = PairedCrossAttentionNet(
            gpcr_dim=gpcr_total_dim, gprot_dim=gprot_dim,
            hidden_dim=256, num_heads=4, dropout=0.3,
        ).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.FloatTensor([len(y_train) / max(1, y_train.sum())]).to(DEVICE)
        )

        best_auc = -1.0
        patience_counter = 0
        for epoch in range(epochs):
            train_ca_epoch(model, train_loader, optimizer, criterion)
            val_metrics = eval_ca(model, test_loader)
            if val_metrics["auc"] > best_auc:
                best_auc = val_metrics["auc"]
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= 15:
                break
        fold_aucs.append(best_auc)
        print(f"  Fold {f + 1} best AUC = {best_auc:.4f}")

    valid = [a for a in fold_aucs if not np.isnan(a)]
    return {
        "auc_mean": round(float(np.mean(valid)), 4) if valid else float("nan"),
        "auc_std": round(float(np.std(valid)), 4) if valid else float("nan"),
        "fold_aucs": [round(float(a), 4) for a in fold_aucs],
    }


def run_cv_mlp(X, y, meta, cluster_list, n_folds=5, epochs=80, lr=1e-4, batch_size=32):
    """Cluster-aware 5-fold CV for the MLP baseline."""

    n = len(y)
    fold_clusters, sample_to_cluster = assign_cluster_folds(meta, cluster_list, n_folds)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    fold_aucs = []
    for f in range(n_folds):
        train_idx, test_idx = _make_train_test_splits(n, sample_to_cluster, fold_clusters, f)
        if len(set(y[test_idx])) < 2:
            fold_aucs.append(float("nan"))
            continue

        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        train_loader = DataLoader(NumpyDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(NumpyDataset(X_test, y_test), batch_size=batch_size)

        model = MLP(input_dim=X.shape[1], hidden_dims=(256, 128), dropout=0.3).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        pos_weight = len(y_train) / max(1, y_train.sum())
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.FloatTensor([pos_weight]).to(DEVICE)
        )

        best_auc = -1.0
        patience_counter = 0
        for epoch in range(epochs):
            train_mlp_epoch(model, train_loader, optimizer, criterion)
            val_metrics = eval_mlp(model, test_loader)
            if val_metrics["auc"] > best_auc:
                best_auc = val_metrics["auc"]
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= 15:
                break
        fold_aucs.append(best_auc)
        print(f"  MLP Fold {f + 1} best AUC = {best_auc:.4f}")

    valid = [a for a in fold_aucs if not np.isnan(a)]
    return {
        "auc_mean": round(float(np.mean(valid)), 4) if valid else float("nan"),
        "auc_std": round(float(np.std(valid)), 4) if valid else float("nan"),
        "fold_aucs": [round(float(a), 4) for a in fold_aucs],
    }


def run_cv_sklearn(X, y, meta, cluster_list, model, n_folds=5):
    """Cluster-aware 5-fold CV for sklearn-compatible models (RF, XGBoost)."""

    n = len(y)
    fold_clusters, sample_to_cluster = assign_cluster_folds(meta, cluster_list, n_folds)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    fold_aucs = []
    for f in range(n_folds):
        train_idx, test_idx = _make_train_test_splits(n, sample_to_cluster, fold_clusters, f)
        if len(set(y[test_idx])) < 2:
            fold_aucs.append(float("nan"))
            continue

        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        m = model.__class__(**model.get_params())
        m.fit(X_train, y_train)
        probs = m.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs) if len(set(y_test)) >= 2 else float("nan")
        fold_aucs.append(auc)
        print(f"  {model.__class__.__name__} Fold {f + 1} AUC = {auc:.4f}")

    valid = [a for a in fold_aucs if not np.isnan(a)]
    return {
        "auc_mean": round(float(np.mean(valid)), 4) if valid else float("nan"),
        "auc_std": round(float(np.std(valid)), 4) if valid else float("nan"),
        "fold_aucs": [round(float(a), 4) for a in fold_aucs],
    }


# ===================================================================
#  Top-level training entry points
# ===================================================================

def train_cross_attention(gpcr_feats, gprot_feats, icl_data, alpha_data, df, cluster_list):
    """Train PairedCrossAttentionNet on all three feature modes and save results."""
    print("-" * 50)
    print("  Cross-Attention Training (650M ESM-2 embeddings)")
    print("-" * 50)

    results = {}
    for mode in ["baseline", "icl_full", "alpha"]:
        if mode == "alpha" and not alpha_data:
            continue
        print(f"\n--- Mode: {mode} ---")
        X, y, meta = build_vectors(df, gpcr_feats, gprot_feats, icl_data, alpha_data, mode=mode)
        print(f"Samples: {len(y)}, Features: {X.shape[1]}")
        res = run_cv_cross_attention(X, y, meta, cluster_list, mode_name=mode,
                                     n_folds=5, epochs=80, lr=1e-4, batch_size=32)
        results[mode] = res
        print(f"Cluster-aware CV AUC = {res['auc_mean']:.4f} +/- {res['auc_std']:.4f}")

    with open(CA_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Cross-attention results saved to {CA_OUTPUT_FILE}")
    return results


def train_baselines(gpcr_feats, gprot_feats, icl_data, alpha_data, df, cluster_list,
                    which=None):
    """Train MLP, Random Forest, and/or XGBoost on the icl_full feature set.

    Parameters
    ----------
    which : set of str or None
        Subset of {"mlp", "rf", "xgb"} to train.  None means all three.
    """
    if which is None:
        which = {"mlp", "rf", "xgb"}

    print("-" * 50)
    print("  Baseline Training (650M, ICL-full)")
    print("-" * 50)

    mode = "icl_full"
    X, y, meta = build_vectors(df, gpcr_feats, gprot_feats, icl_data, alpha_data, mode=mode)
    print(f"\nSamples: {len(y)}, Features: {X.shape[1]}")

    results = {}

    if "mlp" in which:
        print("\nTraining MLP...")
        results["mlp"] = run_cv_mlp(X, y, meta, cluster_list, n_folds=5, epochs=80, lr=1e-4, batch_size=32)
        print(f"MLP Cluster-aware CV AUC = {results['mlp']['auc_mean']:.4f} +/- {results['mlp']['auc_std']:.4f}")

    if "rf" in which:
        print("\nTraining Random Forest...")
        rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1)
        results["random_forest"] = run_cv_sklearn(X, y, meta, cluster_list, rf, n_folds=5)
        print(f"Random Forest Cluster-aware CV AUC = {results['random_forest']['auc_mean']:.4f} +/- {results['random_forest']['auc_std']:.4f}")

    if "xgb" in which:
        print("\nTraining XGBoost...")
        pos_weight = len(y) / max(1, y.sum())
        xgb = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=pos_weight,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            use_label_encoder=False,
        )
        results["xgboost"] = run_cv_sklearn(X, y, meta, cluster_list, xgb, n_folds=5)
        print(f"XGBoost Cluster-aware CV AUC = {results['xgboost']['auc_mean']:.4f} +/- {results['xgboost']['auc_std']:.4f}")

    with open(BASELINE_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Baseline results saved to {BASELINE_OUTPUT_FILE}")
    return results


# ===================================================================
#  Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train GPCR-G protein coupling prediction models under cluster-aware 5-fold CV."
    )
    parser.add_argument(
        "--model", type=str, required=True,
        choices=["ca", "mlp", "rf", "xgb", "all"],
        help="Model to train: ca (cross-attention), mlp, rf (random forest), xgb (xgboost), or all"
    )
    args = parser.parse_args()

    print("=" * 70)
    print(f"  Model training: {args.model}")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    # --- Load shared data ---
    df = pd.read_csv(PAIRING_MATRIX_FILE)
    with open(CLUSTERS_FILE) as f:
        clusters_data = json.load(f)
    cluster_list = clusters_data["clusters"]

    gpcr_feats, gprot_feats = load_features()
    icl_data = load_icl_features()
    alpha_data = load_alpha_features()

    # --- Dispatch ---
    if args.model == "ca":
        train_cross_attention(gpcr_feats, gprot_feats, icl_data, alpha_data, df, cluster_list)
    elif args.model == "mlp":
        train_baselines(gpcr_feats, gprot_feats, icl_data, alpha_data, df, cluster_list,
                        which={"mlp"})
    elif args.model == "rf":
        train_baselines(gpcr_feats, gprot_feats, icl_data, alpha_data, df, cluster_list,
                        which={"rf"})
    elif args.model == "xgb":
        train_baselines(gpcr_feats, gprot_feats, icl_data, alpha_data, df, cluster_list,
                        which={"xgb"})
    elif args.model == "all":
        train_cross_attention(gpcr_feats, gprot_feats, icl_data, alpha_data, df, cluster_list)
        train_baselines(gpcr_feats, gprot_feats, icl_data, alpha_data, df, cluster_list)


if __name__ == "__main__":
    main()
