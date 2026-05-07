#!/usr/bin/env python3
"""
Train additional baseline models (MLP, Random Forest, XGBoost) on the 650M ICL-full
configuration for fair comparison with SVM and Cross-Attention.
"""

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

BASE = Path(__file__).parent
DATA_DIR = BASE / "paired_dataset"
GPCR_FEATURES_FILE = BASE / "server_sync" / "extended_data" / "features" / "esm_features_650m_meanpool.json"
G_PROTEIN_FEATURES_FILE = DATA_DIR / "g_protein_esm_features.json"
ICL_FEATURES_FILE = DATA_DIR / "icl_features_650m.json"
ALPHA_FEATURES_FILE = DATA_DIR / "alphafold_icl_features.json"
PAIRING_MATRIX_FILE = DATA_DIR / "pairing_matrix_raw.csv"
CLUSTERS_FILE = DATA_DIR / "sequence_clusters.json"
OUTPUT_FILE = DATA_DIR / "paired_baselines_650m_results.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Feature loading utilities (identical to train_paired_cross_attention_650m.py)
# ---------------------------------------------------------------------------

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
    if not ICL_FEATURES_FILE.exists():
        return {}
    with open(ICL_FEATURES_FILE) as f:
        return json.load(f)


def load_alpha_features():
    if not ALPHA_FEATURES_FILE.exists():
        return {}
    with open(ALPHA_FEATURES_FILE) as f:
        return json.load(f)


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
    stat_keys = ["length", "mean_hydro", "std_hydro", "net_charge", "pos_charge_ratio",
                 "neg_charge_ratio", "hydrophobic_ratio", "aromatic_ratio"]
    icl2_stat_vec = np.array([icl2_stats.get(k, 0.0) for k in stat_keys])
    icl3_stat_vec = np.array([icl3_stats.get(k, 0.0) for k in stat_keys])
    return icl2_esm, icl2_stat_vec, icl3_esm, icl3_stat_vec


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


def build_vectors(df, gpcr_feats, gprot_feats, icl_data, alpha_data, mode="icl_full"):
    X_list, y_list, meta = [], [], []
    for _, row in df.iterrows():
        gid = row["gpcr_id"]
        gpcr_feat = get_gpcr_feat(gpcr_feats, gid)
        gfam = row["g_protein_family"]
        gprot_feat = gprot_feats.get(gfam)
        if gprot_feat is None:
            gprot_feat = gprot_feats.get(gfam.capitalize()) or gprot_feats.get(gfam.upper())
        if gpcr_feat is None or gprot_feat is None:
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


# ---------------------------------------------------------------------------
# MLP Model
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(512, 256), dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class NumpyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_mlp_epoch(model, loader, optimizer, criterion):
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


# ---------------------------------------------------------------------------
# Cluster-aware CV runners
# ---------------------------------------------------------------------------

def run_cluster_cv_mlp(X, y, meta, cluster_list, n_folds=5, epochs=80, lr=1e-4, batch_size=32):
    n = len(y)
    sample_to_cluster = {i: meta[i]["cluster_id"] for i in range(n)}
    cluster_sizes = defaultdict(int)
    for i in range(n):
        cid = sample_to_cluster[i]
        cluster_sizes[cid] += 1

    fold_clusters = [[] for _ in range(n_folds)]
    fold_size = [0] * n_folds
    sorted_cids = sorted(range(len(cluster_list)), key=lambda c: cluster_sizes[c], reverse=True)
    for cid in sorted_cids:
        if cluster_sizes[cid] == 0:
            continue
        target = int(np.argmin(fold_size))
        fold_clusters[target].append(cid)
        fold_size[target] += cluster_sizes[cid]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    fold_aucs = []
    for f in range(n_folds):
        test_clusters = set(fold_clusters[f])
        test_idx = [i for i in range(n) if sample_to_cluster[i] in test_clusters]
        train_idx = [i for i in range(n) if sample_to_cluster[i] not in test_clusters]
        if len(set(y[test_idx])) < 2:
            fold_aucs.append(float("nan"))
            continue

        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        train_loader = DataLoader(NumpyDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(NumpyDataset(X_test, y_test), batch_size=batch_size)

        model = MLP(input_dim=X.shape[1], hidden_dims=(512, 256), dropout=0.3).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        pos_weight = len(y_train) / max(1, y_train.sum())
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([pos_weight]).to(DEVICE))

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
        print(f"  MLP Fold {f+1} best AUC = {best_auc:.4f}")

    valid = [a for a in fold_aucs if not np.isnan(a)]
    return {
        "auc_mean": round(float(np.mean(valid)), 4) if valid else float("nan"),
        "auc_std": round(float(np.std(valid)), 4) if valid else float("nan"),
        "fold_aucs": [round(float(a), 4) for a in fold_aucs],
    }


def run_cluster_cv_sklearn(X, y, meta, cluster_list, model, n_folds=5):
    n = len(y)
    sample_to_cluster = {i: meta[i]["cluster_id"] for i in range(n)}
    cluster_sizes = defaultdict(int)
    for i in range(n):
        cid = sample_to_cluster[i]
        cluster_sizes[cid] += 1

    fold_clusters = [[] for _ in range(n_folds)]
    fold_size = [0] * n_folds
    sorted_cids = sorted(range(len(cluster_list)), key=lambda c: cluster_sizes[c], reverse=True)
    for cid in sorted_cids:
        if cluster_sizes[cid] == 0:
            continue
        target = int(np.argmin(fold_size))
        fold_clusters[target].append(cid)
        fold_size[target] += cluster_sizes[cid]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    fold_aucs = []
    for f in range(n_folds):
        test_clusters = set(fold_clusters[f])
        test_idx = [i for i in range(n) if sample_to_cluster[i] in test_clusters]
        train_idx = [i for i in range(n) if sample_to_cluster[i] not in test_clusters]
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
        print(f"  {model.__class__.__name__} Fold {f+1} AUC = {auc:.4f}")

    valid = [a for a in fold_aucs if not np.isnan(a)]
    return {
        "auc_mean": round(float(np.mean(valid)), 4) if valid else float("nan"),
        "auc_std": round(float(np.std(valid)), 4) if valid else float("nan"),
        "fold_aucs": [round(float(a), 4) for a in fold_aucs],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  Baseline Training on Paired Dataset (650M, ICL-full)")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    df = pd.read_csv(PAIRING_MATRIX_FILE)
    with open(CLUSTERS_FILE) as f:
        clusters_data = json.load(f)
    cluster_list = clusters_data["clusters"]

    gpcr_feats, gprot_feats = load_features()
    icl_data = load_icl_features()
    alpha_data = load_alpha_features()

    # We evaluate all baselines on the best configuration: 650M ICL-full
    mode = "icl_full"
    print(f"\n--- Mode: {mode} ---")
    X, y, meta = build_vectors(df, gpcr_feats, gprot_feats, icl_data, alpha_data, mode=mode)
    print(f"Samples: {len(y)}, Features: {X.shape[1]}")

    results = {}

    # 1. MLP
    print("\nTraining MLP...")
    results["mlp"] = run_cluster_cv_mlp(X, y, meta, cluster_list, n_folds=5, epochs=80, lr=1e-4, batch_size=32)
    print(f"MLP Cluster-aware CV AUC = {results['mlp']['auc_mean']:.4f} ± {results['mlp']['auc_std']:.4f}")

    # 2. Random Forest
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1)
    results["random_forest"] = run_cluster_cv_sklearn(X, y, meta, cluster_list, rf, n_folds=5)
    print(f"Random Forest Cluster-aware CV AUC = {results['random_forest']['auc_mean']:.4f} ± {results['random_forest']['auc_std']:.4f}")

    # 3. XGBoost
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
    results["xgboost"] = run_cluster_cv_sklearn(X, y, meta, cluster_list, xgb, n_folds=5)
    print(f"XGBoost Cluster-aware CV AUC = {results['xgboost']['auc_mean']:.4f} ± {results['xgboost']['auc_std']:.4f}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
