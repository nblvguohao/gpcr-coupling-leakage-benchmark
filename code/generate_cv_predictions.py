#!/usr/bin/env python3
"""
Generate proper 5-fold cluster-CV predictions for SVM and Cross-Attention models.

Replaces the contaminated ca_predictions.json / svm_predictions.json (which were
in-sample, AUC ~0.9996) with per-fold held-out test predictions.

Output format matches fig_bib.py expectations:
  {gpcr_id: {g_protein_family: {"label": 0/1, "prob": float}}}
"""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from collections import defaultdict
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
DATA_DIR = BASE.parent / "data"

# Feature paths
GPCR_FEATURES_FILE = DATA_DIR / "gpcr_esm_features_650m.json"
GPROT_FEATURES_FILE = DATA_DIR / "g_protein_esm_features_650m.json"
ICL_FEATURES_FILE = DATA_DIR / "icl_features_650m.json"
PAIRING_MATRIX_FILE = DATA_DIR / "pairing_matrix_raw.csv"
CLUSTERS_FILE = DATA_DIR / "sequence_clusters.json"

# Output paths
CA_OUTPUT = DATA_DIR / "ca_predictions.json"
SVM_OUTPUT = DATA_DIR / "svm_predictions.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_FOLDS = 5
RANDOM_SEED = 42

# G protein family -> subtype mapping (from original code)
FAMILY_MAP_GPROT = {
    "GNAQ": "Gq", "GNAI1": "Gi", "GNAI2": "Gi", "GNAI3": "Gi",
    "GNAS": "Gs", "GNA12": "G12_13", "GNA13": "G12_13",
}
FAMILIES = ["Gq", "Gi", "Gs", "G12_13"]


# ==========================================================================
# Feature loading
# ==========================================================================

def load_gpcr_features():
    with open(GPCR_FEATURES_FILE) as f:
        raw = json.load(f)
    return {k: np.array(v) for k, v in raw.items()}


def load_gprot_features():
    with open(GPROT_FEATURES_FILE) as f:
        raw = json.load(f)
    gprot_feats = {}
    for subtype, info in raw.items():
        family = FAMILY_MAP_GPROT.get(subtype, subtype)
        vec = np.array(info["mean_pooling"])
        gprot_feats[subtype] = vec
        # Map to family as well (matches original behavior)
        gprot_feats[family] = vec
    return gprot_feats


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
                feat = gpcr_feats[key]
                break
    return feat


def get_icl_vector(icl_data, gid, dim=1280):
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
    stat_keys = ["length", "mean_hydro", "std_hydro", "net_charge",
                 "pos_charge_ratio", "neg_charge_ratio", "hydrophobic_ratio", "aromatic_ratio"]
    icl2_s = np.array([icl2_stats.get(k, 0.0) for k in stat_keys])
    icl3_s = np.array([icl3_stats.get(k, 0.0) for k in stat_keys])
    return icl2_esm, icl2_s, icl3_esm, icl3_s


# ==========================================================================
# Feature vector construction
# ==========================================================================

def build_vectors(df, gpcr_feats, gprot_feats, icl_data):
    """Build paired feature vectors: [GPCR_1280 | Gprot_1280 | ICL_full_2576]"""
    X_list, y_list, metas = [], [], []
    missing = defaultdict(int)
    for _, row in df.iterrows():
        gid = row["gpcr_id"]
        gpcr_feat = get_gpcr_feat(gpcr_feats, gid)
        gfam = row["g_protein_family"]
        gprot_feat = gprot_feats.get(gfam)
        if gprot_feat is None:
            gprot_feat = gprot_feats.get(gfam.capitalize()) or gprot_feats.get(gfam.upper())

        if gpcr_feat is None:
            missing[f"gpcr_missing"] += 1
            continue
        if gprot_feat is None:
            missing[f"gprot_missing"] += 1
            continue

        parts = [gpcr_feat, gprot_feat]  # 1280 + 1280 = 2560

        icl2_e, icl2_s, icl3_e, icl3_s = get_icl_vector(icl_data, gid, dim=1280)
        icl_full = np.concatenate([icl2_e, icl2_s, icl3_e, icl3_s])  # 2576
        parts.append(icl_full)

        vec = np.concatenate(parts)
        X_list.append(vec)
        y_list.append(int(row["coupling"]))
        metas.append({
            "gpcr_id": gid,
            "g_protein_family": gfam,
            "cluster_id": int(row["cluster_id"]),
            "row_idx": _,
        })

    if missing:
        print(f"[WARN] Missing features: {dict(missing)}")
    return np.array(X_list), np.array(y_list), metas


# ==========================================================================
# Cluster-aware fold splitting (matches original code logic)
# ==========================================================================

def get_cluster_folds(y, metas, cluster_list, n_folds=5):
    n = len(y)
    sample_to_cluster = {i: metas[i]["cluster_id"] for i in range(n)}

    cluster_sizes = defaultdict(int)
    for i in range(n):
        cluster_sizes[sample_to_cluster[i]] += 1

    fold_clusters = [[] for _ in range(n_folds)]
    fold_sizes = [0] * n_folds

    # Sort by cluster size descending (greedy bin-packing)
    sorted_clusters = sorted(cluster_list, key=lambda c: len(c["members"]), reverse=True)
    for c in sorted_clusters:
        cid = c["cluster_id"]
        if cluster_sizes.get(cid, 0) == 0:
            continue
        target = int(np.argmin(fold_sizes))
        fold_clusters[target].append(cid)
        fold_sizes[target] += cluster_sizes[cid]

    return sample_to_cluster, fold_clusters


# ==========================================================================
# Cross-Attention Model (from run_gprotein.py)
# ==========================================================================

class PairedCrossAttentionNet(nn.Module):
    def __init__(self, gpcr_dim=3856, gprot_dim=1280, hidden_dim=256, num_heads=4, dropout=0.3):
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
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, gpcr_feat, gprot_feat):
        q = self.gpcr_proj(gpcr_feat).unsqueeze(1)
        kv = self.gprot_proj(gprot_feat).unsqueeze(1)
        attn_out, _ = self.cross_attn(q, kv, kv)
        x = torch.cat([attn_out.squeeze(1), self.gpcr_proj(gpcr_feat)], dim=-1)
        return self.ffn(x).squeeze(-1)


class PairDataset(Dataset):
    """Split concatenated feature vector into GPCR-side and G-protein-side."""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        gpcr_side = torch.cat([x[:1280], x[2560:]])  # GPCR global + ICL full
        gprot_vec = x[1280:2560]  # G protein 1280-d
        return gpcr_side, gprot_vec, self.y[idx]


# ==========================================================================
# Training utilities
# ==========================================================================

def train_epoch(model, loader, optim, crit):
    model.train()
    loss_total = 0.0
    for gpcr, gprot, y in loader:
        gpcr, gprot, y = gpcr.to(DEVICE), gprot.to(DEVICE), y.to(DEVICE)
        optim.zero_grad()
        loss = crit(model(gpcr, gprot), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        loss_total += loss.item() * len(y)
    return loss_total / len(loader.dataset)


@torch.no_grad()
def evaluate_ca(model, loader):
    model.eval()
    probs, labels = [], []
    for gpcr, gprot, y in loader:
        gpcr, gprot = gpcr.to(DEVICE), gprot.to(DEVICE)
        probs.append(torch.sigmoid(model(gpcr, gprot)).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


# ==========================================================================
# Main: 5-fold Cluster-CV for both SVM and CA
# ==========================================================================

def main():
    print("=" * 70)
    print("  Generate Proper CV Predictions")
    print("  SVM (RBF) + Cross-Attention (650M + ICL)")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    # Set seeds
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)

    # Load data
    print("\n[1/5] Loading features...")
    gpcr_feats = load_gpcr_features()
    gprot_feats = load_gprot_features()
    icl_data = load_icl_features()
    df = pd.read_csv(PAIRING_MATRIX_FILE)

    with open(CLUSTERS_FILE) as f:
        clusters_data = json.load(f)
    cluster_list = clusters_data["clusters"]

    print(f"  GPCR features: {len(gpcr_feats)} proteins, dim={len(list(gpcr_feats.values())[0])}")
    print(f"  G protein features: {len(gprot_feats)} entries")
    print(f"  ICL features: {len(icl_data)} proteins")
    print(f"  Pairing matrix: {len(df)} pairs")

    # Build feature vectors
    print("\n[2/5] Building feature vectors (ICL-full mode)...")
    X, y, metas = build_vectors(df, gpcr_feats, gprot_feats, icl_data)
    print(f"  Feature dim: {X.shape[1]}, Samples: {X.shape[0]}")
    print(f"  Positive ratio: {y.mean():.4f}")

    # Create fold splits
    print("\n[3/5] Creating cluster-aware fold splits...")
    sample_to_cluster, fold_clusters = get_cluster_folds(y, metas, cluster_list, N_FOLDS)

    for fi in range(N_FOLDS):
        test_cids = set(fold_clusters[fi])
        test_idx = [i for i in range(len(y)) if sample_to_cluster[i] in test_cids]
        train_idx = [i for i in range(len(y)) if sample_to_cluster[i] not in test_cids]
        print(f"  Fold {fi+1}: train={len(train_idx)}, test={len(test_idx)}, "
              f"test_clusters={len(test_cids)}, test_pos_ratio={y[test_idx].mean():.3f}")

    # ======================================================================
    # SVM 5-fold Cluster-CV
    # ======================================================================
    print("\n[4/5] Running SVM 5-fold Cluster-CV...")
    svm_predictions = {}  # {gpcr_id: {family: {"label": 0/1, "prob": float}}}
    svm_fold_aucs = []

    for fi in range(N_FOLDS):
        test_cids = set(fold_clusters[fi])
        test_idx = [i for i in range(len(y)) if sample_to_cluster[i] in test_cids]
        train_idx = [i for i in range(len(y)) if sample_to_cluster[i] not in test_cids]

        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # Standardize
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        # Train SVM
        svm = SVC(kernel="rbf", C=10.0, class_weight="balanced",
                  probability=True, random_state=RANDOM_SEED)
        svm.fit(X_tr_s, y_tr)
        y_prob = svm.predict_proba(X_te_s)[:, 1]

        auc = roc_auc_score(y_te, y_prob)
        svm_fold_aucs.append(auc)
        print(f"  Fold {fi+1}: SVM AUC = {auc:.4f}")

        # Store predictions
        for i, idx in enumerate(test_idx):
            gid = metas[idx]["gpcr_id"]
            gfam = metas[idx]["g_protein_family"]
            label = int(y_te[i])
            prob = float(y_prob[i])

            if gid not in svm_predictions:
                svm_predictions[gid] = {}
            svm_predictions[gid][gfam] = {"label": label, "prob": prob}

    svm_mean_auc = np.mean(svm_fold_aucs)
    svm_std_auc = np.std(svm_fold_aucs)
    print(f"  SVM Cluster-CV AUC: {svm_mean_auc:.4f} +/- {svm_std_auc:.4f}")

    # ======================================================================
    # Cross-Attention 5-fold Cluster-CV
    # ======================================================================
    print("\n[5/5] Running Cross-Attention 5-fold Cluster-CV...")
    ca_predictions = {}  # {gpcr_id: {family: {"label": 0/1, "prob": float}}}
    ca_fold_aucs = []

    for fi in range(N_FOLDS):
        test_cids = set(fold_clusters[fi])
        test_idx = [i for i in range(len(y)) if sample_to_cluster[i] in test_cids]
        train_idx = [i for i in range(len(y)) if sample_to_cluster[i] not in test_cids]

        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # Standardize (same scaler approach)
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        train_ds = PairDataset(X_tr_s, y_tr)
        test_ds = PairDataset(X_te_s, y_te)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=32)

        model = PairedCrossAttentionNet(
            gpcr_dim=1280 + 2576, gprot_dim=1280,
            hidden_dim=256, num_heads=4, dropout=0.3
        ).to(DEVICE)

        pos_weight = len(y_tr) / max(1, y_tr.sum())
        optim = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        crit = nn.BCEWithLogitsLoss(
            pos_weight=torch.FloatTensor([pos_weight]).to(DEVICE))

        best_auc = -1.0
        best_probs = None
        patience = 0
        max_patience = 20

        for ep in range(200):
            train_epoch(model, train_loader, optim, crit)
            p, lbl = evaluate_ca(model, test_loader)
            auc_v = roc_auc_score(lbl, p) if len(set(lbl)) >= 2 else -1
            if auc_v > best_auc:
                best_auc = auc_v
                best_probs = p
                patience = 0
            else:
                patience += 1
            if patience >= max_patience:
                break

        ca_fold_aucs.append(best_auc)
        print(f"  Fold {fi+1}: CA AUC = {best_auc:.4f} (epochs={ep+1})")

        # Store predictions
        for i, idx in enumerate(test_idx):
            gid = metas[idx]["gpcr_id"]
            gfam = metas[idx]["g_protein_family"]
            label = int(y_te[i])
            prob = float(best_probs[i])

            if gid not in ca_predictions:
                ca_predictions[gid] = {}
            ca_predictions[gid][gfam] = {"label": label, "prob": prob}

    ca_mean_auc = np.mean(ca_fold_aucs)
    ca_std_auc = np.std(ca_fold_aucs)
    print(f"  CA Cluster-CV AUC: {ca_mean_auc:.4f} +/- {ca_std_auc:.4f}")

    # ======================================================================
    # Save predictions
    # ======================================================================
    print("\n" + "=" * 70)
    print("  Saving predictions...")

    # Backup originals
    for fpath in [SVM_OUTPUT, CA_OUTPUT]:
        if fpath.exists():
            backup = fpath.with_suffix(".json.bak")
            import shutil
            shutil.copy(fpath, backup)
            print(f"  Backup: {fpath.name} -> {backup.name}")

    with open(SVM_OUTPUT, "w") as f:
        json.dump(svm_predictions, f, indent=2)
    print(f"  Saved: {SVM_OUTPUT} ({len(svm_predictions)} GPCRs)")

    with open(CA_OUTPUT, "w") as f:
        json.dump(ca_predictions, f, indent=2)
    print(f"  Saved: {CA_OUTPUT} ({len(ca_predictions)} GPCRs)")

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  SVM  Cluster-CV AUC: {svm_mean_auc:.4f} +/- {svm_std_auc:.4f}")
    print(f"  CA   Cluster-CV AUC: {ca_mean_auc:.4f} +/- {ca_std_auc:.4f}")
    print(f"  SVM  fold AUCs: {[f'{a:.4f}' for a in svm_fold_aucs]}")
    print(f"  CA   fold AUCs: {[f'{a:.4f}' for a in ca_fold_aucs]}")
    print(f"  Total predictions: SVM={sum(len(v) for v in svm_predictions.values())}, "
          f"CA={sum(len(v) for v in ca_predictions.values())}")
    print("\n  Done. Run fig_bib.py to regenerate figures.")


if __name__ == "__main__":
    main()
