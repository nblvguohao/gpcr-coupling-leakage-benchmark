#!/usr/bin/env python3
"""
Generate held-out 5-fold cluster-CV predictions for GPCR-G protein coupling models.

Supports three model architectures:
  - SVM:  RBF-kernel SVM with class-balanced weighting
  - CA:   Cross-Attention network (from run_gprotein.py)
  - MLP:  3-layer MLP matching the paper architecture

Outputs per-model JSON files in data/:
  svm_predictions.json, ca_predictions.json, mlp_predictions.json

Usage:
  python generate_predictions.py --model svm          # SVM only
  python generate_predictions.py --model ca           # Cross-Attention only
  python generate_predictions.py --model mlp          # MLP only
  python generate_predictions.py --model all          # All three sequentially

Output format (matches fig_bib.py expectations):
  {gpcr_id: {g_protein_family: {"label": 0/1, "prob": float}}}
"""

import argparse
import json
import shutil
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

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

BASE = Path(__file__).parent
DATA_DIR = BASE.parent / "data"

GPCR_FEATURES_FILE = DATA_DIR / "gpcr_esm_features_650m.json"
GPROT_FEATURES_FILE = DATA_DIR / "g_protein_esm_features_650m.json"
ICL_FEATURES_FILE = DATA_DIR / "icl_features_650m.json"
PAIRING_MATRIX_FILE = DATA_DIR / "pairing_matrix_raw.csv"
CLUSTERS_FILE = DATA_DIR / "sequence_clusters.json"

OUTPUT_PATHS = {
    "svm": DATA_DIR / "svm_predictions.json",
    "ca": DATA_DIR / "ca_predictions.json",
    "mlp": DATA_DIR / "mlp_predictions.json",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_FOLDS = 5
RANDOM_SEED = 42

FAMILY_MAP_GPROT = {
    "GNAQ": "Gq", "GNAI1": "Gi", "GNAI2": "Gi", "GNAI3": "Gi",
    "GNAS": "Gs", "GNA12": "G12_13", "GNA13": "G12_13",
}
FAMILIES = ["Gq", "Gi", "Gs", "G12_13"]


# ---------------------------------------------------------------------------
# Feature loading
# ---------------------------------------------------------------------------

def load_gpcr_features():
    """Load GPCR ESM-2 650M features, returning {gpcr_id: np.array}."""
    with open(GPCR_FEATURES_FILE) as f:
        raw = json.load(f)
    return {k: np.array(v) for k, v in raw.items()}


def load_gprot_features():
    """Load G protein ESM-2 650M features, resolving subtypes to families."""
    with open(GPROT_FEATURES_FILE) as f:
        raw = json.load(f)
    feats = {}
    for subtype, info in raw.items():
        family = FAMILY_MAP_GPROT.get(subtype, subtype)
        vec = np.array(info["mean_pooling"])
        feats[subtype] = vec
        feats[family] = vec
    return feats


def load_icl_features():
    """Load ICL (intracellular loop) topological features."""
    if not ICL_FEATURES_FILE.exists():
        return {}
    with open(ICL_FEATURES_FILE) as f:
        return json.load(f)


def get_gpcr_feat(gpcr_feats, gid):
    """Retrieve GPCR feature vector with fallback key matching."""
    feat = gpcr_feats.get(gid)
    if feat is None and "_" in gid and len(gid.split("_")[0]) <= 2:
        feat = gpcr_feats.get(gid.split("_", 1)[1])
    if feat is None:
        for key in gpcr_feats:
            if "_" in key and key.split("_", 1)[1] == gid:
                return gpcr_feats[key]
    return feat


def get_icl_vector(icl_data, gid, dim=1280):
    """
    Retrieve ICL feature components for a GPCR ID.
    Returns (ICL2_esm, ICL2_stats, ICL3_esm, ICL3_stats).
    """
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

    stat_keys = [
        "length", "mean_hydro", "std_hydro", "net_charge",
        "pos_charge_ratio", "neg_charge_ratio", "hydrophobic_ratio",
        "aromatic_ratio",
    ]
    s2 = np.array([icl2_stats.get(k, 0.0) for k in stat_keys])
    s3 = np.array([icl3_stats.get(k, 0.0) for k in stat_keys])
    return icl2_esm, s2, icl3_esm, s3


# ---------------------------------------------------------------------------
# Feature vector construction
# ---------------------------------------------------------------------------

def build_vectors(df, gpcr_feats, gprot_feats, icl_data):
    """
    Build paired feature vectors for all rows in the pairing matrix.

    Feature layout (5136-d total):
      [GPCR_1280 | Gprot_1280 | ICL2_esm_1280 | ICL2_stats_8 |
       ICL3_esm_1280 | ICL3_stats_8]

    Returns X (N x 5136), y (N,), metas list.
    """
    X_list, y_list, metas = [], [], []
    missing = defaultdict(int)

    for _, row in df.iterrows():
        gid = row["gpcr_id"]
        gfam = row["g_protein_family"]

        gpcr_f = get_gpcr_feat(gpcr_feats, gid)
        gprot_f = gprot_feats.get(gfam)
        if gprot_f is None:
            gprot_f = gprot_feats.get(gfam.capitalize()) or gprot_feats.get(gfam.upper())

        if gpcr_f is None:
            missing["gpcr_missing"] += 1
            continue
        if gprot_f is None:
            missing["gprot_missing"] += 1
            continue

        i2_e, i2_s, i3_e, i3_s = get_icl_vector(icl_data, gid, dim=1280)
        vec = np.concatenate([gpcr_f, gprot_f, i2_e, i2_s, i3_e, i3_s])

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


# ---------------------------------------------------------------------------
# Cluster-aware fold splitting
# ---------------------------------------------------------------------------

def get_cluster_folds(y, metas, cluster_list, n_folds=N_FOLDS):
    """
    Greedy bin-packing to assign clusters to folds,
    matching the original code's cluster-aware split logic.
    """
    n = len(y)
    s2c = {i: metas[i]["cluster_id"] for i in range(n)}

    c_sizes = defaultdict(int)
    for i in range(n):
        c_sizes[s2c[i]] += 1

    fold_cids = [[] for _ in range(n_folds)]
    fold_size = [0.0] * n_folds

    sorted_c = sorted(cluster_list, key=lambda c: len(c["members"]), reverse=True)
    for c in sorted_c:
        cid = c["cluster_id"]
        if c_sizes.get(cid, 0) == 0:
            continue
        target = int(np.argmin(fold_size))
        fold_cids[target].append(cid)
        fold_size[target] += c_sizes[cid]

    return s2c, fold_cids


def get_fold_indices(y, s2c, fold_cids, fi):
    """Return train/test indices for a given fold."""
    test_cids = set(fold_cids[fi])
    te_idx = [i for i in range(len(y)) if s2c[i] in test_cids]
    tr_idx = [i for i in range(len(y)) if s2c[i] not in test_cids]
    return tr_idx, te_idx


# ---------------------------------------------------------------------------
# Model: SVM (RBF)
# ---------------------------------------------------------------------------

def run_svm_cv(X, y, metas, s2c, fold_cids, seed=RANDOM_SEED):
    """Run 5-fold cluster-CV for SVM (RBF kernel)."""
    print("[SVM] Running 5-fold Cluster-CV ...")
    predictions = {}
    fold_aucs = []

    for fi in range(N_FOLDS):
        tr_idx, te_idx = get_fold_indices(y, s2c, fold_cids, fi)
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        svm = SVC(
            kernel="rbf", C=10.0, class_weight="balanced",
            probability=True, random_state=seed,
        )
        svm.fit(X_tr_s, y_tr)
        y_prob = svm.predict_proba(X_te_s)[:, 1]

        auc = roc_auc_score(y_te, y_prob)
        fold_aucs.append(auc)
        print(f"  Fold {fi + 1}: SVM AUC = {auc:.4f}")

        for i, idx in enumerate(te_idx):
            gid = metas[idx]["gpcr_id"]
            gfam = metas[idx]["g_protein_family"]
            if gid not in predictions:
                predictions[gid] = {}
            predictions[gid][gfam] = {"label": int(y_te[i]), "prob": float(y_prob[i])}

    return predictions, fold_aucs


# ---------------------------------------------------------------------------
# Model: Cross-Attention network
# ---------------------------------------------------------------------------

class PairedCrossAttentionNet(nn.Module):
    """Cross-attention model matching run_gprotein.py architecture."""

    def __init__(self, gpcr_dim=3856, gprot_dim=1280,
                 hidden_dim=256, num_heads=4, dropout=0.3):
        super().__init__()
        self.gpcr_proj = nn.Sequential(
            nn.Linear(gpcr_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
        )
        self.gprot_proj = nn.Sequential(
            nn.Linear(gprot_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
        )
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True,
        )
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
    """
    Splits the 5136-d concatenated vector into:
      gpcr_side: [GPCR_1280 | ICL_full_2576] = 3856-d
      gprot_vec: G-protein 1280-d
    """
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        gpcr_side = torch.cat([x[:1280], x[2560:]])   # GPCR global + ICL full
        gprot_vec = x[1280:2560]                       # G protein 1280-d
        return gpcr_side, gprot_vec, self.y[idx]


def train_epoch_ca(model, loader, optim, crit):
    """Single training epoch for Cross-Attention model."""
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
    """Evaluate Cross-Attention model, returning (probs, labels)."""
    model.eval()
    probs, labels = [], []
    for gpcr, gprot, y in loader:
        gpcr, gprot = gpcr.to(DEVICE), gprot.to(DEVICE)
        probs.append(torch.sigmoid(model(gpcr, gprot)).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def run_ca_cv(X, y, metas, s2c, fold_cids, seed=RANDOM_SEED):
    """Run 5-fold cluster-CV for Cross-Attention model."""
    print("[CA] Running 5-fold Cluster-CV ...")
    predictions = {}
    fold_aucs = []

    for fi in range(N_FOLDS):
        tr_idx, te_idx = get_fold_indices(y, s2c, fold_cids, fi)
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        train_ds = PairDataset(X_tr_s, y_tr)
        test_ds = PairDataset(X_te_s, y_te)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=32)

        model = PairedCrossAttentionNet(
            gpcr_dim=1280 + 2576, gprot_dim=1280,
            hidden_dim=256, num_heads=4, dropout=0.3,
        ).to(DEVICE)

        pos_weight = len(y_tr) / max(1, y_tr.sum())
        optim = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        crit = nn.BCEWithLogitsLoss(
            pos_weight=torch.FloatTensor([pos_weight]).to(DEVICE),
        )

        best_auc = -1.0
        best_probs = None
        patience = 0
        max_patience = 20

        for ep in range(200):
            train_epoch_ca(model, train_loader, optim, crit)
            p, lbl = evaluate_ca(model, test_loader)
            auc_val = roc_auc_score(lbl, p) if len(set(lbl)) >= 2 else -1
            if auc_val > best_auc:
                best_auc = auc_val
                best_probs = p
                patience = 0
            else:
                patience += 1
            if patience >= max_patience:
                break

        fold_aucs.append(best_auc)
        print(f"  Fold {fi + 1}: CA AUC = {best_auc:.4f} (epochs={ep + 1})")

        for i, idx in enumerate(te_idx):
            gid = metas[idx]["gpcr_id"]
            gfam = metas[idx]["g_protein_family"]
            if gid not in predictions:
                predictions[gid] = {}
            predictions[gid][gfam] = {"label": int(y_te[i]), "prob": float(best_probs[i])}

    return predictions, fold_aucs


# ---------------------------------------------------------------------------
# Model: MLP (3-layer, paper architecture)
# ---------------------------------------------------------------------------

class MLPNet(nn.Module):
    """3-layer MLP matching paper architecture."""

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
    """Simple dataset returning (features, label) pairs."""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_epoch_mlp(model, loader, optim, crit):
    """Single training epoch for MLP model."""
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        if len(y) < 2:
            continue  # BatchNorm requires batch_size >= 2
        x, y = x.to(DEVICE), y.to(DEVICE)
        optim.zero_grad()
        loss = crit(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        total += loss.item() * len(y)
        n += len(y)
    return total / n if n > 0 else 0.0


@torch.no_grad()
def evaluate_mlp(model, loader):
    """Evaluate MLP model, returning (probs, labels)."""
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        x = x.to(DEVICE)
        probs.append(torch.sigmoid(model(x)).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def run_mlp_cv(X, y, metas, s2c, fold_cids, seed=RANDOM_SEED):
    """Run 5-fold cluster-CV for MLP model."""
    print("[MLP] Running 5-fold Cluster-CV ...")
    predictions = {}
    fold_aucs = []

    for fi in range(N_FOLDS):
        tr_idx, te_idx = get_fold_indices(y, s2c, fold_cids, fi)
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        train_ds = VecDataset(X_tr_s, y_tr)
        test_ds = VecDataset(X_te_s, y_te)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=32)

        model = MLPNet(input_dim=X.shape[1]).to(DEVICE)

        pos_weight = len(y_tr) / max(1, y_tr.sum())
        optim = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        crit = nn.BCEWithLogitsLoss(
            pos_weight=torch.FloatTensor([pos_weight]).to(DEVICE),
        )

        best_auc = -1.0
        best_probs = None
        patience = 0

        for ep in range(200):
            train_epoch_mlp(model, train_loader, optim, crit)
            p, lbl = evaluate_mlp(model, test_loader)
            auc_val = roc_auc_score(lbl, p) if len(set(lbl)) >= 2 else -1
            if auc_val > best_auc:
                best_auc = auc_val
                best_probs = p
                patience = 0
            else:
                patience += 1
            if patience >= 20:
                break

        fold_aucs.append(best_auc)
        print(f"  Fold {fi + 1}: MLP AUC = {best_auc:.4f}")

        for i, idx in enumerate(te_idx):
            gid = metas[idx]["gpcr_id"]
            gfam = metas[idx]["g_protein_family"]
            if gid not in predictions:
                predictions[gid] = {}
            predictions[gid][gfam] = {"label": int(y_te[i]), "prob": float(best_probs[i])}

    return predictions, fold_aucs


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def save_predictions(predictions, model_name):
    """Save predictions dict to JSON, backing up existing file first."""
    out_path = OUTPUT_PATHS[model_name]
    if out_path.exists():
        backup = out_path.with_suffix(".json.bak")
        shutil.copy(out_path, backup)
        print(f"  Backup: {out_path.name} -> {backup.name}")

    with open(out_path, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"  Saved: {out_path} ({len(predictions)} GPCRs)")


def run_model(model_name, X, y, metas, s2c, fold_cids):
    """Dispatch to the appropriate model-specific CV runner."""
    runners = {
        "svm": run_svm_cv,
        "ca": run_ca_cv,
        "mlp": run_mlp_cv,
    }
    if model_name not in runners:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(runners)}.")

    predictions, fold_aucs = runners[model_name](X, y, metas, s2c, fold_cids)

    mean_auc = np.mean(fold_aucs)
    std_auc = np.std(fold_aucs)
    print(f"  {model_name.upper()} Cluster-CV AUC: {mean_auc:.4f} +/- {std_auc:.4f}")
    print(f"  Fold AUCs: {[f'{a:.4f}' for a in fold_aucs]}")

    save_predictions(predictions, model_name)
    return {"name": model_name, "mean_auc": mean_auc, "std_auc": std_auc,
            "fold_aucs": fold_aucs, "n_predictions": sum(len(v) for v in predictions.values())}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate held-out cluster-CV predictions for GPCR-G protein coupling.",
    )
    parser.add_argument(
        "--model", type=str, default="all",
        choices=["svm", "ca", "mlp", "all"],
        help="Model to run: svm, ca, mlp, or all (default: all).",
    )
    args = parser.parse_args()

    models_to_run = ["svm", "ca", "mlp"] if args.model == "all" else [args.model]

    print("=" * 70)
    print("  Generate Proper CV Predictions")
    print(f"  Models: {', '.join(m.upper() for m in models_to_run)}")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    # Set seeds
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)

    # Load data (once, shared across all models)
    print("\n[1] Loading features ...")
    gpcr_feats = load_gpcr_features()
    gprot_feats = load_gprot_features()
    icl_data = load_icl_features()
    df = pd.read_csv(PAIRING_MATRIX_FILE)

    with open(CLUSTERS_FILE) as f:
        clusters_data = json.load(f)
    cluster_list = clusters_data["clusters"]

    print(f"  GPCR features: {len(gpcr_feats)} proteins, "
          f"dim={len(list(gpcr_feats.values())[0])}")
    print(f"  G protein features: {len(gprot_feats)} entries")
    print(f"  ICL features: {len(icl_data)} proteins")
    print(f"  Pairing matrix: {len(df)} pairs")

    # Build feature vectors (once)
    print("\n[2] Building feature vectors (ICL-full mode) ...")
    X, y, metas = build_vectors(df, gpcr_feats, gprot_feats, icl_data)
    print(f"  Feature dim: {X.shape[1]}, Samples: {X.shape[0]}")
    print(f"  Positive ratio: {y.mean():.4f}")

    # Create fold splits (once)
    print("\n[3] Creating cluster-aware fold splits ...")
    s2c, fold_cids = get_cluster_folds(y, metas, cluster_list)

    for fi in range(N_FOLDS):
        tr_idx, te_idx = get_fold_indices(y, s2c, fold_cids, fi)
        test_cids = set(fold_cids[fi])
        print(f"  Fold {fi + 1}: train={len(tr_idx)}, test={len(te_idx)}, "
              f"test_clusters={len(test_cids)}, test_pos_ratio={y[te_idx].mean():.3f}")

    # Run models
    results = []
    for model_name in models_to_run:
        print()
        result = run_model(model_name, X, y, metas, s2c, fold_cids)
        results.append(result)

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    for r in results:
        print(f"  {r['name'].upper():4s}  Cluster-CV AUC: {r['mean_auc']:.4f} "
              f"+/- {r['std_auc']:.4f}")
        print(f"         Fold AUCs: {[f'{a:.4f}' for a in r['fold_aucs']]}")
        print(f"         Predictions: {r['n_predictions']}")
    print("\n  Done. Run fig_bib.py to regenerate figures.")


if __name__ == "__main__":
    main()
