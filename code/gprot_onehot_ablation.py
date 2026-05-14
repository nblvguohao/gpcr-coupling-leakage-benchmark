#!/usr/bin/env python3
"""
G protein one-hot ablation: Replace G protein sequence embeddings (1280-d)
with 4-dimensional one-hot family identity vectors.

Tests reviewer concern #3: does the paired formulation benefit come from
G protein sequence information, or merely from family identity?

Compares three conditions:
  (a) GPCR features only → multi-task (no G protein info at all)
  (b) GPCR + 4-dim one-hot family → paired with identity only
  (c) GPCR + 1280-d G protein embedding → paired with full sequence info

The difference (c) - (b) isolates the contribution of G protein sequence info.
The difference (b) - (a) isolates the contribution of explicit family identity.
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
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
DATA_DIR = BASE.parent / "data"

GPCR_FEATURES_FILE = DATA_DIR / "gpcr_esm_features_650m.json"
GPROT_FEATURES_FILE = DATA_DIR / "g_protein_esm_features_650m.json"
ICL_FEATURES_FILE = DATA_DIR / "icl_features_650m.json"
PAIRING_MATRIX_FILE = DATA_DIR / "pairing_matrix_raw.csv"
CLUSTERS_FILE = DATA_DIR / "sequence_clusters.json"
OUTPUT_FILE = DATA_DIR / "gprot_onehot_ablation.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_FOLDS = 5
RANDOM_SEED = 42
FAMILIES = ["Gq", "Gi", "Gs", "G12_13"]
FAMILY_MAP_GPROT = {
    "GNAQ": "Gq", "GNAI1": "Gi", "GNAI2": "Gi", "GNAI3": "Gi",
    "GNAS": "Gs", "GNA12": "G12_13", "GNA13": "G12_13",
}


# ==========================================================================
# Feature loading (same as generate_cv_predictions.py)
# ==========================================================================

def load_gpcr_features():
    with open(GPCR_FEATURES_FILE) as f:
        return {k: np.array(v) for k, v in json.load(f).items()}


def load_gprot_features():
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
    with open(ICL_FEATURES_FILE) as f:
        return json.load(f) if ICL_FEATURES_FILE.exists() else {}


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


def build_vectors(df, gpcr_feats, gprot_feats, icl_data, gprot_mode="embedding"):
    """
    gprot_mode: 'embedding' → 1280-d protein embedding
                'onehot' → 4-d family one-hot
                'none' → no G protein info (multi-task style)
    """
    X_list, y_list, metas = [], [], []
    for _, row in df.iterrows():
        gid = row["gpcr_id"]
        gpcr_f = get_gpcr_feat(gpcr_feats, gid)
        gf = row["g_protein_family"]

        # G protein encoding
        if gprot_mode == "embedding":
            gprot_f = gprot_feats.get(gf)
            if gprot_f is None:
                gprot_f = gprot_feats.get(gf.capitalize()) or gprot_feats.get(gf.upper())
            if gprot_f is None: continue
        elif gprot_mode == "onehot":
            gprot_f = np.array([1.0 if gf == f else 0.0 for f in FAMILIES], dtype=np.float64)
        else:  # "none"
            gprot_f = np.array([])

        if gpcr_f is None: continue

        i2_e, i2_s, i3_e, i3_s = get_icl_vector(icl_data, gid, 1280)
        parts = [gpcr_f, gprot_f, i2_e, i2_s, i3_e, i3_s]
        X_list.append(np.concatenate(parts))
        y_list.append(int(row["coupling"]))
        metas.append({"gpcr_id": gid, "family": gf, "cluster_id": int(row["cluster_id"])})

    return np.array(X_list), np.array(y_list), metas


# ==========================================================================
# Cluster folds (same as generate_cv_predictions.py)
# ==========================================================================

def get_cluster_folds(y, metas, cluster_list, n_folds=5):
    n = len(y)
    s2c = {i: metas[i]["cluster_id"] for i in range(n)}
    c_sizes = defaultdict(int)
    for i in range(n): c_sizes[s2c[i]] += 1

    fold_cids = [[] for _ in range(n_folds)]
    fold_size = [0.0] * n_folds
    sorted_c = sorted(cluster_list, key=lambda c: len(c["members"]), reverse=True)
    for c in sorted_c:
        cid = c["cluster_id"]
        if c_sizes.get(cid, 0) == 0: continue
        target = int(np.argmin(fold_size))
        fold_cids[target].append(cid)
        fold_size[target] += c_sizes[cid]
    return s2c, fold_cids


# ==========================================================================
# Cross-Attention Model (adjusted for variable G protein dim)
# ==========================================================================

class PairedCANet(nn.Module):
    def __init__(self, gpcr_dim=3856, gprot_dim=1280, hidden_dim=256, num_heads=4, dropout=0.3):
        super().__init__()
        self.gpcr_proj = nn.Sequential(nn.Linear(gpcr_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.gprot_proj = nn.Sequential(nn.Linear(gprot_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim//2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, 1),
        )

    def forward(self, gpcr_feat, gprot_feat):
        q = self.gpcr_proj(gpcr_feat).unsqueeze(1)
        kv = self.gprot_proj(gprot_feat).unsqueeze(1)
        attn_out, _ = self.cross_attn(q, kv, kv)
        x = torch.cat([attn_out.squeeze(1), self.gpcr_proj(gpcr_feat)], dim=-1)
        return self.ffn(x).squeeze(-1)


class PairDataset(Dataset):
    def __init__(self, X, y, gprot_dim=1280):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.gprot_dim = gprot_dim

    def __len__(self): return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        gpcr_side = torch.cat([x[:1280], x[1280 + self.gprot_dim:]])
        gprot_vec = x[1280:1280 + self.gprot_dim]
        return gpcr_side, gprot_vec, self.y[idx]


def train_epoch(model, loader, optim, crit):
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
def evaluate_ca(model, loader):
    model.eval()
    probs, labels = [], []
    for gpcr, gprot, y in loader:
        gpcr, gprot = gpcr.to(DEVICE), gprot.to(DEVICE)
        probs.append(torch.sigmoid(model(gpcr, gprot)).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def run_svm_cv(X, y, s2c, fold_cids):
    fold_aucs = []
    for fi in range(N_FOLDS):
        test_cids = set(fold_cids[fi])
        te_idx = [i for i in range(len(y)) if s2c[i] in test_cids]
        tr_idx = [i for i in range(len(y)) if s2c[i] not in test_cids]
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr_idx]); X_te = scaler.transform(X[te_idx])
        svm = SVC(kernel="rbf", C=10.0, class_weight="balanced", probability=True,
                  random_state=RANDOM_SEED)
        svm.fit(X_tr, y[tr_idx])
        p = svm.predict_proba(X_te)[:, 1]
        if len(set(y[te_idx])) >= 2:
            fold_aucs.append(roc_auc_score(y[te_idx], p))
    return np.mean(fold_aucs), np.std(fold_aucs)


def run_ca_cv(X, y, s2c, fold_cids, gprot_dim):
    fold_aucs = []
    for fi in range(N_FOLDS):
        test_cids = set(fold_cids[fi])
        te_idx = [i for i in range(len(y)) if s2c[i] in test_cids]
        tr_idx = [i for i in range(len(y)) if s2c[i] not in test_cids]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr_idx]); X_te = scaler.transform(X[te_idx])

        tr_ds = PairDataset(X_tr, y[tr_idx], gprot_dim)
        te_ds = PairDataset(X_te, y[te_idx], gprot_dim)
        tr_ld = DataLoader(tr_ds, batch_size=32, shuffle=True)
        te_ld = DataLoader(te_ds, batch_size=32)

        gpcr_dim = 1280 + 2576  # GPCR global + ICL full
        model = PairedCANet(gpcr_dim=gpcr_dim, gprot_dim=gprot_dim).to(DEVICE)
        pw = len(y[tr_idx]) / max(1, y[tr_idx].sum())
        optim = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        crit = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([pw]).to(DEVICE))

        best_auc, best_p, patience = -1.0, None, 0
        for ep in range(200):
            train_epoch(model, tr_ld, optim, crit)
            p, lbl = evaluate_ca(model, te_ld)
            auc = roc_auc_score(lbl, p) if len(set(lbl)) >= 2 else -1
            if auc > best_auc:
                best_auc, best_p, patience = auc, p, 0
            else:
                patience += 1
            if patience >= 20: break
        fold_aucs.append(best_auc)
    return np.mean(fold_aucs), np.std(fold_aucs)


# ==========================================================================
# Main
# ==========================================================================

def main():
    print("=" * 70)
    print("  G Protein One-Hot Ablation Experiment")
    print("  Tests: embedding vs onehot vs no G protein info")
    print("=" * 70)

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)

    gpcr_feats = load_gpcr_features()
    gprot_feats = load_gprot_features()
    icl_data = load_icl_features()
    df = pd.read_csv(PAIRING_MATRIX_FILE)
    df = df.dropna(subset=["cluster_id"]).copy()

    with open(CLUSTERS_FILE) as f:
        cluster_list = json.load(f)["clusters"]

    print(f"  GPCR features: {len(gpcr_feats)}, G protein: {len(gprot_feats)}, Pairs: {len(df)}")

    configs = [
        ("embedding (1280-d)", "embedding", 1280),
        ("onehot (4-d)", "onehot", 4),
    ]

    results = {}
    for name, mode, gdim in configs:
        print(f"\n{'='*70}")
        print(f"  Config: {name}")
        print(f"{'='*70}")
        X, y, metas = build_vectors(df, gpcr_feats, gprot_feats, icl_data, gprot_mode=mode)
        print(f"  Feature dim: {X.shape[1]}, Samples: {len(y)}, Pos ratio: {y.mean():.3f}")

        s2c, fold_cids = get_cluster_folds(y, metas, cluster_list)

        # SVM
        svm_auc, svm_std = run_svm_cv(X, y, s2c, fold_cids)
        print(f"  SVM (RBF):     AUC = {svm_auc:.4f} +/- {svm_std:.4f}")

        # CA
        ca_auc, ca_std = run_ca_cv(X, y, s2c, fold_cids, gdim)
        print(f"  Cross-Attn:   AUC = {ca_auc:.4f} +/- {ca_std:.4f}")

        results[name] = {"svm_auc": svm_auc, "svm_std": svm_std,
                         "ca_auc": ca_auc, "ca_std": ca_std}

    # Multi-task reference: GPCR-only (from paper)
    print(f"\n{'='*70}")
    print(f"  Reference (from paper):")
    print(f"{'='*70}")
    print(f"  Multi-task CA (no G protein info): AUC = 0.802 +/- 0.019")
    print(f"  Paired CA 650M + ICL (embedding):  AUC = 0.862 +/- 0.025")

    # Save
    out = {
        "description": "G protein one-hot vs embedding ablation",
        "multi_task_reference_auc": 0.802,
        "paired_embedding_reference_auc": 0.862,
        "results": {k: {"svm_auc": round(v["svm_auc"], 4),
                        "svm_std": round(v["svm_std"], 4),
                        "ca_auc": round(v["ca_auc"], 4),
                        "ca_std": round(v["ca_std"], 4)} for k, v in results.items()},
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
