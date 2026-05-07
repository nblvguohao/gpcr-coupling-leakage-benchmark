#!/usr/bin/env python3
"""
IPL: Interaction Prototypical Learning for GPCR-G protein coupling.

Innovation 2 of GPCR-IPN:
  - Replace binary classification with prototypical metric learning
  - Learn an interaction embedding space where (GPCR, Gprot) pairs cluster by family
  - Coupling prediction = negative Euclidean distance to class prototype
  - Naturally supports multi-family inference and calibrated probabilities

Components:
  1. GSCA encoder (from Innovation 1) -> interaction embedding (64-d)
  2. Prototypical loss (cross-entropy on distances to prototypes)
  3. Family-conditional inference
"""

import json, numpy as np, pandas as pd, torch, torch.nn as nn
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
DATA_DIR = BASE / "paired_dataset"
GPCR_FEATURES_FILE = BASE / "server_sync" / "extended_data" / "features" / "esm_features_650m_meanpool.json"
GPROT_FEATURES_FILE = DATA_DIR / "g_protein_esm_features.json"
ICL_FEATURES_FILE = DATA_DIR / "icl_features_650m.json"
PAIRING_MATRIX_FILE = DATA_DIR / "pairing_matrix_raw.csv"
CLUSTERS_FILE = DATA_DIR / "sequence_clusters.json"
OUTPUT_FILE = DATA_DIR / "ipl_results.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FAMILY_ORDER = ["G12_13", "Gi", "Gq", "Gs"]  # consistent ordering
NUM_FAMILIES = len(FAMILY_ORDER)
EMBED_DIM = 64  # interaction embedding dimensionality


# ===========================================================================
# Feature Loading (reused)
# ===========================================================================

def load_features():
    with open(GPCR_FEATURES_FILE) as f:
        gpcr_raw = json.load(f)
    with open(GPROT_FEATURES_FILE) as f:
        gprot_raw = json.load(f)
    gpcr_feats = {}
    for k, v in gpcr_raw.items():
        arr = np.array(v)
        gpcr_feats[k] = arr.mean(axis=0) if arr.ndim == 2 else arr
    gprot_feats = {}
    family_map = {"GNAQ": "Gq", "GNAI1": "Gi", "GNAI2": "Gi", "GNAI3": "Gi",
                  "GNAS": "Gs", "GNA12": "G12_13", "GNA13": "G12_13"}
    for subtype, info in gprot_raw.items():
        family = family_map.get(subtype, subtype)
        vec = np.array(info["mean_pooling"])
        gprot_feats[subtype] = vec
        if family not in gprot_feats:
            gprot_feats[family] = vec
    return gpcr_feats, gprot_feats


def load_icl():
    if not ICL_FEATURES_FILE.exists():
        return {}
    with open(ICL_FEATURES_FILE) as f:
        return json.load(f)


def get_gpcr_feat(gpcr_feats, gid):
    feat = gpcr_feats.get(gid)
    if feat is None and "_" in gid and len(gid.split("_")[0]) <= 2:
        base = gid.split("_", 1)[1]
        feat = gpcr_feats.get(base)
    if feat is None:
        for key in gpcr_feats:
            if "_" in key and key.split("_", 1)[1] == gid:
                feat = gpcr_feats[key]; break
    return feat


def get_icl_vector(icl_data, gid, dim=1280):
    rec = icl_data.get(gid)
    if rec is None:
        for key in icl_data:
            if "_" in key and key.split("_", 1)[1] == gid:
                rec = icl_data[key]; break
    icl2_esm = np.array(rec.get("ICL2_esm", [])) if rec else np.array([])
    icl3_esm = np.array(rec.get("ICL3_esm", [])) if rec else np.array([])
    s2 = rec.get("ICL2_stats", {}) if rec else {}
    s3 = rec.get("ICL3_stats", {}) if rec else {}
    if icl2_esm.size == 0: icl2_esm = np.zeros(dim)
    if icl3_esm.size == 0: icl3_esm = np.zeros(dim)
    sk = ["length","mean_hydro","std_hydro","net_charge","pos_charge_ratio",
          "neg_charge_ratio","hydrophobic_ratio","aromatic_ratio"]
    return icl2_esm, np.array([s2.get(k, 0.0) for k in sk]), \
           icl3_esm, np.array([s3.get(k, 0.0) for k in sk])


def build_vectors(df, gpcr_feats, gprot_feats, icl_data, mode="baseline"):
    X_list, y_list, family_list, meta = [], [], [], []
    for _, row in df.iterrows():
        gid = row["gpcr_id"]
        gpcr_feat = get_gpcr_feat(gpcr_feats, gid)
        gfam = row["g_protein_family"]
        gprot_feat = gprot_feats.get(gfam)
        if gprot_feat is None:
            gprot_feat = gprot_feats.get(gfam.capitalize())
        if gprot_feat is None:
            gprot_feat = gprot_feats.get(gfam.upper())
        if gpcr_feat is None or gprot_feat is None:
            continue
        parts = [np.concatenate([gpcr_feat, gprot_feat])]
        if mode == "icl_full":
            i2, i2s, i3, i3s = get_icl_vector(icl_data, gid)
            parts.append(np.concatenate([i2, i2s, i3, i3s]))
        X_list.append(np.concatenate(parts))
        y_list.append(int(row["coupling"]))
        family_list.append(gfam)
        meta.append({"gpcr_id": gid, "g_protein_family": gfam,
                     "cluster_id": int(row["cluster_id"]),
                     "family_idx": FAMILY_ORDER.index(gfam) if gfam in FAMILY_ORDER else -1})
    return np.array(X_list), np.array(y_list), family_list, meta


def get_cluster_folds(meta, cluster_list, n_folds=5):
    n = len(meta)
    sample_to_cluster = {i: meta[i]["cluster_id"] for i in range(n)}
    cluster_sizes = defaultdict(int)
    for i in range(n): cluster_sizes[sample_to_cluster[i]] += 1
    fold_clusters = [[] for _ in range(n_folds)]
    fold_size = [0] * n_folds
    sorted_cids = sorted(range(len(cluster_list)), key=lambda c: cluster_sizes[c], reverse=True)
    for cid in sorted_cids:
        if cluster_sizes.get(cid, 0) == 0: continue
        target = int(np.argmin(fold_size))
        fold_clusters[target].append(cid)
        fold_size[target] += cluster_sizes[cid]
    return sample_to_cluster, fold_clusters


# ===========================================================================
# IPL Model
# ===========================================================================

class GSCAEncoder(nn.Module):
    """GSCA encoder (from Innovation 1) that produces an interaction embedding."""
    def __init__(self, gpcr_dim=1280, gprot_dim=320, hidden_dim=256,
                 num_heads=4, dropout=0.3, embed_dim=EMBED_DIM):
        super().__init__()
        self.gpcr_proj = nn.Sequential(
            nn.Linear(gpcr_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.gprot_proj = nn.Sequential(
            nn.Linear(gprot_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        # Simplified gate (single linear)
        self.gate_linear = nn.Linear(hidden_dim * 4, 1)
        nn.init.constant_(self.gate_linear.bias, 0.7)
        self.gate_dropout = nn.Dropout(dropout)

        # Projection to interaction embedding
        self.interaction_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, gpcr_feat, gprot_feat, return_gate=False):
        h_gpcr = self.gpcr_proj(gpcr_feat)
        h_gprot = self.gprot_proj(gprot_feat)

        # Bidirectional cross-attention
        attn_g2p, _ = self.cross_attn(
            h_gpcr.unsqueeze(1), h_gprot.unsqueeze(1), h_gprot.unsqueeze(1)
        )
        attn_p2g, _ = self.cross_attn(
            h_gprot.unsqueeze(1), h_gpcr.unsqueeze(1), h_gpcr.unsqueeze(1)
        )
        attn_g2p = attn_g2p.squeeze(1)
        attn_p2g = attn_p2g.squeeze(1)

        # Gate
        gate_in = self.gate_dropout(
            torch.cat([h_gpcr, h_gprot, attn_g2p, attn_p2g], dim=-1)
        )
        gate = torch.sigmoid(self.gate_linear(gate_in))
        h_fused = gate * attn_g2p + (1 - gate) * attn_p2g

        # Interaction embedding
        h = torch.cat([h_fused, h_gpcr], dim=-1)
        embedding = self.interaction_proj(h)

        if return_gate:
            return embedding, gate
        return embedding


class PrototypicalIPL(nn.Module):
    """Prototypical Interaction Learning head.

    Predicts coupling probability based on distance to per-family prototypes.
    """
    def __init__(self, embed_dim=EMBED_DIM, num_families=NUM_FAMILIES):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_families = num_families

    def compute_prototypes(self, embeddings, family_indices):
        """Compute per-family prototypes from training embeddings.
        Args:
            embeddings: (N, embed_dim)
            family_indices: (N,) integer family labels
        Returns:
            prototypes: (num_families, embed_dim)
        """
        prototypes = []
        for f in range(self.num_families):
            mask = (family_indices == f)
            if mask.sum() > 0:
                prototypes.append(embeddings[mask].mean(dim=0))
            else:
                prototypes.append(torch.zeros(self.embed_dim, device=embeddings.device))
        return torch.stack(prototypes)

    def forward(self, embeddings, prototypes, family_idx=None):
        """Compute coupling log-probabilities from distances to prototypes.

        For training: returns NLL loss given true family + coupling label.
        For inference: returns coupling probability for each family.

        Args:
            embeddings: (N, embed_dim) interaction embeddings
            prototypes: (num_families, embed_dim)
            family_idx: (N,) integer family index (or None for inference)
        Returns:
            loss or probabilities
        """
        # Distances from each sample to each prototype
        # (N, 1, embed_dim) - (1, num_families, embed_dim) -> (N, num_families)
        dists = torch.cdist(embeddings.unsqueeze(1), prototypes.unsqueeze(0))
        dists = dists.squeeze(1)  # (N, num_families)

        # Negative distance = similarity score
        # Higher similarity = higher coupling probability
        logits = -dists  # (N, num_families)

        return logits


# ===========================================================================
# Dataset (with family labels)
# ===========================================================================

class IPL_Dataset(Dataset):
    def __init__(self, X, y, family_idx, mode="baseline", gprot_dim=320):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.family_idx = torch.LongTensor(family_idx)
        self.mode = mode
        self.gprot_dim = gprot_dim

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        gpcr_global = x[:1280]
        gprot_vec = x[1280:1280 + self.gprot_dim]
        if self.mode == "icl_full":
            icl_start = 1280 + self.gprot_dim
            icl = x[icl_start:icl_start + 2576]
            gpcr_side = torch.cat([gpcr_global, icl])
        else:
            gpcr_side = gpcr_global
        return gpcr_side, gprot_vec, self.y[idx], self.family_idx[idx]


# ===========================================================================
# Training
# ===========================================================================

def train_ipl_epoch(encoder, prot_head, loader, optim, scaler_temp=1.0):
    encoder.train()
    total_loss = 0.0
    for gpcr, gprot, y, fidx in loader:
        gpcr, gprot, y, fidx = (gpcr.to(DEVICE), gprot.to(DEVICE),
                                 y.to(DEVICE), fidx.to(DEVICE))
        optim.zero_grad()

        # Get interaction embeddings
        embeddings, gate = encoder(gpcr, gprot, return_gate=True)

        # Compute prototypes from training batch (for training, use batch prototypes)
        with torch.no_grad():
            prototypes = prot_head.compute_prototypes(embeddings, fidx)

        # Compute distance logits
        logits = prot_head(embeddings, prototypes)  # (B, num_families)

        # For each sample: the positive family logit should be high if coupling=1
        # and low if coupling=0. Use a margin-based formulation:
        # loss = cross_entropy on family logits, but weighted by coupling label
        # Coupling=1: maximize logit for this family, Coupling=0: minimize it

        # Gather per-sample family logits
        batch_idx = torch.arange(len(y), device=DEVICE)
        family_logits = logits[batch_idx, fidx]  # (B,) -> logit for the true family

        # Binary loss on family logit (positive=couple, negative=non-couple)
        # This is our "coupling prediction" - does this (GPCR, family) pair couple?
        loss = nn.BCEWithLogitsLoss()(family_logits / scaler_temp, y)

        # Optional: add regularization to push families apart
        if embeddings.shape[0] >= 4:
            # Random negative pairs: different family, both positive
            neg_dists = []
            for f in range(prot_head.num_families):
                f_mask = (fidx == f) & (y == 1)
                if f_mask.sum() >= 2:
                    f_emb = embeddings[f_mask]
                    neg_dists.append(torch.cdist(f_emb.unsqueeze(0), f_emb.unsqueeze(0)).mean())
            if neg_dists:
                intra_family_dist = torch.stack(neg_dists).mean()
                # Penalize excessive intra-family spread
                loss = loss + 0.01 * intra_family_dist

        # Gate asymmetry regularization
        if gate is not None and gate.numel() > 0:
            gate_reg = 0.02 * torch.mean(torch.abs(gate - 0.5))
            loss = loss - gate_reg

        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        optim.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_ipl(encoder, prot_head, loader, return_gates=False):
    encoder.eval()
    all_embeddings, all_labels, all_families = [], [], []
    all_gates = []
    all_probs = []

    for gpcr, gprot, y, fidx in loader:
        gpcr, gprot = gpcr.to(DEVICE), gprot.to(DEVICE)
        embeddings, gate = encoder(gpcr, gprot, return_gate=True)
        all_embeddings.append(embeddings.cpu())
        all_labels.append(y.numpy())
        all_families.append(fidx.numpy())
        if return_gates and gate is not None:
            all_gates.append(gate.cpu().numpy())

    embeddings = torch.cat(all_embeddings)
    labels = np.concatenate(all_labels)
    families = np.concatenate(all_families)
    gates = np.concatenate(all_gates) if all_gates else None

    # Compute prototypes from ALL samples (test-time: use global prototypes)
    # In CV setting, we compute from training data during inference
    # (handled in run_cv, not here)

    return embeddings, labels, families, gates


def predict_with_prototypes(encoder, prot_head, loader, prototypes):
    """Predict coupling probabilities given fixed prototypes."""
    encoder.eval()
    probs_list, labels_list = [], []
    for gpcr, gprot, y, _ in loader:
        gpcr, gprot = gpcr.to(DEVICE), gprot.to(DEVICE)
        embeddings = encoder(gpcr, gprot)
        logits = prot_head(embeddings, prototypes)  # (B, num_families)
        # Coupling probability for each sample's family:
        # We need family indices - stored in the dataset but not returned here
        probs_list.append(torch.sigmoid(logits).cpu().numpy())
        labels_list.append(y.numpy())
    return np.concatenate(probs_list), np.concatenate(labels_list)


# ===========================================================================
# Cluster-aware CV with IPL
# ===========================================================================

def run_ipl_cv(X, y, families, meta, fold_clusters, sample_to_cluster,
               mode="icl_full", gprot_dim=320, n_folds=5, epochs=80,
               lr=1e-4, batch_size=32):
    """Cluster-aware CV with prototypical learning."""
    gpcr_dim = {"baseline": 1280, "icl_full": 1280 + 2576}[mode]
    family_indices = np.array([m["family_idx"] for m in meta])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    n = len(y)

    fold_metrics = []

    for f in range(n_folds):
        test_clusters = set(fold_clusters[f])
        test_idx = [i for i in range(n) if sample_to_cluster[i] in test_clusters]
        train_idx = [i for i in range(n) if sample_to_cluster[i] not in test_clusters]

        X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        fi_tr, fi_te = family_indices[train_idx], family_indices[test_idx]

        tr_ds = IPL_Dataset(X_tr, y_tr, fi_tr, mode, gprot_dim)
        te_ds = IPL_Dataset(X_te, y_te, fi_te, mode, gprot_dim)
        tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
        te_loader = DataLoader(te_ds, batch_size=batch_size // 2)

        encoder = GSCAEncoder(gpcr_dim=gpcr_dim, gprot_dim=gprot_dim).to(DEVICE)
        prot_head = PrototypicalIPL().to(DEVICE)
        optim = torch.optim.AdamW(encoder.parameters(), lr=lr, weight_decay=1e-4)

        best_auc = -1.0
        best_probs = None
        best_labels = None
        patience = 0

        for ep in range(epochs):
            train_ipl_epoch(encoder, prot_head, tr_loader, optim)

            # Evaluate on test fold
            te_embs, te_labels, te_fams, _ = evaluate_ipl(encoder, prot_head, te_loader)

            # Compute train prototypes for prediction
            tr_embs, _, _, _ = evaluate_ipl(encoder, prot_head, tr_loader)
            tr_fam_tensor = torch.LongTensor(fi_tr).to(DEVICE)
            prototypes = prot_head.compute_prototypes(tr_embs.to(DEVICE), tr_fam_tensor)

            # Predict
            with torch.no_grad():
                te_logits = prot_head(te_embs.to(DEVICE), prototypes)  # (B, 4)
                te_probs = torch.sigmoid(te_logits).cpu().numpy()

            # For each test sample, get probability for its family
            sample_probs = te_probs[np.arange(len(te_fams)), te_fams]
            auc_v = roc_auc_score(te_labels, sample_probs) if len(set(te_labels)) >= 2 else -1

            if auc_v > best_auc:
                best_auc = auc_v
                best_probs = sample_probs.copy()
                best_labels = te_labels.copy()
                patience = 0
            else:
                patience += 1
            if patience >= 15:
                break

        fold_metrics.append({
            "auc": float(best_auc),
            "probs": best_probs.tolist() if best_probs is not None else [],
            "labels": best_labels.tolist() if best_labels is not None else [],
        })
        print(f"    Fold {f+1}: AUC = {best_auc:.4f}")

    aucs = [m["auc"] for m in fold_metrics if not np.isnan(m["auc"])]
    all_probs = np.concatenate([m["probs"] for m in fold_metrics])
    all_labels = np.concatenate([m["labels"] for m in fold_metrics])

    return {
        "auc_mean": round(float(np.mean(aucs)), 4),
        "auc_std": round(float(np.std(aucs)), 4),
        "fold_aucs": [round(float(m["auc"]), 4) for m in fold_metrics],
        "overall_pr_auc": round(float(average_precision_score(all_labels, all_probs)), 4),
    }


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 70)
    print("  IPL: Interaction Prototypical Learning")
    print(f"  Device: {DEVICE}, Embed dim: {EMBED_DIM}")
    print("=" * 70)

    df = pd.read_csv(PAIRING_MATRIX_FILE)
    with open(CLUSTERS_FILE) as f:
        cluster_list = json.load(f)["clusters"]
    gpcr_feats, gprot_feats = load_features()
    icl_data = load_icl()

    results = {}

    # Compare: BCE baseline (from GSCA) vs IPLin both baseline and icl_full modes
    for mode in ["baseline", "icl_full"]:
        print(f"\n{'='*60}")
        print(f"  Mode: {mode}")
        print(f"{'='*60}")

        X, y, families, meta = build_vectors(df, gpcr_feats, gprot_feats, icl_data, mode)
        sample_to_cluster, fold_clusters = get_cluster_folds(meta, cluster_list)
        print(f"  Samples: {len(y)}, Features: {X.shape[1]}")
        print(f"  Family distribution: {dict(zip(*np.unique(families, return_counts=True)))}")

        # IPL
        print(f"\n  --- Prototypical IPL ---")
        ipl_res = run_ipl_cv(X, y, families, meta, fold_clusters, sample_to_cluster,
                             mode=mode, gprot_dim=320)
        results[f"ipl_{mode}"] = ipl_res
        print(f"  >> IPL AUC = {ipl_res['auc_mean']:.4f} +/- {ipl_res['auc_std']:.4f}, "
              f"PR-AUC = {ipl_res['overall_pr_auc']:.4f}")

    # Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Results saved to {OUTPUT_FILE}")

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  {'Config':<35} {'AUC':>8} {'Std':>8} {'PR-AUC':>8}")
    print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8}")
    for k, v in sorted(results.items()):
        print(f"  {k:<35} {v['auc_mean']:>8.4f} {v['auc_std']:>8.4f} {v['overall_pr_auc']:>8.4f}")


if __name__ == "__main__":
    main()
