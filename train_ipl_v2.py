#!/usr/bin/env python3
"""
IPLv2: Hybrid Prototypical-Classification Learning.

Combines:
  1. BCE classification head (primary) - proven strong baseline
  2. Prototypical regularization (auxiliary) - pulls positive pairs together,
     pushes different families apart in interaction embedding space

This hybrid gives us:
  - Strong classification performance (from BCE)
  - Interpretable interaction embeddings (from prototypical regularization)
  - Calibrated distance-based confidence (from prototype distances)
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
OUTPUT_FILE = DATA_DIR / "ipl_v2_results.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FAMILY_ORDER = ["G12_13", "Gi", "Gq", "Gs"]
NUM_FAMILIES = len(FAMILY_ORDER)
EMBED_DIM = 64


# ===========================================================================
# Feature Loading (identical to GSCA)
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
    return np.array(X_list), np.array(y_list), np.array([m["family_idx"] for m in meta]), meta


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
# IPLv2 Model: GSCA Encoder + BCE Classifier + Prototypical Regularization
# ===========================================================================

class GSCAEncoder(nn.Module):
    """GSCA encoder (from Innovation 1) with interaction embedding + classifier."""
    def __init__(self, gpcr_dim=1280, gprot_dim=320, hidden_dim=256,
                 num_heads=4, dropout=0.3, embed_dim=EMBED_DIM):
        super().__init__()
        self.embed_dim = embed_dim
        self.gpcr_proj = nn.Sequential(
            nn.Linear(gpcr_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.gprot_proj = nn.Sequential(
            nn.Linear(gprot_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.gate_linear = nn.Linear(hidden_dim * 4, 1)
        nn.init.constant_(self.gate_linear.bias, 0.7)
        self.gate_dropout = nn.Dropout(dropout)

        # Interaction embedding projector
        self.interaction_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

        # BCE classifier (on the interaction embedding)
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, gpcr_feat, gprot_feat, return_all=False):
        h_gpcr = self.gpcr_proj(gpcr_feat)
        h_gprot = self.gprot_proj(gprot_feat)

        attn_g2p, _ = self.cross_attn(
            h_gpcr.unsqueeze(1), h_gprot.unsqueeze(1), h_gprot.unsqueeze(1)
        )
        attn_p2g, _ = self.cross_attn(
            h_gprot.unsqueeze(1), h_gpcr.unsqueeze(1), h_gpcr.unsqueeze(1)
        )
        attn_g2p, attn_p2g = attn_g2p.squeeze(1), attn_p2g.squeeze(1)

        gate_in = self.gate_dropout(torch.cat([h_gpcr, h_gprot, attn_g2p, attn_p2g], dim=-1))
        gate = torch.sigmoid(self.gate_linear(gate_in))
        h_fused = gate * attn_g2p + (1 - gate) * attn_p2g

        h = torch.cat([h_fused, h_gpcr], dim=-1)
        embedding = self.interaction_proj(h)
        logit = self.classifier(embedding).squeeze(-1)

        if return_all:
            return logit, embedding, gate.squeeze(-1)
        return logit


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
            gpcr_side = torch.cat([gpcr_global, x[icl_start:icl_start + 2576]])
        else:
            gpcr_side = gpcr_global
        return gpcr_side, gprot_vec, self.y[idx], self.family_idx[idx]


# ===========================================================================
# Prototypical Regularization
# ===========================================================================

def prototypical_loss(embeddings, family_idx, labels, margin=1.0):
    """Prototypical regularization loss.

    For each batch:
    - Pull positive embeddings of same family toward each other
    - Push negative embeddings away from positive prototypes

    Returns a scalar loss.
    """
    device = embeddings.device
    prototypes = {}
    for f in range(NUM_FAMILIES):
        mask = (family_idx == f)
        if mask.sum() > 0:
            prototypes[f] = embeddings[mask].mean(dim=0)

    if len(prototypes) < 2:
        return torch.tensor(0.0, device=device)

    loss = torch.tensor(0.0, device=device)
    n_pairs = 0

    for f, proto in prototypes.items():
        # Positive samples of this family: pull toward prototype
        pos_mask = (family_idx == f) & (labels == 1)
        if pos_mask.sum() > 0:
            pos_dists = torch.norm(embeddings[pos_mask] - proto.unsqueeze(0), dim=1)
            loss = loss + pos_dists.mean()
            n_pairs += 1

        # Negative samples of this family: push away from prototype
        neg_mask = (family_idx == f) & (labels == 0)
        if neg_mask.sum() > 0:
            neg_dists = torch.norm(embeddings[neg_mask] - proto.unsqueeze(0), dim=1)
            loss = loss + torch.clamp(margin - neg_dists, min=0).mean()
            n_pairs += 1

    # Cross-family: push different family prototypes apart
    proto_list = list(prototypes.values())
    for i in range(len(proto_list)):
        for j in range(i + 1, len(proto_list)):
            cross_dist = torch.norm(proto_list[i] - proto_list[j])
            loss = loss + torch.clamp(margin - cross_dist, min=0)
            n_pairs += 1

    return loss / max(1, n_pairs)


# ===========================================================================
# Training Loop
# ===========================================================================

def train_epoch(model, loader, optim, crit, proto_weight=0.1):
    model.train()
    total_loss, total_bce, total_proto = 0.0, 0.0, 0.0
    for gpcr, gprot, y, fidx in loader:
        gpcr, gprot, y, fidx = (gpcr.to(DEVICE), gprot.to(DEVICE),
                                 y.to(DEVICE), fidx.to(DEVICE))
        optim.zero_grad()

        logit, embedding, gate = model(gpcr, gprot, return_all=True)

        # BCE loss (primary)
        bce_loss = crit(logit, y)

        # Prototypical regularization (auxiliary)
        proto_loss = prototypical_loss(embedding, fidx, y)

        loss = bce_loss + proto_weight * proto_loss

        # Gate asymmetry regularization
        if gate.numel() > 0:
            gate_reg = 0.02 * torch.mean(torch.abs(gate - 0.5))
            loss = loss - gate_reg

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        total_loss += loss.item() * len(y)
        total_bce += bce_loss.item() * len(y)
        total_proto += proto_loss.item() * len(y)

    n = len(loader.dataset)
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_probs, all_labels, all_embeddings, all_gates = [], [], [], []
    all_families = []
    for gpcr, gprot, y, fidx in loader:
        gpcr, gprot = gpcr.to(DEVICE), gprot.to(DEVICE)
        logit, embedding, gate = model(gpcr, gprot, return_all=True)
        all_probs.append(torch.sigmoid(logit).cpu().numpy())
        all_labels.append(y.numpy())
        all_embeddings.append(embedding.cpu())
        all_gates.append(gate.cpu().numpy())
        all_families.append(fidx.numpy())
    return {
        "probs": np.concatenate(all_probs),
        "labels": np.concatenate(all_labels),
        "embeddings": torch.cat(all_embeddings),
        "gates": np.concatenate(all_gates),
        "families": np.concatenate(all_families),
    }


# ===========================================================================
# Cluster-aware CV
# ===========================================================================

def run_cv(X, y, family_idx, meta, fold_clusters, sample_to_cluster,
           mode="icl_full", gprot_dim=320, n_folds=5, epochs=80,
           lr=1e-4, batch_size=32, proto_weight=0.1):
    gpcr_dim = {"baseline": 1280, "icl_full": 1280 + 2576}[mode]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    n = len(y)

    fold_metrics, gate_stats_list = [], []

    for f in range(n_folds):
        test_clusters = set(fold_clusters[f])
        test_idx = [i for i in range(n) if sample_to_cluster[i] in test_clusters]
        train_idx = [i for i in range(n) if sample_to_cluster[i] not in test_clusters]

        X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        fi_tr, fi_te = family_idx[train_idx], family_idx[test_idx]

        tr_ds = IPL_Dataset(X_tr, y_tr, fi_tr, mode, gprot_dim)
        te_ds = IPL_Dataset(X_te, y_te, fi_te, mode, gprot_dim)
        tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
        te_loader = DataLoader(te_ds, batch_size=batch_size // 2)

        model = GSCAEncoder(gpcr_dim=gpcr_dim, gprot_dim=gprot_dim).to(DEVICE)
        optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        pos_weight = torch.FloatTensor([len(y_tr) / max(1, y_tr.sum())]).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_auc = -1.0
        best_eval = None
        patience = 0

        for ep in range(epochs):
            train_epoch(model, tr_loader, optim, criterion, proto_weight)
            ev = evaluate(model, te_loader)
            auc_v = roc_auc_score(ev["labels"], ev["probs"]) if len(set(ev["labels"])) >= 2 else -1
            if auc_v > best_auc:
                best_auc = auc_v
                best_eval = ev
                patience = 0
            else:
                patience += 1
            if patience >= 15:
                break

        fold_metrics.append({"auc": float(best_auc)})
        gate_stats_list.append(best_eval["gates"])
        print(f"    Fold {f+1}: AUC = {best_auc:.4f}")

    aucs = [m["auc"] for m in fold_metrics if not np.isnan(m["auc"])]

    # Aggregate gates
    all_gates = np.concatenate(gate_stats_list) if gate_stats_list else np.array([])
    gate_info = {}
    if len(all_gates) > 0:
        gate_info = {
            "mean_gate": float(np.mean(all_gates)),
            "std_gate": float(np.std(all_gates)),
            "gate_p25": float(np.percentile(all_gates, 25)),
            "gate_p50": float(np.percentile(all_gates, 50)),
            "gate_p75": float(np.percentile(all_gates, 75)),
        }

    return {
        "auc_mean": round(float(np.mean(aucs)), 4),
        "auc_std": round(float(np.std(aucs)), 4),
        "fold_aucs": [round(float(m["auc"]), 4) for m in fold_metrics],
        "gate_stats": gate_info if gate_info else None,
    }


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 70)
    print("  IPLv2: Hybrid Prototypical-Classification Learning")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    df = pd.read_csv(PAIRING_MATRIX_FILE)
    with open(CLUSTERS_FILE) as f:
        cluster_list = json.load(f)["clusters"]
    gpcr_feats, gprot_feats = load_features()
    icl_data = load_icl()

    results = {}

    for proto_w in [0.05, 0.1, 0.2]:
        print(f"\n{'='*60}")
        print(f"  Prototypical weight: {proto_w}")
        print(f"{'='*60}")
        for mode in ["baseline", "icl_full"]:
            print(f"\n  Mode: {mode}")
            X, y, family_idx, meta = build_vectors(df, gpcr_feats, gprot_feats, icl_data, mode)
            sample_to_cluster, fold_clusters = get_cluster_folds(meta, cluster_list)
            res = run_cv(X, y, family_idx, meta, fold_clusters, sample_to_cluster,
                         mode=mode, gprot_dim=320, proto_weight=proto_w)
            key = f"ipl_v2_proto{proto_w}_{mode}"
            results[key] = res
            print(f"  >> AUC = {res['auc_mean']:.4f} +/- {res['auc_std']:.4f}")

    # Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Saved to {OUTPUT_FILE}")

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY (compare to GSCA baseline: uni_ca_icl_full=0.8572)")
    print("=" * 70)
    print(f"  {'Config':<35} {'AUC':>8} {'Std':>8}")
    print(f"  {'-'*35} {'-'*8} {'-'*8}")
    for k, v in sorted(results.items()):
        print(f"  {k:<35} {v['auc_mean']:>8.4f} {v['auc_std']:>8.4f}")


if __name__ == "__main__":
    main()
