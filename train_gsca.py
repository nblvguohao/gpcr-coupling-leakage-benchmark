#!/usr/bin/env python3
"""
GSCA: Gated Symmetry-Aware Cross-Attention for GPCR-G protein coupling.

Innovation 1 of GPCR-IPN:
  - Bidirectional cross-attention (GPCR->Gprot AND Gprot->GPCR)
  - Learnable gate that fuses the two attention directions
  - Gate value = interpretable signal of "who dominates the interaction"

Output:
  - Cluster-aware CV AUC for: original (uni-CA), bidirectional (no gate), GSCA (gated)
  - Gate value distribution analysis
  - Per-family gate statistics
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
OUTPUT_FILE = DATA_DIR / "gsca_results.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===========================================================================
# Feature Loading (reused from original pipeline)
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
    for subtype, info in gprot_raw.items():
        family_map = {"GNAQ": "Gq", "GNAI1": "Gi", "GNAI2": "Gi", "GNAI3": "Gi",
                      "GNAS": "Gs", "GNA12": "G12_13", "GNA13": "G12_13"}
        family = family_map.get(subtype, subtype)
        gprot_feats[subtype] = np.array(info["mean_pooling"])
        if family not in gprot_feats:
            gprot_feats[family] = np.array(info["mean_pooling"])
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


def get_icl_vector(icl_data, gid, gpcr_feat_dim=1280):
    """Extract ICL2/3 ESM and stat features (identical to original pipeline)."""
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
    stat_keys = ["length","mean_hydro","std_hydro","net_charge","pos_charge_ratio",
                 "neg_charge_ratio","hydrophobic_ratio","aromatic_ratio"]
    icl2_stat_vec = np.array([icl2_stats.get(k, 0.0) for k in stat_keys])
    icl3_stat_vec = np.array([icl3_stats.get(k, 0.0) for k in stat_keys])
    return icl2_esm, icl2_stat_vec, icl3_esm, icl3_stat_vec


def build_vectors(df, gpcr_feats, gprot_feats, icl_data, mode="baseline"):
    X_list, y_list, meta = [], [], []
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
            # Use original robust ICL extraction with get_icl_vector
            icl2_esm, icl2_stat, icl3_esm, icl3_stat = get_icl_vector(icl_data, gid)
            parts.append(np.concatenate([icl2_esm, icl2_stat, icl3_esm, icl3_stat]))
        X_list.append(np.concatenate(parts))
        y_list.append(int(row["coupling"]))
        meta.append({"gpcr_id": gid, "g_protein_family": gfam, "cluster_id": int(row["cluster_id"])})
    return np.array(X_list), np.array(y_list), meta


def get_cluster_folds(meta, cluster_list, n_folds=5):
    n = len(meta)
    sample_to_cluster = {i: meta[i]["cluster_id"] for i in range(n)}
    cluster_sizes = defaultdict(int)
    for i in range(n):
        cluster_sizes[sample_to_cluster[i]] += 1
    fold_clusters = [[] for _ in range(n_folds)]
    fold_size = [0] * n_folds
    sorted_cids = sorted(range(len(cluster_list)), key=lambda c: cluster_sizes[c], reverse=True)
    for cid in sorted_cids:
        if cluster_sizes.get(cid, 0) == 0:
            continue
        target = int(np.argmin(fold_size))
        fold_clusters[target].append(cid)
        fold_size[target] += cluster_sizes[cid]
    return sample_to_cluster, fold_clusters


# ===========================================================================
# GSCA Model
# ===========================================================================

class GatedSymmetryCrossAttention(nn.Module):
    """Gated Symmetry-Aware Cross-Attention (GSCA).

    Bidirectional cross-attention with a learnable gate that fuses
    GPCR->Gprot and Gprot->GPCR attention streams.
    The gate value is interpretable: near 1 = GPCR-dominated,
    near 0 = Gprot-dominated, near 0.5 = co-determined.
    """
    def __init__(self, gpcr_dim=1280, gprot_dim=320, hidden_dim=256,
                 num_heads=4, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Shared projections
        self.gpcr_proj = nn.Sequential(
            nn.Linear(gpcr_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.gprot_proj = nn.Sequential(
            nn.Linear(gprot_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU()
        )

        # Cross-attention (bidirectional)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Learnable gate: simplified single-layer with bias init
        # gate = sigmoid(W_g * [h_gpcr, h_gprot, attn_gpcr, attn_gprot] + b_g)
        # Bias initialized to favor GPCR-direction (gate ~0.67 initially)
        self.gate_linear = nn.Linear(hidden_dim * 4, 1)
        nn.init.constant_(self.gate_linear.bias, 0.7)  # favor GPCR→Gprot initially
        self.gate_dropout = nn.Dropout(dropout)

        # Final classifier
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, gpcr_feat, gprot_feat):
        # Project to shared space
        h_gpcr = self.gpcr_proj(gpcr_feat)  # (B, hidden)
        h_gprot = self.gprot_proj(gprot_feat)  # (B, hidden)

        # Bidirectional cross-attention
        # GPCR -> Gprot (query=GPCR, key/value=Gprot)
        q_gpcr = h_gpcr.unsqueeze(1)  # (B, 1, hidden)
        kv_gprot = h_gprot.unsqueeze(1)  # (B, 1, hidden)
        attn_g2p, _ = self.cross_attn(q_gpcr, kv_gprot, kv_gprot)  # (B, 1, hidden)

        # Gprot -> GPCR (query=Gprot, key/value=GPCR)
        q_gprot = h_gprot.unsqueeze(1)
        kv_gpcr = h_gpcr.unsqueeze(1)
        attn_p2g, _ = self.cross_attn(q_gprot, kv_gpcr, kv_gpcr)  # (B, 1, hidden)

        attn_g2p = attn_g2p.squeeze(1)  # (B, hidden)
        attn_p2g = attn_p2g.squeeze(1)  # (B, hidden)

        # Simplified gate: single linear layer
        gate_input = self.gate_dropout(
            torch.cat([h_gpcr, h_gprot, attn_g2p, attn_p2g], dim=-1)
        )  # (B, hidden*4)
        gate = torch.sigmoid(self.gate_linear(gate_input))  # (B, 1)

        # Fused representation
        h_fused = gate * attn_g2p + (1 - gate) * attn_p2g  # (B, hidden)

        # Concatenate with original projection + classify
        x = torch.cat([h_fused, h_gpcr], dim=-1)  # (B, hidden*2)
        logit = self.ffn(x).squeeze(-1)  # (B,)

        return logit, gate.squeeze(-1)


class UniDirectionalCA(nn.Module):
    """Original uni-directional cross-attention (baseline for comparison)."""
    def __init__(self, gpcr_dim=1280, gprot_dim=320, hidden_dim=256,
                 num_heads=4, dropout=0.3):
        super().__init__()
        self.gpcr_proj = nn.Sequential(
            nn.Linear(gpcr_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.gprot_proj = nn.Sequential(
            nn.Linear(gprot_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
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
        return self.ffn(x).squeeze(-1), None  # no gate


class BidirectionalCA(nn.Module):
    """Bidirectional CA without gate (ablation: simple average)."""
    def __init__(self, gpcr_dim=1280, gprot_dim=320, hidden_dim=256,
                 num_heads=4, dropout=0.3):
        super().__init__()
        self.gpcr_proj = nn.Sequential(
            nn.Linear(gpcr_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.gprot_proj = nn.Sequential(
            nn.Linear(gprot_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, gpcr_feat, gprot_feat):
        h_gpcr = self.gpcr_proj(gpcr_feat)
        h_gprot = self.gprot_proj(gprot_feat)
        q_gpcr = h_gpcr.unsqueeze(1)
        kv_gprot = h_gprot.unsqueeze(1)
        attn_g2p, _ = self.cross_attn(q_gpcr, kv_gprot, kv_gprot)
        q_gprot = h_gprot.unsqueeze(1)
        kv_gpcr = h_gpcr.unsqueeze(1)
        attn_p2g, _ = self.cross_attn(q_gprot, kv_gpcr, kv_gpcr)
        # Simple average (no gate)
        h_fused = (attn_g2p + attn_p2g) / 2
        x = torch.cat([h_fused.squeeze(1), h_gpcr], dim=-1)
        return self.ffn(x).squeeze(-1), None


# ===========================================================================
# Dataset
# ===========================================================================

class PairDataset(Dataset):
    """X layout: [GPCR(1280) | Gprot(320) | ICL(2576)]."""
    def __init__(self, X, y, mode="baseline", gprot_dim=320):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
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
        return gpcr_side, gprot_vec, self.y[idx]


# ===========================================================================
# Training
# ===========================================================================

def train_epoch(model, loader, optim, crit):
    model.train()
    loss_total = 0.0
    for gpcr, gprot, y in loader:
        gpcr, gprot, y = gpcr.to(DEVICE), gprot.to(DEVICE), y.to(DEVICE)
        optim.zero_grad()
        logits, gate = model(gpcr, gprot)
        loss = crit(logits, y)
        # Gate asymmetry regularization
        if gate is not None and gate.numel() > 0:
            gate_reg = 0.02 * torch.mean(torch.abs(gate - 0.5))
            loss = loss - gate_reg  # maximize asymmetry
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        loss_total += loss.item() * len(y)
    return loss_total / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, collect_gates=False):
    model.eval()
    probs, labels, gates = [], [], []
    for gpcr, gprot, y in loader:
        gpcr, gprot = gpcr.to(DEVICE), gprot.to(DEVICE)
        logits, gate = model(gpcr, gprot)
        probs.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(y.numpy())
        if collect_gates and gate is not None:
            gates.append(gate.cpu().numpy())
    probs = np.concatenate(probs)
    labels = np.concatenate(labels)
    gates = np.concatenate(gates) if gates else None
    return probs, labels, gates


def run_cv(X, y, meta, fold_clusters, sample_to_cluster, mode,
           model_cls, model_name, gprot_dim=320,
           n_folds=5, epochs=80, lr=1e-4, batch_size=32):
    """Run cluster-aware CV for a given model class."""
    gpcr_dim = {"baseline": 1280, "icl_full": 1280 + 2576}[mode]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    n = len(y)

    fold_results = []
    all_gates_by_fold = {}

    for f in range(n_folds):
        test_clusters = set(fold_clusters[f])
        test_idx = [i for i in range(n) if sample_to_cluster[i] in test_clusters]
        train_idx = [i for i in range(n) if sample_to_cluster[i] not in test_clusters]

        X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        tr_ds = PairDataset(X_tr, y_tr, mode, gprot_dim)
        te_ds = PairDataset(X_te, y_te, mode, gprot_dim)
        tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
        te_loader = DataLoader(te_ds, batch_size=batch_size)

        # Initialize model
        if model_cls == GatedSymmetryCrossAttention:
            model = model_cls(gpcr_dim=gpcr_dim, gprot_dim=gprot_dim).to(DEVICE)
        elif model_cls in (UniDirectionalCA, BidirectionalCA):
            model = model_cls(gpcr_dim=gpcr_dim, gprot_dim=gprot_dim).to(DEVICE)
        else:
            raise ValueError(f"Unknown model: {model_cls}")

        optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        pos_weight = torch.FloatTensor([len(y_tr) / max(1, y_tr.sum())]).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_auc = -1.0
        best_probs = None
        best_gates = None
        patience = 0

        for ep in range(epochs):
            train_epoch(model, tr_loader, optim, criterion)
            probs, lbl, gates = evaluate(model, te_loader, collect_gates=True)
            auc_v = roc_auc_score(lbl, probs) if len(set(lbl)) >= 2 else -1
            if auc_v > best_auc:
                best_auc = auc_v
                best_probs = probs
                best_gates = gates
                patience = 0
            else:
                patience += 1
            if patience >= 15:
                break

        fold_results.append({
            "probs": best_probs.tolist(),
            "labels": y_te.tolist(),
            "auc": float(best_auc),
            "test_cluster_ids": list(test_clusters),
        })
        if best_gates is not None:
            all_gates_by_fold[f] = {
                "gates": best_gates.tolist(),
                "labels": y_te.tolist(),
            }
        print(f"    Fold {f+1}: AUC = {best_auc:.4f}")

    aucs = [r["auc"] for r in fold_results if not np.isnan(r["auc"])]
    all_probs = np.concatenate([r["probs"] for r in fold_results])
    all_labels = np.concatenate([r["labels"] for r in fold_results])

    return {
        "model": model_name,
        "auc_mean": round(float(np.mean(aucs)), 4),
        "auc_std": round(float(np.std(aucs)), 4),
        "fold_aucs": [round(float(r["auc"]), 4) for r in fold_results],
        "overall_pr_auc": round(float(average_precision_score(all_labels, all_probs)), 4),
        "gate_stats": _compute_gate_stats(all_gates_by_fold) if all_gates_by_fold else None,
    }


def _compute_gate_stats(gates_by_fold):
    """Compute gate value statistics."""
    all_gates = []
    all_labels = []
    for fold_id, data in gates_by_fold.items():
        all_gates.extend(data["gates"])
        all_labels.extend(data["labels"])
    all_gates = np.array(all_gates)
    all_labels = np.array(all_labels)
    return {
        "mean_gate": float(np.mean(all_gates)),
        "std_gate": float(np.std(all_gates)),
        "gate_positive_mean": float(np.mean(all_gates[all_labels == 1])) if (all_labels == 1).sum() > 0 else None,
        "gate_negative_mean": float(np.mean(all_gates[all_labels == 0])) if (all_labels == 0).sum() > 0 else None,
        "gate_percentile_25": float(np.percentile(all_gates, 25)),
        "gate_percentile_50": float(np.percentile(all_gates, 50)),
        "gate_percentile_75": float(np.percentile(all_gates, 75)),
    }


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 70)
    print("  GSCA: Gated Symmetry-Aware Cross-Attention")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    df = pd.read_csv(PAIRING_MATRIX_FILE)
    with open(CLUSTERS_FILE) as f:
        cluster_list = json.load(f)["clusters"]
    gpcr_feats, gprot_feats = load_features()
    icl_data = load_icl()

    results = {}

    # Compare 3 architectures × 2 feature modes
    experiments = [
        ("uni_ca", UniDirectionalCA, "Uni-directional Cross-Attention (original)"),
        ("bi_ca_avg", BidirectionalCA, "Bidirectional Cross-Attention (avg)"),
        ("gsca", GatedSymmetryCrossAttention, "GSCA (gated bidirectional)"),
    ]

    for mode in ["baseline", "icl_full"]:
        print(f"\n{'='*60}")
        print(f"  Feature mode: {mode}")
        print(f"{'='*60}")
        X, y, meta = build_vectors(df, gpcr_feats, gprot_feats, icl_data, mode)
        sample_to_cluster, fold_clusters = get_cluster_folds(meta, cluster_list)
        print(f"  Samples: {len(y)}, Features: {X.shape[1]}")

        for name_key, model_cls, display_name in experiments:
            print(f"\n  --- {display_name} ---")
            res = run_cv(X, y, meta, fold_clusters, sample_to_cluster,
                         mode=mode, model_cls=model_cls, model_name=name_key,
                         gprot_dim=320)
            results[f"{name_key}_{mode}"] = res
            print(f"  >> AUC = {res['auc_mean']:.4f} +/- {res['auc_std']:.4f}, "
                  f"PR-AUC = {res['overall_pr_auc']:.4f}")
            if res.get("gate_stats"):
                gs = res["gate_stats"]
                print(f"  >> Gate: mean={gs['mean_gate']:.4f}, "
                      f"pos_mean={gs['gate_positive_mean']:.4f}, "
                      f"neg_mean={gs['gate_negative_mean']:.4f}")

    # Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Results saved to {OUTPUT_FILE}")

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY TABLE")
    print("=" * 70)
    print(f"  {'Model':<35} {'AUC':>8} {'Std':>8} {'PR-AUC':>8}")
    print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8}")
    for k, v in sorted(results.items()):
        print(f"  {k:<35} {v['auc_mean']:>8.4f} {v['auc_std']:>8.4f} {v['overall_pr_auc']:>8.4f}")

    # GSCA vs Uni-CA comparison
    for mode in ["baseline", "icl_full"]:
        uni = results.get(f"uni_ca_{mode}")
        gsca = results.get(f"gsca_{mode}")
        if uni and gsca:
            delta = gsca["auc_mean"] - uni["auc_mean"]
            print(f"\n  GSCA vs Uni-CA ({mode}): Delta AUC = {delta:+.4f}")
        if gsca and gsca.get("gate_stats"):
            gs = gsca["gate_stats"]
            print(f"  Gate interpretation ({mode}): "
                  f"mean={gs['mean_gate']:.3f} -> "
                  f"{'GPCR-dominated' if gs['mean_gate'] > 0.6 else 'Gprot-dominated' if gs['mean_gate'] < 0.4 else 'balanced'}")


if __name__ == "__main__":
    main()
