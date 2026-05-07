#!/usr/bin/env python3
"""
Gradient-based feature attribution for the 650M Cross-Attention model.
Computes average absolute gradient of the predicted probability w.r.t. each input feature.
"""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
DATA_DIR = BASE / "paired_dataset"
GPCR_FEATURES_FILE = BASE / "server_sync" / "extended_data" / "features" / "esm_features_650m_meanpool.json"
G_PROTEIN_FEATURES_FILE = DATA_DIR / "g_protein_esm_features.json"
ICL_FEATURES_FILE = DATA_DIR / "icl_features_650m.json"
ALPHA_FEATURES_FILE = DATA_DIR / "alphafold_icl_features.json"
PAIRING_MATRIX_FILE = DATA_DIR / "pairing_matrix_raw.csv"
OUTPUT_DIR = BASE / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        "tm_mean_plddt", "global_plddt_mean", "global_plddt_std",
        "high_confidence_ratio_70", "high_confidence_ratio_90",
        "sasa_mean", "sasa_buried_ratio", "contact_density", "mean_contacts_per_residue",
        "tm5_tm6_cyto_ca_distance", "icl2_end_to_end_ca_distance",
        "icl3_end_to_end_ca_distance", "tm5_tm6_cyto_dihedral_angle",
        "icl2_aromatic_centroid_depth", "interface_patch_sasa",
        "interface_patch_sasa_ratio", "icl2_helix_ratio", "icl2_sheet_ratio",
        "icl2_coil_ratio", "icl3_helix_ratio", "icl3_sheet_ratio", "icl3_coil_ratio",
        "icl2_mean_pae", "icl2_intra_pae", "icl3_mean_pae", "icl3_intra_pae",
        "icl2_tm5_pae", "icl2_tm6_pae", "icl3_tm5_pae", "icl3_tm6_pae",
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
        meta.append({"gpcr_id": gid, "g_protein_family": gfam, "coupling": int(row["coupling"])})
    return np.array(X_list), np.array(y_list), meta


class PairedCrossAttentionNet(nn.Module):
    def __init__(self, gpcr_dim=1280, gprot_dim=320, hidden_dim=256, num_heads=4, dropout=0.3):
        super().__init__()
        self.gpcr_proj = nn.Sequential(
            nn.Linear(gpcr_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
        )
        self.gprot_proj = nn.Sequential(
            nn.Linear(gprot_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
        )
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, gpcr_feat, gprot_feat):
        q = self.gpcr_proj(gpcr_feat).unsqueeze(1)
        kv = self.gprot_proj(gprot_feat).unsqueeze(1)
        attn_out, _ = self.cross_attn(q, kv, kv)
        x = torch.cat([attn_out.squeeze(1), self.gpcr_proj(gpcr_feat)], dim=-1)
        return self.ffn(x).squeeze(-1)


class PairDataset(Dataset):
    def __init__(self, X, y, gpcr_total_dim):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.gpcr_total_dim = gpcr_total_dim

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        return x[:self.gpcr_total_dim], x[1280:1600], self.y[idx]


def train_epoch(model, loader, optimizer, criterion):
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


def compute_gradients(model, loader, gpcr_total_dim):
    model.eval()
    all_grads = []
    for gpcr, gprot, y in loader:
        gpcr, gprot = gpcr.to(DEVICE), gprot.to(DEVICE)
        gpcr.requires_grad_(True)
        gprot.requires_grad_(True)
        out = torch.sigmoid(model(gpcr, gprot))
        loss = out.sum()
        loss.backward()
        grad = torch.cat([gpcr.grad, gprot.grad], dim=1).abs().cpu().numpy()
        all_grads.append(grad)
        gpcr.requires_grad_(False)
        gprot.requires_grad_(False)
        model.zero_grad()
    return np.concatenate(all_grads, axis=0)


def plot_feature_group_importance(mean_grads, gpcr_total_dim, output_path):
    # Define feature groups
    gpcr_esm_end = 1280
    icl2_esm_end = 1280 + 1280
    icl2_stat_end = 1280 + 1280 + 8
    icl3_esm_end = 1280 + 1280 + 8 + 1280
    icl3_stat_end = 1280 + 1280 + 8 + 1280 + 8
    alpha_end = gpcr_total_dim

    groups = {
        "GPCR ESM global": mean_grads[:gpcr_esm_end].mean(),
        "ICL2 ESM": mean_grads[gpcr_esm_end:icl2_esm_end].mean(),
        "ICL2 stats": mean_grads[icl2_esm_end:icl2_stat_end].mean(),
        "ICL3 ESM": mean_grads[icl2_stat_end:icl3_esm_end].mean(),
        "ICL3 stats": mean_grads[icl3_esm_end:icl3_stat_end].mean(),
        "AlphaFold": mean_grads[icl3_stat_end:alpha_end].mean() if alpha_end > icl3_stat_end else 0.0,
        "G-protein ESM": mean_grads[gpcr_total_dim:gpcr_total_dim+320].mean(),
    }

    names = list(groups.keys())
    vals = list(groups.values())

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#3498db", "#2ecc71", "#27ae60", "#2ecc71", "#27ae60", "#95a5a6", "#e74c3c"]
    bars = ax.barh(names, vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Mean absolute gradient", fontweight="bold")
    ax.set_title("Feature-group sensitivity in 650M Cross-Attention model", fontweight="bold", fontsize=13)
    for bar, val in zip(bars, vals):
        ax.text(val + 0.0005, bar.get_y() + bar.get_height()/2, f"{val:.4f}", va="center", fontsize=10)
    sns.despine(top=True, right=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"[OK] Saved feature-group importance to {output_path}")


def main():
    print("=" * 70)
    print("  Gradient-based attribution for 650M Cross-Attention")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    df = pd.read_csv(PAIRING_MATRIX_FILE)
    gpcr_feats, gprot_feats = load_features()
    icl_data = load_icl_features()
    alpha_data = load_alpha_features()

    mode = "alpha" if alpha_data else "icl_full"
    X, y, meta = build_vectors(df, gpcr_feats, gprot_feats, icl_data, alpha_data, mode=mode)
    print(f"[INFO] Training set: {len(y)} samples, {X.shape[1]} features, mode={mode}")

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    gpcr_total_dim = 1280 + 2576 + 38 if mode == "alpha" else 1280 + 2576
    ds = PairDataset(Xs, y, gpcr_total_dim)
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    model = PairedCrossAttentionNet(gpcr_dim=gpcr_total_dim, gprot_dim=320, hidden_dim=256, num_heads=4, dropout=0.3).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    pos_weight = len(y) / max(1, y.sum())
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([pos_weight]).to(DEVICE))

    for epoch in range(80):
        loss = train_epoch(model, loader, optimizer, criterion)
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d} loss={loss:.4f}")

    print("[INFO] Computing gradients...")
    pred_loader = DataLoader(PairDataset(Xs, y, gpcr_total_dim), batch_size=32)
    grads = compute_gradients(model, pred_loader, gpcr_total_dim)
    mean_grads = grads.mean(axis=0)

    plot_feature_group_importance(mean_grads, gpcr_total_dim, OUTPUT_DIR / "figure5_feature_group_importance.png")

    # Save top individual dimensions
    top_idx = np.argsort(mean_grads)[::-1][:20]
    results = {"top_dimensions": [{"dim": int(i), "mean_abs_grad": float(mean_grads[i])} for i in top_idx],
               "feature_group_means": {k: float(v) for k, v in {
                   "gpcr_esm_global": mean_grads[:1280].mean(),
                   "icl2_esm": mean_grads[1280:2560].mean(),
                   "icl2_stats": mean_grads[2560:2568].mean(),
                   "icl3_esm": mean_grads[2568:3848].mean(),
                   "icl3_stats": mean_grads[3848:3856].mean(),
                   "alphafold": mean_grads[3856:gpcr_total_dim].mean() if gpcr_total_dim > 3856 else 0.0,
                   "gprotein_esm": mean_grads[gpcr_total_dim:gpcr_total_dim+320].mean(),
               }.items()}}
    with open(OUTPUT_DIR / "gradient_attribution_650m.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[OK] Saved attribution stats to {OUTPUT_DIR / 'gradient_attribution_650m.json'}")


if __name__ == "__main__":
    main()
