#!/usr/bin/env python3
"""
Final GPCR-IPN ablation: compare mean-pooling vs SGSP with GSCA+IPL.

Configurations:
  1. Mean-pooled + BCE (baseline)
  2. Mean-pooled + BCE + Proto (IPL)
  3. SGSP + BCE
  4. SGSP + BCE + Proto (full GPCR-IPN)

All use GSCA encoder + ICL-full features.
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
GPCR_MEAN_FILE = BASE / "server_sync" / "extended_data" / "features" / "esm_features_650m_meanpool.json"
GPCR_SGSP_FILE = DATA_DIR / "sgsp_embeddings_650m.json"
GPROT_FILE = DATA_DIR / "g_protein_esm_features.json"
ICL_FILE = DATA_DIR / "icl_features_650m.json"
PAIRING_FILE = DATA_DIR / "pairing_matrix_raw.csv"
CLUSTERS_FILE = DATA_DIR / "sequence_clusters.json"
OUTPUT_FILE = DATA_DIR / "final_ablation_results.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBED_DIM = 64

FAMILY_ORDER = ["G12_13", "Gi", "Gq", "Gs"]
NUM_FAMILIES = 4


def load_gpcr(path):
    with open(path) as f:
        raw = json.load(f)
    if isinstance(list(raw.values())[0], dict) and "embedding" in list(raw.values())[0]:
        return {k: np.array(v["embedding"]) for k, v in raw.items()}
    return {k: np.array(v) for k, v in raw.items()}


def load_gprot():
    with open(GPROT_FILE) as f:
        raw = json.load(f)
    feats = {}
    fm = {"GNAQ": "Gq", "GNAI1": "Gi", "GNAI2": "Gi", "GNAI3": "Gi",
          "GNAS": "Gs", "GNA12": "G12_13", "GNA13": "G12_13"}
    for sub, info in raw.items():
        f = fm.get(sub, sub)
        vec = np.array(info["mean_pooling"])
        feats[sub] = vec
        if f not in feats: feats[f] = vec
    return feats


def load_icl():
    if not ICL_FILE.exists(): return {}
    with open(ICL_FILE) as f: return json.load(f)


def get_gpcr_feat(feats, gid):
    feat = feats.get(gid)
    if feat is None and "_" in gid and len(gid.split("_")[0]) <= 2:
        base = gid.split("_", 1)[1]
        feat = feats.get(base)
    if feat is None:
        for k in feats:
            if "_" in k and k.split("_", 1)[1] == gid:
                feat = feats[k]; break
    return feat


def get_icl_vec(icl_data, gid, dim=1280):
    rec = icl_data.get(gid)
    if rec is None:
        for k in icl_data:
            if "_" in k and k.split("_", 1)[1] == gid:
                rec = icl_data[k]; break
    i2e = np.array(rec.get("ICL2_esm", [])) if rec else np.array([])
    i3e = np.array(rec.get("ICL3_esm", [])) if rec else np.array([])
    s2 = rec.get("ICL2_stats", {}) if rec else {}
    s3 = rec.get("ICL3_stats", {}) if rec else {}
    if i2e.size == 0: i2e = np.zeros(dim)
    if i3e.size == 0: i3e = np.zeros(dim)
    sk = ["length","mean_hydro","std_hydro","net_charge","pos_charge_ratio",
          "neg_charge_ratio","hydrophobic_ratio","aromatic_ratio"]
    return i2e, np.array([s2.get(k, 0.0) for k in sk]), i3e, np.array([s3.get(k, 0.0) for k in sk])


def build_vectors(df, gpcr_feats, gprot_feats, icl_data):
    X_list, y_list, fi_list, meta = [], [], [], []
    for _, row in df.iterrows():
        gid = row["gpcr_id"]
        gf = get_gpcr_feat(gpcr_feats, gid)
        gfam = row["g_protein_family"]
        pf = gprot_feats.get(gfam)
        if pf is None: pf = gprot_feats.get(gfam.capitalize())
        if pf is None: pf = gprot_feats.get(gfam.upper())
        if gf is None or pf is None: continue
        parts = [np.concatenate([gf, pf])]
        i2, i2s, i3, i3s = get_icl_vec(icl_data, gid)
        parts.append(np.concatenate([i2, i2s, i3, i3s]))
        X_list.append(np.concatenate(parts))
        y_list.append(int(row["coupling"]))
        fi = FAMILY_ORDER.index(gfam) if gfam in FAMILY_ORDER else -1
        fi_list.append(fi)
        meta.append({"gpcr_id": gid, "g_protein_family": gfam,
                     "cluster_id": int(row["cluster_id"]), "family_idx": fi})
    return np.array(X_list), np.array(y_list), np.array(fi_list), meta


def get_folds(meta, cluster_list, n_folds=5):
    n = len(meta)
    s2c = {i: meta[i]["cluster_id"] for i in range(n)}
    cs = defaultdict(int)
    for i in range(n): cs[s2c[i]] += 1
    fc = [[] for _ in range(n_folds)]
    fs = [0] * n_folds
    sc = sorted(range(len(cluster_list)), key=lambda c: cs[c], reverse=True)
    for cid in sc:
        if cs.get(cid, 0) == 0: continue
        t = int(np.argmin(fs))
        fc[t].append(cid); fs[t] += cs[cid]
    return s2c, fc


# ===== Model =====

class GPCR_IPN(nn.Module):
    """GSCA encoder + optional prototypical regularization."""
    def __init__(self, gpcr_dim=1280+2576, gprot_dim=320, hidden_dim=256, num_heads=4, dropout=0.3):
        super().__init__()
        self.gpcr_proj = nn.Sequential(nn.Linear(gpcr_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.gprot_proj = nn.Sequential(nn.Linear(gprot_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.gate_linear = nn.Linear(hidden_dim * 4, 1)
        nn.init.constant_(self.gate_linear.bias, 0.7)
        self.gate_dropout = nn.Dropout(dropout)
        self.interaction_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, EMBED_DIM),
        )
        self.classifier = nn.Linear(EMBED_DIM, 1)

    def forward(self, gpcr_feat, gprot_feat, return_all=False):
        hg = self.gpcr_proj(gpcr_feat)
        hp = self.gprot_proj(gprot_feat)
        a_g2p, _ = self.cross_attn(hg.unsqueeze(1), hp.unsqueeze(1), hp.unsqueeze(1))
        a_p2g, _ = self.cross_attn(hp.unsqueeze(1), hg.unsqueeze(1), hg.unsqueeze(1))
        a_g2p, a_p2g = a_g2p.squeeze(1), a_p2g.squeeze(1)
        gi = self.gate_dropout(torch.cat([hg, hp, a_g2p, a_p2g], dim=-1))
        gate = torch.sigmoid(self.gate_linear(gi))
        hf = gate * a_g2p + (1 - gate) * a_p2g
        embed = self.interaction_proj(torch.cat([hf, hg], dim=-1))
        logit = self.classifier(embed).squeeze(-1)
        if return_all: return logit, embed, gate.squeeze(-1)
        return logit


def proto_reg(embeddings, family_idx, labels, margin=1.0):
    device = embeddings.device
    protos = {}
    for f in range(NUM_FAMILIES):
        m = (family_idx == f)
        if m.sum() > 0: protos[f] = embeddings[m].mean(dim=0)
    if len(protos) < 2: return torch.tensor(0.0, device=device)
    loss = torch.tensor(0.0, device=device)
    n = 0
    for f, p in protos.items():
        pos = (family_idx == f) & (labels == 1)
        if pos.sum() > 0: loss += torch.norm(embeddings[pos] - p.unsqueeze(0), dim=1).mean(); n += 1
        neg = (family_idx == f) & (labels == 0)
        if neg.sum() > 0: loss += torch.clamp(margin - torch.norm(embeddings[neg] - p.unsqueeze(0), dim=1), min=0).mean(); n += 1
    pl = list(protos.values())
    for i in range(len(pl)):
        for j in range(i+1, len(pl)):
            loss += torch.clamp(margin - torch.norm(pl[i] - pl[j]), min=0); n += 1
    return loss / max(1, n)


# ===== Training =====

class PairDataset(Dataset):
    def __init__(self, X, y, fi, gprot_dim=320):
        self.X = torch.FloatTensor(X); self.y = torch.FloatTensor(y)
        self.fi = torch.LongTensor(fi); self.gprot_dim = gprot_dim
    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        x = self.X[idx]
        gpcr_global = x[:1280]  # GPCR embedding (mean or SGSP) is first 1280
        gp = x[1280:1280 + self.gprot_dim]
        icl = x[1280 + self.gprot_dim:1280 + self.gprot_dim + 2576]
        return torch.cat([gpcr_global, icl]), gp, self.y[idx], self.fi[idx]


def train_epoch(model, loader, optim, crit, proto_w=0.0):
    model.train(); tl = 0.0
    for gpc, gp, y, fi in loader:
        gpc, gp, y, fi = gpc.to(DEVICE), gp.to(DEVICE), y.to(DEVICE), fi.to(DEVICE)
        optim.zero_grad()
        logit, emb, gate = model(gpc, gp, return_all=True)
        loss = crit(logit, y)
        if proto_w > 0: loss += proto_w * proto_reg(emb, fi, y)
        if gate.numel() > 0: loss -= 0.02 * torch.mean(torch.abs(gate - 0.5))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        tl += loss.item() * len(y)
    return tl / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader):
    model.eval(); probs, labels = [], []
    for gpc, gp, y, _ in loader:
        gpc, gp = gpc.to(DEVICE), gp.to(DEVICE)
        logit, _, _ = model(gpc, gp, return_all=True)
        probs.append(torch.sigmoid(logit).cpu().numpy())
        labels.append(y.numpy())
    if not probs:  # empty loader
        return np.array([]), np.array([])
    return np.concatenate(probs), np.concatenate(labels)


def run_cv(X, y, fi, meta, fold_clusters, s2c, proto_w=0.0,
           n_folds=5, epochs=80, lr=1e-4, batch_size=32):
    scaler = StandardScaler(); Xs = scaler.fit_transform(X)
    n = len(y); fold_metrics = []
    for f in range(n_folds):
        tc = set(fold_clusters[f])
        te_idx = [i for i in range(n) if s2c[i] in tc]
        tr_idx = [i for i in range(n) if s2c[i] not in tc]
        X_tr, X_te = Xs[tr_idx], Xs[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        fi_tr, fi_te = fi[tr_idx], fi[te_idx]
        tr_ds = PairDataset(X_tr, y_tr, fi_tr); te_ds = PairDataset(X_te, y_te, fi_te)
        tr_ld = DataLoader(tr_ds, batch_size, shuffle=True)
        te_ld = DataLoader(te_ds, batch_size // 2)
        model = GPCR_IPN().to(DEVICE)
        optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        pw = torch.FloatTensor([len(y_tr) / max(1, y_tr.sum())]).to(DEVICE)
        crit = nn.BCEWithLogitsLoss(pos_weight=pw)
        best_auc, best_p, best_l = -1.0, None, None
        patience = 0
        for ep in range(epochs):
            train_epoch(model, tr_ld, optim, crit, proto_w)
            p, l = evaluate(model, te_ld)
            auc = roc_auc_score(l, p) if len(set(l)) >= 2 else -1
            if auc > best_auc: best_auc = auc; best_p, best_l = p.copy(), l.copy(); patience = 0
            else: patience += 1
            if patience >= 15: break
        fold_metrics.append({"auc": float(best_auc), "probs": best_p.tolist(), "labels": best_l.tolist()})
        print(f"    Fold {f+1}: AUC = {best_auc:.4f}")
    aucs = [m["auc"] for m in fold_metrics if not np.isnan(m["auc"])]
    all_p = np.concatenate([m["probs"] for m in fold_metrics])
    all_l = np.concatenate([m["labels"] for m in fold_metrics])
    return {"auc_mean": round(float(np.mean(aucs)), 4), "auc_std": round(float(np.std(aucs)), 4),
            "fold_aucs": [round(m["auc"], 4) for m in fold_metrics],
            "pr_auc": round(float(average_precision_score(all_l, all_p)), 4)}


# ===== Main =====

def main():
    print("=" * 70)
    print("  GPCR-IPN: Final Ablation Study")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    df = pd.read_csv(PAIRING_FILE)
    with open(CLUSTERS_FILE) as f: cluster_list = json.load(f)["clusters"]
    gprot = load_gprot()
    icl = load_icl()

    results = {}
    configs = [
        ("mean_pooled_BCE", load_gpcr(GPCR_MEAN_FILE), 0.0),
        ("mean_pooled_IPL", load_gpcr(GPCR_MEAN_FILE), 0.1),
        ("SGSP_BCE", load_gpcr(GPCR_SGSP_FILE), 0.0),
        ("SGSP_IPL", load_gpcr(GPCR_SGSP_FILE), 0.1),
    ]

    for name, gpcr, pw in configs:
        print(f"\n{'='*60}")
        print(f"  Config: {name}")
        print(f"{'='*60}")
        X, y, fi, meta = build_vectors(df, gpcr, gprot, icl)
        s2c, fc = get_folds(meta, cluster_list)
        print(f"  Samples: {len(y)}, Features: {X.shape[1]}")
        res = run_cv(X, y, fi, meta, fc, s2c, proto_w=pw)
        results[name] = res
        print(f"  >> AUC = {res['auc_mean']:.4f} +/- {res['auc_std']:.4f}, PR-AUC = {res['pr_auc']:.4f}")

    with open(OUTPUT_FILE, "w") as f: json.dump(results, f, indent=2)
    print(f"\n[OK] Saved to {OUTPUT_FILE}")

    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)
    print(f"  {'Config':<25} {'AUC':>8} {'Std':>8} {'PR-AUC':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8}")
    for k, v in results.items():
        print(f"  {k:<25} {v['auc_mean']:>8.4f} {v['auc_std']:>8.4f} {v['pr_auc']:>8.4f}")
    if "SGSP_IPL" in results and "mean_pooled_BCE" in results:
        d = results["SGSP_IPL"]["auc_mean"] - results["mean_pooled_BCE"]["auc_mean"]
        print(f"\n  GPCR-IPN vs mean-pooled BCE: Delta AUC = {d:+.4f}")


if __name__ == "__main__":
    main()
