#!/usr/bin/env python3
"""
Experiment: Replace 8M (320-d) G protein embeddings with 650M (1280-d).
Compares SVM, Cross-Attention, MLP under baseline and ICL-full configs.

Key question: does upgrading G protein to 1280-d close the GPCR/Gprot sensitivity gap?
"""

import json, numpy as np, pandas as pd, torch, torch.nn as nn
from pathlib import Path
from collections import defaultdict
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import Dataset, DataLoader

BASE = Path(__file__).parent
DATA_DIR = BASE / "paired_dataset"
GPCR_FEATURES_FILE = BASE / "server_sync" / "extended_data" / "features" / "esm_features_650m_meanpool.json"
GPROT_ORIG_FILE = DATA_DIR / "g_protein_esm_features.json"          # 320-d (original)
GPROT_650M_FILE = DATA_DIR / "g_protein_esm_features_650m.json"      # 1280-d (new)
ICL_FEATURES_FILE = DATA_DIR / "icl_features_650m.json"
ALPHA_FEATURES_FILE = DATA_DIR / "alphafold_icl_features.json"
PAIRING_MATRIX_FILE = DATA_DIR / "pairing_matrix_raw.csv"
CLUSTERS_FILE = DATA_DIR / "sequence_clusters.json"
OUTPUT_FILE = DATA_DIR / "gprot_650m_experiment_results.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Feature Loading
# ---------------------------------------------------------------------------

def load_gpcr_features():
    with open(GPCR_FEATURES_FILE) as f:
        raw = json.load(f)
    return {k: np.array(v) for k, v in raw.items()}

def load_gprot_features(path, is_650m=False):
    with open(path) as f:
        raw = json.load(f)
    gprot_feats = {}
    family_map = {
        "GNAQ": "Gq", "GNAI1": "Gi", "GNAI2": "Gi", "GNAI3": "Gi",
        "GNAS": "Gs", "GNA12": "G12_13", "GNA13": "G12_13",
    }
    for subtype, info in raw.items():
        family = family_map.get(subtype, subtype)
        vec = np.array(info["mean_pooling"])
        assert vec.ndim == 1, f"{subtype} embedding not 1-d: {vec.shape}"
        gprot_feats[subtype] = vec
        gprot_feats[family] = vec  # family gets same embedding as first subtype
    return gprot_feats

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
    icl2_stats = rec.get("ICL2_stats", {}) if rec else {}
    icl3_stats = rec.get("ICL3_stats", {}) if rec else {}
    if icl2_esm.size == 0: icl2_esm = np.zeros(dim)
    if icl3_esm.size == 0: icl3_esm = np.zeros(dim)
    skeys = ["length","mean_hydro","std_hydro","net_charge","pos_charge_ratio",
             "neg_charge_ratio","hydrophobic_ratio","aromatic_ratio"]
    icl2_s = np.array([icl2_stats.get(k, 0.0) for k in skeys])
    icl3_s = np.array([icl3_stats.get(k, 0.0) for k in skeys])
    return icl2_esm, icl2_s, icl3_esm, icl3_s

def build_vectors(df, gpcr_feats, gprot_feats, icl_data, mode="baseline", gprot_dim=320):
    X_list, y_list, metas = [], [], []
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
        if mode in ("icl_full", "alpha"):
            icl2_e, icl2_s, icl3_e, icl3_s = get_icl_vector(icl_data, gid, dim=1280)
            parts.append(np.concatenate([icl2_e, icl2_s, icl3_e, icl3_s]))
        X_list.append(np.concatenate(parts))
        y_list.append(int(row["coupling"]))
        metas.append({"gpcr_id": gid, "g_protein_family": gfam, "cluster_id": int(row["cluster_id"])})
    return np.array(X_list), np.array(y_list), metas

# ---------------------------------------------------------------------------
# Cluster-aware CV splitting
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class PairedCrossAttentionNet(nn.Module):
    def __init__(self, gpcr_dim=1280, gprot_dim=320, hidden_dim=256, num_heads=4, dropout=0.3):
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

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(512, 256), dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x).squeeze(-1)

class PairDataset(Dataset):
    """Splits X into (GPCR-side features, G-protein embedding, label).

    X layout per sample: [GPCR(1280) | Gprot(d) | ICL(2576)]
    GPCR-side for the model = [GPCR(1280), ICL(2576)] = 3856-d (for icl_full)
    Gprot embedding = sliced from position 1280, length gprot_dim.
    """
    def __init__(self, X, y, mode="baseline", gprot_dim=320):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.mode = mode
        self.gprot_dim = gprot_dim
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        x = self.X[idx]
        # GPCR global embedding: first 1280 dims
        gpcr_global = x[:1280]
        # G protein embedding: starts at 1280, length gprot_dim
        gprot_vec = x[1280:1280 + self.gprot_dim]
        if self.mode == "baseline":
            # GPCR-side = just GPCR global (1280)
            gpcr_side = gpcr_global
        elif self.mode == "icl_full":
            # ICL features start at 1280 + gprot_dim
            icl_start = 1280 + self.gprot_dim
            icl_features = x[icl_start:icl_start + 2576]
            gpcr_side = torch.cat([gpcr_global, icl_features])
        else:
            gpcr_side = gpcr_global  # fallback
        return gpcr_side, gprot_vec, self.y[idx]

# ---------------------------------------------------------------------------
# Training & Evaluation
# ---------------------------------------------------------------------------

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
def evaluate(model, loader):
    model.eval()
    probs, labels = [], []
    for gpcr, gprot, y in loader:
        gpcr, gprot = gpcr.to(DEVICE), gprot.to(DEVICE)
        probs.append(torch.sigmoid(model(gpcr, gprot)).cpu().numpy())
        labels.append(y.numpy())
    probs, labels = np.concatenate(probs), np.concatenate(labels)
    return probs, labels

def compute_metrics(labels, probs):
    preds = (probs >= 0.5).astype(int)
    return {
        "auc": roc_auc_score(labels, probs) if len(set(labels)) >= 2 else float("nan"),
        "pr_auc": average_precision_score(labels, probs),
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }

def run_cross_attention_cv(X, y, meta, fold_clusters, sample_to_cluster, mode,
                           gprot_dim=320, n_folds=5, epochs=80, lr=1e-4, batch_size=32):
    gpcr_dim = {"baseline": 1280, "icl_full": 1280 + 2576}[mode]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    n = len(y)
    all_probs, all_labels = np.array([]), np.array([])
    fold_metrics = []
    for f in range(n_folds):
        test_clusters = set(fold_clusters[f])
        test_idx = [i for i in range(n) if sample_to_cluster[i] in test_clusters]
        train_idx = [i for i in range(n) if sample_to_cluster[i] not in test_clusters]
        X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        train_ds = PairDataset(X_tr, y_tr, mode, gprot_dim)
        test_ds = PairDataset(X_te, y_te, mode, gprot_dim)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size)
        model = PairedCrossAttentionNet(gpcr_dim=gpcr_dim, gprot_dim=gprot_dim).to(DEVICE)
        optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        crit = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([len(y_tr)/max(1,y_tr.sum())]).to(DEVICE))
        best_auc = -1.0
        best_probs = None
        patience = 0
        for ep in range(epochs):
            train_epoch(model, train_loader, optim, crit)
            p, lbl = evaluate(model, test_loader)
            auc_v = roc_auc_score(lbl, p) if len(set(lbl)) >= 2 else -1
            if auc_v > best_auc:
                best_auc = auc_v
                best_probs = p
                patience = 0
            else:
                patience += 1
            if patience >= 15: break
        fm = compute_metrics(y_te, best_probs)
        fold_metrics.append(fm)
        all_probs = np.concatenate([all_probs, best_probs]) if len(all_probs) else best_probs
        all_labels = np.concatenate([all_labels, y_te]) if len(all_labels) else y_te
    aucs = [m["auc"] for m in fold_metrics if not np.isnan(m["auc"])]
    return {
        "auc_mean": round(float(np.mean(aucs)), 4),
        "auc_std": round(float(np.std(aucs)), 4),
        "fold_metrics": fold_metrics,
        "overall_pr_auc": round(average_precision_score(all_labels, all_probs), 4),
    }

def run_mlp_cv(X, y, meta, fold_clusters, sample_to_cluster, input_dim,
               n_folds=5, epochs=80, lr=1e-4, batch_size=32):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    n = len(y)
    all_probs, all_labels = np.array([]), np.array([])
    fold_metrics = []
    for f in range(n_folds):
        test_clusters = set(fold_clusters[f])
        test_idx = [i for i in range(n) if sample_to_cluster[i] in test_clusters]
        train_idx = [i for i in range(n) if sample_to_cluster[i] not in test_clusters]
        X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        tr_ds = torch.utils.data.TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr))
        te_ds = torch.utils.data.TensorDataset(torch.FloatTensor(X_te), torch.FloatTensor(y_te))
        tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
        te_loader = DataLoader(te_ds, batch_size=batch_size)
        model = MLP(input_dim).to(DEVICE)
        optim = torch.optim.Adam(model.parameters(), lr=lr)
        crit = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([len(y_tr)/max(1,y_tr.sum())]).to(DEVICE))
        best_auc = -1.0
        best_probs = None
        patience = 0
        for ep in range(epochs):
            model.train()
            for bx, by in tr_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                optim.zero_grad()
                crit(model(bx), by).backward()
                optim.step()
            model.eval()
            with torch.no_grad():
                p_all = torch.sigmoid(model(te_loader.dataset.tensors[0].to(DEVICE))).cpu().numpy()
            lbl = te_loader.dataset.tensors[1].numpy()
            auc_v = roc_auc_score(lbl, p_all) if len(set(lbl)) >= 2 else -1
            if auc_v > best_auc:
                best_auc = auc_v; best_probs = p_all; patience = 0
            else:
                patience += 1
            if patience >= 15: break
        fm = compute_metrics(y_te, best_probs)
        fold_metrics.append(fm)
        all_probs = np.concatenate([all_probs, best_probs]) if len(all_probs) else best_probs
        all_labels = np.concatenate([all_labels, y_te]) if len(all_labels) else y_te
    aucs = [m["auc"] for m in fold_metrics if not np.isnan(m["auc"])]
    return {
        "auc_mean": round(float(np.mean(aucs)), 4),
        "auc_std": round(float(np.std(aucs)), 4),
        "fold_metrics": fold_metrics,
        "overall_pr_auc": round(average_precision_score(all_labels, all_probs), 4),
    }

def run_svm_cv(X, y, meta, fold_clusters, sample_to_cluster, n_folds=5):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    n = len(y)
    all_probs, all_labels = np.array([]), np.array([])
    fold_metrics = []
    for f in range(n_folds):
        test_clusters = set(fold_clusters[f])
        test_idx = [i for i in range(n) if sample_to_cluster[i] in test_clusters]
        train_idx = [i for i in range(n) if sample_to_cluster[i] not in test_clusters]
        X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        svm = SVC(C=10.0, kernel="rbf", class_weight="balanced", probability=True, random_state=42)
        svm.fit(X_tr, y_tr)
        probs = svm.predict_proba(X_te)[:, 1]
        fm = compute_metrics(y_te, probs)
        fold_metrics.append(fm)
        all_probs = np.concatenate([all_probs, probs]) if len(all_probs) else probs
        all_labels = np.concatenate([all_labels, y_te]) if len(all_labels) else y_te
    aucs = [m["auc"] for m in fold_metrics if not np.isnan(m["auc"])]
    return {
        "auc_mean": round(float(np.mean(aucs)), 4),
        "auc_std": round(float(np.std(aucs)), 4),
        "fold_metrics": fold_metrics,
        "overall_pr_auc": round(average_precision_score(all_labels, all_probs), 4),
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  G Protein 650M Embedding Experiment")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    df = pd.read_csv(PAIRING_MATRIX_FILE)
    with open(CLUSTERS_FILE) as f:
        cluster_list = json.load(f)["clusters"]
    gpcr_feats = load_gpcr_features()
    icl_data = load_icl()
    print(f"  GPCR features: {len(gpcr_feats)}")
    print(f"  Pairs: {len(df)}")

    results = {}
    gprot_configs = [
        ("gprot_8m_320d", GPROT_ORIG_FILE, 320),
        ("gprot_650m_1280d", GPROT_650M_FILE, 1280),
    ]

    for suffix, gprot_path, gprot_dim in gprot_configs:
        print(f"\n{'='*60}")
        print(f"  Config: G protein = {suffix}")
        print(f"{'='*60}")
        gprot_feats = load_gprot_features(gprot_path, is_650m=(gprot_dim==1280))

        for mode in ["baseline", "icl_full"]:
            print(f"\n  --- Mode: {mode} ---")
            X, y, meta = build_vectors(df, gpcr_feats, gprot_feats, icl_data, mode=mode, gprot_dim=gprot_dim)
            sample_to_cluster, fold_clusters = get_cluster_folds(meta, cluster_list)
            print(f"  Samples: {len(y)}, Features: {X.shape[1]}")

            # SVM
            print(f"  [SVM] ...")
            svm_res = run_svm_cv(X, y, meta, fold_clusters, sample_to_cluster)
            key_svm = f"svm_{suffix}_{mode}"
            results[key_svm] = svm_res
            print(f"    AUC = {svm_res['auc_mean']:.4f} ± {svm_res['auc_std']:.4f}, PR-AUC = {svm_res['overall_pr_auc']:.4f}")

            # gpcr_total_dim = offset where G protein embedding starts in the concatenated vector
            # Vector layout: [GPCR(1280) | Gprot(dim) | ICL(2576)]
            # So gpcr_total_dim = 1280 for baseline, 1280+2576 for icl_full (regardless of gprot_dim)
            non_gprot_dim = {"baseline": 1280, "icl_full": 1280+2576}[mode]

            print(f"  [Cross-Attention] mode={mode}, gprot_dim={gprot_dim} ...")
            ca_res = run_cross_attention_cv(
                X, y, meta, fold_clusters, sample_to_cluster,
                mode=mode, gprot_dim=gprot_dim
            )
            key_ca = f"crossattn_{suffix}_{mode}"
            results[key_ca] = ca_res
            print(f"    AUC = {ca_res['auc_mean']:.4f} ± {ca_res['auc_std']:.4f}, PR-AUC = {ca_res['overall_pr_auc']:.4f}")

            # MLP (only on ICL-full 650M config)
            if suffix == "gprot_650m_1280d" and mode == "icl_full":
                print(f"  [MLP] input_dim={X.shape[1]} ...")
                mlp_res = run_mlp_cv(X, y, meta, fold_clusters, sample_to_cluster, input_dim=X.shape[1])
                key_mlp = f"mlp_{suffix}_{mode}"
                results[key_mlp] = mlp_res
                print(f"    AUC = {mlp_res['auc_mean']:.4f} ± {mlp_res['auc_std']:.4f}, PR-AUC = {mlp_res['overall_pr_auc']:.4f}")

    # Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Results saved to {OUTPUT_FILE}")

    # Summary comparison table
    print("\n" + "="*70)
    print("  COMPARISON SUMMARY")
    print("="*70)
    print(f"  {'Config':<40} {'AUC':>8} {'PR-AUC':>8}")
    print(f"  {'-'*40} {'-'*8} {'-'*8}")
    orig_baseline = [f"  {k:<40} {v['auc_mean']:>8.4f} {v['overall_pr_auc']:>8.4f}"
                      for k, v in results.items()]
    for line in orig_baseline:
        print(line)

    # Key comparisons
    print("\n  === Key Comparisons ===")
    for gprot_label in ["gprot_8m_320d", "gprot_650m_1280d"]:
        for mode in ["baseline", "icl_full"]:
            svm_k = f"svm_{gprot_label}_{mode}"
            ca_k = f"crossattn_{gprot_label}_{mode}"
            if svm_k in results and ca_k in results:
                delta = results[ca_k]["auc_mean"] - results[svm_k]["auc_mean"]
                print(f"  {gprot_label}/{mode}: CrossAttn-SVM ΔAUC = {delta:+.4f}")

if __name__ == "__main__":
    main()
