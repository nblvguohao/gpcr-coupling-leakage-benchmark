#!/usr/bin/env python3
"""
Generate MLP 650M + ICL predictions under cluster-aware 5-fold CV.
Mirrors generate_cv_predictions.py exactly, swapping CA → MLP.
"""
import json, numpy as np, pandas as pd, torch, torch.nn as nn
from pathlib import Path
from collections import defaultdict
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
OUTPUT_FILE = DATA_DIR / "mlp_predictions.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_FOLDS, RS = 5, 42
FAMILY_MAP_GPROT = {
    "GNAQ": "Gq", "GNAI1": "Gi", "GNAI2": "Gi", "GNAI3": "Gi",
    "GNAS": "Gs", "GNA12": "G12_13", "GNA13": "G12_13",
}


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
        feats[subtype] = vec; feats[family] = vec
    return feats


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


def build_vectors(df, gpcr_feats, gprot_feats, icl_data):
    X_list, y_list, metas = [], [], []
    for _, row in df.iterrows():
        gid = row["gpcr_id"]
        gf = row["g_protein_family"]
        gpcr_f = get_gpcr_feat(gpcr_feats, gid)
        gprot_f = gprot_feats.get(gf)
        if gprot_f is None:
            gprot_f = gprot_feats.get(gf.capitalize()) or gprot_feats.get(gf.upper())
        if gpcr_f is None or gprot_f is None: continue
        i2_e, i2_s, i3_e, i3_s = get_icl_vector(icl_data, gid, 1280)
        parts = [gpcr_f, gprot_f, i2_e, i2_s, i3_e, i3_s]
        X_list.append(np.concatenate(parts))
        y_list.append(int(row["coupling"]))
        metas.append({"gpcr_id": gid, "g_protein_family": gf,
                      "cluster_id": int(row["cluster_id"]), "row_idx": _})
    return np.array(X_list), np.array(y_list), metas


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


class MLPNet(nn.Module):
    """3-layer MLP exactly matching paper architecture."""
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
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X); self.y = torch.FloatTensor(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


def train_epoch(model, loader, optim, crit):
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        if len(y) < 2: continue  # BatchNorm needs batch_size >= 2
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
def evaluate(model, loader):
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        x = x.to(DEVICE)
        probs.append(torch.sigmoid(model(x)).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def main():
    print("=" * 70)
    print("  Generate MLP 650M+ICL Cluster-CV Predictions")
    print(f"  Device: {DEVICE}")
    print("=" * 70)
    torch.manual_seed(RS); np.random.seed(RS)
    if torch.cuda.is_available(): torch.cuda.manual_seed(RS)

    gpcr_feats = load_gpcr_features()
    gprot_feats = load_gprot_features()
    icl_data = load_icl_features()
    df = pd.read_csv(PAIRING_MATRIX_FILE)
    with open(CLUSTERS_FILE) as f:
        cluster_list = json.load(f)["clusters"]

    print(f"  GPCR features: {len(gpcr_feats)}, Pairs: {len(df)}")
    X, y, metas = build_vectors(df, gpcr_feats, gprot_feats, icl_data)
    print(f"  Feature dim: {X.shape[1]}, Pos ratio: {y.mean():.4f}")
    s2c, fold_cids = get_cluster_folds(y, metas, cluster_list)

    mlp_preds = {}
    mlp_fold_aucs = []
    for fi in range(N_FOLDS):
        test_cids = set(fold_cids[fi])
        te_idx = [i for i in range(len(y)) if s2c[i] in test_cids]
        tr_idx = [i for i in range(len(y)) if s2c[i] not in test_cids]
        X_tr, X_te = X[tr_idx], X[te_idx]; y_tr, y_te = y[tr_idx], y[te_idx]
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr); X_te_s = scaler.transform(X_te)
        tr_ds = VecDataset(X_tr_s, y_tr); te_ds = VecDataset(X_te_s, y_te)
        tr_ld = DataLoader(tr_ds, batch_size=32, shuffle=True)
        te_ld = DataLoader(te_ds, batch_size=32)
        model = MLPNet(input_dim=X.shape[1]).to(DEVICE)
        pw = len(y_tr) / max(1, y_tr.sum())
        optim = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        crit = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([pw]).to(DEVICE))
        best_auc, best_p, patience = -1.0, None, 0
        for ep in range(200):
            train_epoch(model, tr_ld, optim, crit)
            p, lbl = evaluate(model, te_ld)
            auc_val = roc_auc_score(lbl, p) if len(set(lbl)) >= 2 else -1
            if auc_val > best_auc:
                best_auc, best_p, patience = auc_val, p, 0
            else:
                patience += 1
            if patience >= 20: break
        mlp_fold_aucs.append(best_auc)
        print(f"  Fold {fi+1}: MLP AUC = {best_auc:.4f}")
        for i, idx in enumerate(te_idx):
            gid = metas[idx]["gpcr_id"]
            gfam = metas[idx]["g_protein_family"]
            if gid not in mlp_preds: mlp_preds[gid] = {}
            mlp_preds[gid][gfam] = {"label": int(y_te[i]), "prob": float(best_p[i])}

    mean_auc = np.mean(mlp_fold_aucs)
    print(f"\n  MLP Cluster-CV AUC: {mean_auc:.4f} +/- {np.std(mlp_fold_aucs):.4f}")
    print(f"  Fold AUCs: {[f'{a:.4f}' for a in mlp_fold_aucs]}")
    with open(OUTPUT_FILE, "w") as f:
        json.dump(mlp_preds, f, indent=2)
    print(f"  Saved: {OUTPUT_FILE} ({len(mlp_preds)} GPCRs)")


if __name__ == "__main__":
    main()
