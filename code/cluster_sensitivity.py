#!/usr/bin/env python3
"""
Cluster threshold sensitivity analysis using ESM-2 embedding cosine similarity.
Shows how Cluster-CV AUC varies with clustering strictness.
"""
import json, numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, pairwise_distances
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
DATA_DIR = BASE.parent / "data"
OUTPUT_FILE = DATA_DIR / "cluster_sensitivity.json"
N_FOLDS, RS = 5, 42

FAMILY_MAP_GPROT = {
    "GNAQ": "Gq", "GNAI1": "Gi", "GNAI2": "Gi", "GNAI3": "Gi",
    "GNAS": "Gs", "GNA12": "G12_13", "GNA13": "G12_13",
}

print("Loading data...")
with open(DATA_DIR / "gpcr_esm_features_650m.json") as f:
    gpcr_raw = json.load(f)
gpcr_emb = {}  # deduplicated base_id -> 1280-d embedding
for k, v in gpcr_raw.items():
    arr = np.array(v)
    gpcr_emb[k] = arr.mean(axis=0) if arr.ndim == 2 else arr

with open(DATA_DIR / "g_protein_esm_features_650m.json") as f:
    raw = json.load(f)
gprot_feats = {}
for s, info in raw.items():
    fam = FAMILY_MAP_GPROT.get(s, s)
    vec = np.array(info["mean_pooling"])
    gprot_feats[s] = vec; gprot_feats[fam] = vec

with open(DATA_DIR / "icl_features_650m.json") as f:
    icl_data = json.load(f)

df = pd.read_csv(DATA_DIR / "pairing_matrix_raw.csv").dropna(subset=["cluster_id"])

# Build GPCR embedding matrix for clustering
gpcr_ids = sorted(gpcr_emb.keys())
emb_matrix = np.stack([gpcr_emb[gid] for gid in gpcr_ids])
print(f"GPCR embeddings: {emb_matrix.shape}")

# Cosine distance matrix
cos_dist = pairwise_distances(emb_matrix, metric="cosine")
cos_sim = 1 - cos_dist
print(f"Cosine similarity: mean={cos_sim.mean():.4f}, median={np.median(cos_sim):.4f}")

# Cluster at given cosine similarity threshold
def cluster_at_cosine(threshold):
    n = len(gpcr_ids)
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[ra] = rb

    for i in range(n):
        for j in range(i+1, n):
            if cos_sim[i, j] >= threshold and find(i) != find(j):
                union(i, j)

    clusters = defaultdict(list)
    for i in range(n):
        clusters[find(i)].append(gpcr_ids[i])
    return list(clusters.values())

# Feature vectors
def get_gpcr_f(gid):
    f = gpcr_emb.get(gid)
    if f is None and "_" in gid and len(gid.split("_")[0]) <= 2:
        f = gpcr_emb.get(gid.split("_", 1)[1])
    if f is None:
        for k in gpcr_emb:
            if "_" in k and k.split("_", 1)[1] == gid:
                return gpcr_emb[k]
    return f

def get_icl(gid):
    r = icl_data.get(gid)
    if r is None:
        for k in icl_data:
            if "_" in k and k.split("_",1)[1] == gid:
                r = icl_data[k]; break
    e2 = np.array(r.get("ICL2_esm", [])) if r else np.array([])
    e3 = np.array(r.get("ICL3_esm", [])) if r else np.array([])
    if e2.size == 0: e2 = np.zeros(1280)
    if e3.size == 0: e3 = np.zeros(1280)
    sk = ["length","mean_hydro","std_hydro","net_charge","pos_charge_ratio",
          "neg_charge_ratio","hydrophobic_ratio","aromatic_ratio"]
    s2 = np.array([(r.get("ICL2_stats",{}).get(k,0.0) if r else 0.0) for k in sk])
    s3 = np.array([(r.get("ICL3_stats",{}).get(k,0.0) if r else 0.0) for k in sk])
    return e2, s2, e3, s3

X_list, y_list, metas = [], [], []
for _, row in df.iterrows():
    gid, gf = row["gpcr_id"], row["g_protein_family"]
    gpcr_f = get_gpcr_f(gid)
    gprot_f = gprot_feats.get(gf)
    if gprot_f is None: gprot_f = gprot_feats.get(gf.capitalize()) or gprot_feats.get(gf.upper())
    if gpcr_f is None or gprot_f is None: continue
    i2_e, i2_s, i3_e, i3_s = get_icl(gid)
    X_list.append(np.concatenate([gpcr_f, gprot_f, i2_e, i2_s, i3_e, i3_s]))
    y_list.append(int(row["coupling"]))
    metas.append({"gpcr_id": gid})
X, y = np.array(X_list), np.array(y_list)
print(f"Features: {X.shape}, Pos ratio: {y.mean():.4f}")

# Random-CV
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RS)
random_aucs = []
for tr, te in skf.split(X, y):
    sc = StandardScaler()
    X_tr = sc.fit_transform(X[tr]); X_te = sc.transform(X[te])
    svm = SVC(kernel="rbf", C=10.0, class_weight="balanced", probability=True, random_state=RS)
    svm.fit(X_tr, y[tr])
    if len(set(y[te])) >= 2: random_aucs.append(roc_auc_score(y[te], svm.predict_proba(X_te)[:,1]))
random_auc = np.mean(random_aucs)
print(f"Random-CV AUC: {random_auc:.4f} +/- {np.std(random_aucs):.4f}")

# Cluster + CV at each threshold
# Cosine similarity thresholds: ~0.7 is high similarity, ~0.9 is very high
thresholds = [0.99, 0.98, 0.97, 0.96, 0.95, 0.93, 0.90, 0.85, 0.80]
print(f"\n{'CosThr':>8s} {'N_Clust':>10s} {'Singl':>8s} {'CV AUC':>10s} {'Gap':>10s}")
print("-" * 52)

results = []
for thresh in thresholds:
    members = cluster_at_cosine(thresh)
    n_c = len(members)
    n_s = sum(1 for m in members if len(m) == 1)

    g2c = {}
    for ci, mems in enumerate(members):
        for m in mems: g2c[m] = ci

    s_clust = []
    for meta in metas:
        gid = meta["gpcr_id"]
        base = gid.split("_",1)[1] if "_" in gid and len(gid.split("_")[0]) <= 2 else gid
        s_clust.append(g2c.get(gid, g2c.get(base, -1)))

    c_sizes = defaultdict(int)
    for c in s_clust: c_sizes[c] += 1

    fold_cids = [[] for _ in range(N_FOLDS)]
    fold_size = [0] * N_FOLDS
    for cid in sorted(set(s_clust), key=lambda c: c_sizes[c], reverse=True):
        tgt = int(np.argmin(fold_size))
        fold_cids[tgt].append(cid)
        fold_size[tgt] += c_sizes[cid]

    fold_aucs = []
    for fi in range(N_FOLDS):
        tc = set(fold_cids[fi])
        te = [i for i, c in enumerate(s_clust) if c in tc]
        tr = [i for i, c in enumerate(s_clust) if c not in tc]
        if len(te) < 10 or len(set(y[te])) < 2: continue
        sc = StandardScaler()
        X_tr = sc.fit_transform(X[tr]); X_te = sc.transform(X[te])
        svm = SVC(kernel="rbf", C=10.0, class_weight="balanced", probability=True, random_state=RS)
        svm.fit(X_tr, y[tr])
        fold_aucs.append(roc_auc_score(y[te], svm.predict_proba(X_te)[:,1]))

    m_auc = np.mean(fold_aucs) if fold_aucs else float("nan")
    gap = random_auc - m_auc
    results.append({"cosine_threshold": thresh, "n_clusters": n_c, "n_singletons": n_s,
                    "auc": round(m_auc, 4), "auc_std": round(float(np.std(fold_aucs)), 4) if fold_aucs else 0,
                    "gap": round(gap, 4)})
    print(f"  {thresh:>8.3f}  {n_c:>10d}  {n_s:>8d}  {m_auc:>10.4f}  {gap:>8.4f}")

with open(OUTPUT_FILE, "w") as f:
    json.dump({"method": "ESM-2 650M embedding cosine similarity single-linkage clustering",
               "original_method": "3-mer Jaccard (threshold 0.30, 387 clusters)",
               "note": "Cosine similarity on ESM-2 embeddings used as sequence similarity proxy for sensitivity analysis",
               "random_cv_auc": round(random_auc, 4),
               "random_cv_std": round(float(np.std(random_aucs)), 4),
               "thresholds": results}, f, indent=2)
print(f"\nSaved to {OUTPUT_FILE}")
