#!/usr/bin/env python3
"""
Dimension alignment control experiments (Point 4 of revision).

Four minimal control groups to test whether the ICL-global feature dimension
mismatch degradation is due to feature-scale/representation-space compatibility
or a genuine dimension alignment requirement.

Controls (using SVM RBF, cluster-aware 5-fold CV):
  (C1) 1280-d ICL -> PCA to 320-d, GPCR 1280 -> PCA to 320-d, concat at 320-d
  (C2) 320-d ICL -> learned linear projection to 1280-d, concat at 1280-d
  (C3) Block-wise z-score normalization before concatenation
  (C4) Zero-padding 320-d ICL to 1280-d, concat at 1280-d

Note: 8M GPCR embeddings (320-d) not available on disk. For C1, GPCR 1280-d
global embeddings are PCA-projected to 320-d as a proxy (ESM-2 embeddings are
hierarchical — lower PCA dims approximate smaller-model spaces).
"""
import json, numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
DATA_DIR = BASE.parent / "data"
OUTPUT_FILE = DATA_DIR / "dimension_alignment_controls.json"

GPCR_650M_FILE = DATA_DIR / "gpcr_esm_features_650m.json"
ICL_320_FILE = DATA_DIR / "icl_features.json"       # 320-d (8M)
ICL_1280_FILE = DATA_DIR / "icl_features_650m.json"  # 1280-d (650M)
GPROT_FILE = DATA_DIR / "g_protein_esm_features_650m.json"
PAIRING_FILE = DATA_DIR / "pairing_matrix_raw.csv"
CLUSTERS_FILE = DATA_DIR / "sequence_clusters.json"

N_FOLDS, RS = 5, 42
FAMILY_MAP = {"GNAQ":"Gq","GNAI1":"Gi","GNAI2":"Gi","GNAI3":"Gi",
              "GNAS":"Gs","GNA12":"G12_13","GNA13":"G12_13"}


def load_json(path):
    if not path.exists(): return {}
    with open(path) as f: return json.load(f)


def load_gprot_embed():
    raw = load_json(GPROT_FILE)
    feats = {}
    for s, info in raw.items():
        fam = FAMILY_MAP.get(s, s)
        vec = np.array(info["mean_pooling"])
        feats[fam] = vec
    return feats


def get_gpcr(gpcr_feats, gid):
    f = gpcr_feats.get(gid)
    if f is None and "_" in gid and len(gid.split("_")[0]) <= 2:
        f = gpcr_feats.get(gid.split("_", 1)[1])
    if f is None:
        for k in gpcr_feats:
            if "_" in k and k.split("_", 1)[1] == gid: return gpcr_feats[k]
    return f


def get_icl(icl_data, gid, dim):
    rec = icl_data.get(gid)
    if rec is None:
        for k in icl_data:
            if "_" in k and k.split("_",1)[1] == gid: rec = icl_data[k]; break
    e2 = np.array(rec.get("ICL2_esm", [])) if rec else np.array([])
    e3 = np.array(rec.get("ICL3_esm", [])) if rec else np.array([])
    if e2.size == 0: e2 = np.zeros(dim)
    if e3.size == 0: e3 = np.zeros(dim)
    sk = ["length","mean_hydro","std_hydro","net_charge","pos_charge_ratio",
          "neg_charge_ratio","hydrophobic_ratio","aromatic_ratio"]
    s2 = np.array([(rec.get("ICL2_stats",{}).get(k,0.0) if rec else 0.0) for k in sk])
    s3 = np.array([(rec.get("ICL3_stats",{}).get(k,0.0) if rec else 0.0) for k in sk])
    return e2, s2, e3, s3


def build_df(gpcr_feats, gprot_feats, icl_data, df, gpcr_dim, icl_dim):
    Xl, yl, metas = [], [], []
    for _, row in df.iterrows():
        gid, gf = row["gpcr_id"], row["g_protein_family"]
        gpcr_f = get_gpcr(gpcr_feats, gid)
        gprot_f = gprot_feats.get(gf)
        if gprot_f is None: gprot_f = gprot_feats.get(gf.capitalize()) or gprot_feats.get(gf.upper())
        if gpcr_f is None or gprot_f is None: continue
        i2e, i2s, i3e, i3s = get_icl(icl_data, gid, icl_dim)
        Xl.append(np.concatenate([gpcr_f, gprot_f, i2e, i2s, i3e, i3s]))
        yl.append(int(row["coupling"]))
        metas.append({"gpcr_id": gid, "cluster_id": int(row["cluster_id"])})
    return np.array(Xl), np.array(yl), metas


def get_cluster_folds(y, metas, cluster_list):
    n = len(y)
    s2c = {i: metas[i]["cluster_id"] for i in range(n)}
    c_sizes = defaultdict(int)
    for i in range(n): c_sizes[s2c[i]] += 1
    fold_cids = [[] for _ in range(N_FOLDS)]
    fold_size = [0.0] * N_FOLDS
    for c in sorted(cluster_list, key=lambda c: len(c["members"]), reverse=True):
        cid = c["cluster_id"]
        if c_sizes.get(cid, 0) == 0: continue
        t = int(np.argmin(fold_size))
        fold_cids[t].append(cid); fold_size[t] += c_sizes[cid]
    return s2c, fold_cids


def run_svm_cv(X, y, metas, cluster_list, desc=""):
    s2c, fold_cids = get_cluster_folds(y, metas, cluster_list)
    aucs = []
    for fi in range(N_FOLDS):
        tc = set(fold_cids[fi])
        te = [i for i in range(len(y)) if s2c[i] in tc]
        tr = [i for i in range(len(y)) if s2c[i] not in tc]
        sc = StandardScaler()
        X_tr = sc.fit_transform(X[tr]); X_te = sc.transform(X[te])
        svm = SVC(kernel="rbf", C=10.0, class_weight="balanced",
                  probability=True, random_state=RS)
        svm.fit(X_tr, y[tr])
        if len(set(y[te])) >= 2:
            aucs.append(roc_auc_score(y[te], svm.predict_proba(X_te)[:,1]))
    return float(np.mean(aucs)), float(np.std(aucs)), aucs


def main():
    print("=" * 70)
    print("  Dimension Alignment Control Experiments")
    print("=" * 70)
    np.random.seed(RS)

    # Load data
    gpcr_650m_raw = load_json(GPCR_650M_FILE)
    gpcr_650m = {k: np.array(v) for k, v in gpcr_650m_raw.items()}
    icl_320 = load_json(ICL_320_FILE)
    icl_1280 = load_json(ICL_1280_FILE)
    gprot_feats = load_gprot_embed()
    df = pd.read_csv(PAIRING_FILE).dropna(subset=["cluster_id"])
    with open(CLUSTERS_FILE) as f: cluster_list = json.load(f)["clusters"]

    print(f"  650M GPCR: {len(gpcr_650m)}")
    print(f"  320-d ICL: {len(icl_320)}, 1280-d ICL: {len(icl_1280)}")
    print(f"  Pairs: {len(df)}")

    # Build feature matrices
    X_1280, y_1280, m_1280 = build_df(gpcr_650m, gprot_feats, icl_1280, df, 1280, 1280)
    X_mis, y_mis, m_mis = build_df(gpcr_650m, gprot_feats, icl_320, df, 1280, 320)
    print(f"  1280+1280 matched: {X_1280.shape}")
    print(f"  1280+320 mismatched: {X_mis.shape}")

    # Reference baselines
    ref_1280_auc, ref_1280_std, _ = run_svm_cv(X_1280, y_1280, m_1280, cluster_list, "1280 matched")
    ref_mis_auc, ref_mis_std, _ = run_svm_cv(X_mis, y_mis, m_mis, cluster_list, "mismatched")
    # Paper-reported 320 matched: SVM 0.8301 (from main.tex Table tab:dimension)
    ref_320_paper = 0.8301
    print(f"\n  SVM 1280+1280 matched:     {ref_1280_auc:.4f} +/- {ref_1280_std:.4f}")
    print(f"  SVM 1280+320 mismatched:   {ref_mis_auc:.4f} +/- {ref_mis_std:.4f}")
    print(f"  SVM 320+320 matched (paper): {ref_320_paper:.4f}")
    print(f"  Mismatch penalty: {ref_1280_auc - ref_mis_auc:.4f}")

    results = {
        "reference": {
            "1280_matched": {"auc": ref_1280_auc, "std": ref_1280_std},
            "mismatched_1280_320": {"auc": ref_mis_auc, "std": ref_mis_std},
            "320_matched_paper": ref_320_paper,
        },
        "controls": {},
    }

    # Extract ICL embedding blocks from feature matrices
    # X_1280 structure: [gpcr(1280) | gprot(1280) | icl2_e(1280) | icl2_s(8) | icl3_e(1280) | icl3_s(8)] = 5136
    # X_mis structure:  [gpcr(1280) | gprot(1280) | icl2_e(320)  | icl2_s(8) | icl3_e(320)  | icl3_s(8)] = 3216

    gpcr_global_1280 = X_1280[:, :1280]
    gprot_1280 = X_1280[:, 1280:2560]
    icl2_e_1280 = X_1280[:, 2560:2560+1280]
    icl2_s = X_1280[:, 2560+1280:2560+1280+8]
    icl3_e_1280 = X_1280[:, 2560+1280+8:2560+1280+8+1280]
    icl3_s = X_1280[:, 2560+1280+8+1280:]

    gpcr_global_1280_mis = X_mis[:, :1280]
    icl2_e_320 = X_mis[:, 2560:2560+320]
    icl2_s_mis = X_mis[:, 2560+320:2560+320+8]
    icl3_e_320 = X_mis[:, 2560+320+8:2560+320+8+320]
    icl3_s_mis = X_mis[:, 2560+320+8+320:]

    # ========================================================================
    # C1: 1280-d ICL -> PCA to 320-d; GPCR 1280 -> PCA to 320-d; concat at 320-d
    # ========================================================================
    print("\n--- C1: PCA projection to 320-d for both GPCR global and ICL ---")
    pca_gpcr = PCA(n_components=320, random_state=RS)
    pca_icl2 = PCA(n_components=320, random_state=RS)
    pca_icl3 = PCA(n_components=320, random_state=RS)

    gpcr_320_proj = pca_gpcr.fit_transform(gpcr_global_1280)
    icl2_320_proj = pca_icl2.fit_transform(icl2_e_1280)
    icl3_320_proj = pca_icl3.fit_transform(icl3_e_1280)

    print(f"  PCA GPCR 1280->320: explained var = {pca_gpcr.explained_variance_ratio_.sum():.4f}")
    print(f"  PCA ICL2 1280->320: explained var = {pca_icl2.explained_variance_ratio_.sum():.4f}")
    print(f"  PCA ICL3 1280->320: explained var = {pca_icl3.explained_variance_ratio_.sum():.4f}")

    X_c1 = np.concatenate([
        gpcr_320_proj, gprot_1280,  # GPCR 320 + Gprot 1280
        icl2_320_proj, icl2_s, icl3_320_proj, icl3_s,
    ], axis=1)
    c1_auc, c1_std, c1_folds = run_svm_cv(X_c1, y_1280, m_1280, cluster_list, "C1")
    results["controls"]["C1_PCA_all_to_320"] = {
        "description": "GPCR 1280->320 PCA + ICL 1280->320 PCA + Gprot 1280 (all projected to 320-d except Gprot)",
        "auc": c1_auc, "std": c1_std, "fold_aucs": [float(a) for a in c1_folds],
        "gpcr_explained_var": float(pca_gpcr.explained_variance_ratio_.sum()),
        "icl2_explained_var": float(pca_icl2.explained_variance_ratio_.sum()),
        "icl3_explained_var": float(pca_icl3.explained_variance_ratio_.sum()),
    }
    print(f"  C1 AUC: {c1_auc:.4f} +/- {c1_std:.4f}")
    print(f"  delta vs 1280 matched: {c1_auc - ref_1280_auc:+.4f}")
    print(f"  delta vs 320 matched (paper {ref_320_paper}): {c1_auc - ref_320_paper:+.4f}")

    # ========================================================================
    # C2: 320-d ICL -> learned linear projection to 1280-d + 1280-d global
    # ========================================================================
    print("\n--- C2: Learned linear projection 320-d ICL -> 1280-d ---")
    # Train LR: predict 1280-d ICL from 320-d ICL (same GPCRs have both)
    # Need to match GPCRs across 1280 and 320 ICL datasets
    gids_1280 = [m["gpcr_id"] for m in m_1280]
    gids_mis = [m["gpcr_id"] for m in m_mis]
    gid_to_idx_1280 = {g: i for i, g in enumerate(gids_1280)}
    gid_to_idx_mis = {g: i for i, g in enumerate(gids_mis)}
    common = sorted(set(gid_to_idx_1280) & set(gid_to_idx_mis))
    idx_1280_sub = [gid_to_idx_1280[g] for g in common]
    idx_mis_sub = [gid_to_idx_mis[g] for g in common]
    print(f"  Common GPCRs: {len(common)}")

    icl2_320_sub = icl2_e_320[idx_mis_sub]
    icl3_320_sub = icl3_e_320[idx_mis_sub]
    icl2_1280_sub = icl2_e_1280[idx_1280_sub]
    icl3_1280_sub = icl3_e_1280[idx_1280_sub]

    lr2 = LinearRegression(); lr3 = LinearRegression()
    lr2.fit(icl2_320_sub, icl2_1280_sub); lr3.fit(icl3_320_sub, icl3_1280_sub)
    print(f"  LR ICL2 320->1280: R^2 = {lr2.score(icl2_320_sub, icl2_1280_sub):.4f}")
    print(f"  LR ICL3 320->1280: R^2 = {lr3.score(icl3_320_sub, icl3_1280_sub):.4f}")

    # Apply to all mismatched data
    icl2_1280_proj = lr2.predict(icl2_e_320)
    icl3_1280_proj = lr3.predict(icl3_e_320)

    X_c2 = np.concatenate([
        gpcr_global_1280_mis, gprot_1280,
        icl2_1280_proj, icl2_s_mis,
        icl3_1280_proj, icl3_s_mis,
    ], axis=1)
    c2_auc, c2_std, c2_folds = run_svm_cv(X_c2, y_mis, m_mis, cluster_list, "C2")
    results["controls"]["C2_learned_proj_320icl_to_1280"] = {
        "description": "320-d ICL projected to 1280-d via linear regression, combined with 1280-d global",
        "auc": c2_auc, "std": c2_std, "fold_aucs": [float(a) for a in c2_folds],
        "icl2_r2": float(lr2.score(icl2_320_sub, icl2_1280_sub)),
        "icl3_r2": float(lr3.score(icl3_320_sub, icl3_1280_sub)),
    }
    print(f"  C2 AUC: {c2_auc:.4f} +/- {c2_std:.4f}")
    print(f"  delta vs mismatched: {c2_auc - ref_mis_auc:+.4f}")
    print(f"  delta vs 1280 matched: {c2_auc - ref_1280_auc:+.4f}")

    # ========================================================================
    # C3: Block-wise z-score normalization
    # ========================================================================
    print("\n--- C3: Block-wise z-score normalization ---")
    def zscore_block(block):
        mean = block.mean(axis=0)
        std = block.std(axis=0)
        std[std == 0] = 1.0
        return (block - mean) / std

    X_c3 = np.concatenate([
        zscore_block(gpcr_global_1280_mis),
        gprot_1280,  # keep G protein block as-is
        zscore_block(icl2_e_320), icl2_s_mis,
        zscore_block(icl3_e_320), icl3_s_mis,
    ], axis=1)
    c3_auc, c3_std, c3_folds = run_svm_cv(X_c3, y_mis, m_mis, cluster_list, "C3")
    results["controls"]["C3_block_zscore"] = {
        "description": "GPCR global and ICL blocks z-score normalized separately before concat (mismatched dims)",
        "auc": c3_auc, "std": c3_std, "fold_aucs": [float(a) for a in c3_folds],
    }
    print(f"  C3 AUC: {c3_auc:.4f} +/- {c3_std:.4f}")
    print(f"  delta vs mismatched (no zscore): {c3_auc - ref_mis_auc:+.4f}")

    # ========================================================================
    # C4: Zero-padding 320-d ICL -> 1280-d
    # ========================================================================
    print("\n--- C4: Zero-padding 320-d ICL -> 1280-d ---")
    icl2_e_padded = np.pad(icl2_e_320, ((0,0),(0,960)), mode='constant')
    icl3_e_padded = np.pad(icl3_e_320, ((0,0),(0,960)), mode='constant')

    X_c4 = np.concatenate([
        gpcr_global_1280_mis, gprot_1280,
        icl2_e_padded, icl2_s_mis,
        icl3_e_padded, icl3_s_mis,
    ], axis=1)
    c4_auc, c4_std, c4_folds = run_svm_cv(X_c4, y_mis, m_mis, cluster_list, "C4")
    results["controls"]["C4_zero_pad_320icl_to_1280"] = {
        "description": "320-d ICL zero-padded to 1280-d, combined with 1280-d global",
        "auc": c4_auc, "std": c4_std, "fold_aucs": [float(a) for a in c4_folds],
    }
    print(f"  C4 AUC: {c4_auc:.4f} +/- {c4_std:.4f}")
    print(f"  delta vs mismatched: {c4_auc - ref_mis_auc:+.4f}")

    # ========================================================================
    # Summary
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"  SUMMARY: Dimension Alignment Controls")
    print(f"{'='*70}")
    print(f"  {'Configuration':<55s} {'AUC':>8s} {'Std':>8s}")
    print(f"  {'-'*77}")
    print(f"  {'1280+1280 matched (reference)':<55s} {ref_1280_auc:>8.4f} {ref_1280_std:>8.4f}")
    print(f"  {'1280+320 mismatched (reference)':<55s} {ref_mis_auc:>8.4f} {ref_mis_std:>8.4f}")
    print(f"  {'320+320 matched (paper reference)':<55s} {ref_320_paper:>8.4f} {'—':>8s}")
    print(f"  {'C1: PCA all->320 (GPCR 1280->320 + ICL 1280->320)':<55s} {c1_auc:>8.4f} {c1_std:>8.4f}")
    print(f"  {'C2: LR ICL 320->1280 + GPCR 1280':<55s} {c2_auc:>8.4f} {c2_std:>8.4f}")
    print(f"  {'C3: Block-wise z-score (mismatched dims)':<55s} {c3_auc:>8.4f} {c3_std:>8.4f}")
    print(f"  {'C4: Zero-pad ICL 320->1280 + GPCR 1280':<55s} {c4_auc:>8.4f} {c4_std:>8.4f}")

    print(f"\n  Interpretation:")
    mm_penalty = ref_1280_auc - ref_mis_auc
    print(f"    Mismatch penalty (1280-1280 vs 1280-320): {mm_penalty:+.4f} AUC")
    for label, delta in [
        ("C1 PCA->320", c1_auc - ref_320_paper),
        ("C2 LR->1280", c2_auc - ref_1280_auc),
        ("C3 zscore ", c3_auc - ref_mis_auc),
        ("C4 pad    ", c4_auc - ref_mis_auc),
    ]:
        if abs(delta) < 0.005:
            interp = "RECOVERS — mismatch is feature-scale issue"
        elif delta > 0.002:
            interp = "partially recovers"
        else:
            interp = "does NOT recover — dimension alignment requirement supported"
        print(f"    {label}: deltaAUC={delta:+.4f} -> {interp}")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
