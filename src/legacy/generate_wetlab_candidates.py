#!/usr/bin/env python3
"""
Train a full-data ICL-full SVM on all 1,639 pairs and select wet-lab candidates.

Outputs:
  - paired_dataset/svm_predictions_all_pairs.json
    { gpcr_id: { g_protein_family: { "label": 0/1, "prob": float, ... } } }
  - paired_dataset/wetlab_candidates.json
    { "high_conf_positive": [...], "medium_conf": [...],
      "high_conf_negative": [...], "disputed": [...] }
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
DATA_DIR = BASE / "paired_dataset"
GPCR_FEATURES_FILE = BASE / "server_sync" / "extended_data" / "features" / "esm_features_100samples.json"
G_PROTEIN_FEATURES_FILE = DATA_DIR / "g_protein_esm_features.json"
ICL_FEATURES_FILE = DATA_DIR / "icl_features.json"
ALPHA_FEATURES_FILE = DATA_DIR / "alphafold_icl_features.json"
PAIRING_MATRIX_FILE = DATA_DIR / "pairing_matrix_raw.csv"
PREDICTIONS_FILE = DATA_DIR / "svm_predictions_all_pairs.json"
CANDIDATES_FILE = DATA_DIR / "wetlab_candidates.json"


# ---------------------------------------------------------------------------
# Feature loading (same as paired_cross_validation_enhanced.py)
# ---------------------------------------------------------------------------

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


def get_icl_vector(icl_data, gid, gpcr_feat_dim=320):
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
        "tm_mean_plddt",
        "global_plddt_mean", "global_plddt_std",
        "high_confidence_ratio_70", "high_confidence_ratio_90",
        "sasa_mean", "sasa_buried_ratio",
        "contact_density", "mean_contacts_per_residue",
        # Geometric features
        "tm5_tm6_cyto_ca_distance", "icl2_end_to_end_ca_distance",
        "icl3_end_to_end_ca_distance", "tm5_tm6_cyto_dihedral_angle",
        "icl2_aromatic_centroid_depth", "interface_patch_sasa",
        "interface_patch_sasa_ratio", "icl2_helix_ratio", "icl2_sheet_ratio",
        "icl2_coil_ratio", "icl3_helix_ratio", "icl3_sheet_ratio",
        "icl3_coil_ratio",
        # PAE features
        "icl2_mean_pae", "icl2_intra_pae",
        "icl3_mean_pae", "icl3_intra_pae",
        "icl2_tm5_pae", "icl2_tm6_pae",
        "icl3_tm5_pae", "icl3_tm6_pae",
    ]
    if rec is None:
        return np.zeros(len(keys))
    return np.array([rec.get(k, 0.0) for k in keys])


def build_vectors(df, gpcr_feats, gprot_feats, icl_data, alpha_data, mode="icl_full"):
    X_list, y_list, meta = [], [], []
    missing = defaultdict(int)
    for _, row in df.iterrows():
        gid = row["gpcr_id"]
        gpcr_feat = get_gpcr_feat(gpcr_feats, gid)
        gfam = row["g_protein_family"]
        gprot_feat = gprot_feats.get(gfam)
        if gprot_feat is None:
            gprot_feat = gprot_feats.get(gfam.capitalize()) or gprot_feats.get(gfam.upper())
        if gpcr_feat is None or gprot_feat is None:
            missing[f"missing_{gid}_{gfam}"] += 1
            continue

        base_vec = np.concatenate([gpcr_feat, gprot_feat])
        vec_parts = [base_vec]

        if mode in ("icl_full", "alpha"):
            icl2_esm, icl2_stat, icl3_esm, icl3_stat = get_icl_vector(icl_data, gid, len(gpcr_feat))
            vec_parts.append(np.concatenate([icl2_esm, icl2_stat, icl3_esm, icl3_stat]))
        if mode == "alpha":
            vec_parts.append(get_alpha_vector(alpha_data, gid))

        X_list.append(np.concatenate(vec_parts))
        y_list.append(int(row["coupling"]))
        meta.append({
            "gpcr_id": gid,
            "g_protein_family": gfam,
            "cluster_id": int(row["cluster_id"]),
            "coupling": int(row["coupling"]),
        })
    return np.array(X_list), np.array(y_list), meta


def main():
    print("=" * 70)
    print("  Full-Data SVM Training + Wet-Lab Candidate Selection")
    print("=" * 70)

    df = pd.read_csv(PAIRING_MATRIX_FILE)
    gpcr_feats, gprot_feats = load_features()
    icl_data = load_icl_features()
    alpha_data = load_alpha_features()

    # Use ICL-full (best SVM configuration)
    X, y, meta = build_vectors(df, gpcr_feats, gprot_feats, icl_data, alpha_data, mode="icl_full")
    print(f"[INFO] Training set: {len(y)} samples, {X.shape[1]} features")

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    svm = SVC(kernel="rbf", C=10.0, class_weight="balanced", probability=True, random_state=42)
    svm.fit(Xs, y)
    probs = svm.predict_proba(Xs)[:, 1]

    # Build predictions per GPCR
    predictions = defaultdict(dict)
    for i, m in enumerate(meta):
        predictions[m["gpcr_id"]][m["g_protein_family"]] = {
            "label": int(m["coupling"]),
            "prob": round(float(probs[i]), 4),
        }

    with open(PREDICTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(dict(predictions), f, indent=2, ensure_ascii=False)
    print(f"[OK] Saved predictions to {PREDICTIONS_FILE}")

    # Candidate selection
    records = []
    for gid, fams in predictions.items():
        for gfam, info in fams.items():
            records.append({
                "gpcr_id": gid,
                "g_protein_family": gfam,
                "label": info["label"],
                "prob": info["prob"],
            })

    df_pred = pd.DataFrame(records)

    # Note: SVM is very confident on this dataset. Distribution is bimodal with
    # almost no samples in 0.30-0.70 range. We adapt criteria accordingly.

    # 1) High-confidence positive
    #    - Confirmed positives (label==1, prob>0.85) as positive controls
    #    - Best false positives (label==0, highest prob) as novel predictions
    high_conf_positive = []
    confirmed_pos = df_pred[(df_pred["label"] == 1) & (df_pred["prob"] > 0.85)].sort_values("prob", ascending=False)
    novel_pos = df_pred[(df_pred["label"] == 0)].sort_values("prob", ascending=False)
    high_conf_positive.extend(confirmed_pos.head(2).to_dict("records"))
    high_conf_positive.extend(novel_pos.head(1).to_dict("records"))

    # 2) Medium-confidence: select those closest to prob=0.50 (true boundary cases)
    df_pred["dist_to_0_5"] = (df_pred["prob"] - 0.5).abs()
    medium_conf = df_pred.nsmallest(3, "dist_to_0_5").sort_values("prob", ascending=False).to_dict("records")

    # 3) High-confidence negative: prob < 0.15, label==0 (strong non-coupling prediction)
    high_conf_negative = df_pred[(df_pred["prob"] < 0.15) & (df_pred["label"] == 0)].sort_values("prob", ascending=True).head(2).to_dict("records")

    # 4) Disputed: most confident misclassified cases
    #    - False negative: label==1 with lowest prob
    #    - False positive: label==0 with highest prob
    false_neg = df_pred[(df_pred["label"] == 1)].nsmallest(1, "prob")
    false_pos = df_pred[(df_pred["label"] == 0)].nlargest(1, "prob")
    disputed = []
    if not false_neg.empty:
        disputed.extend(false_neg.to_dict("records"))
    if not false_pos.empty:
        disputed.extend(false_pos.to_dict("records"))

    candidates = {
        "high_conf_positive": high_conf_positive,
        "medium_conf": medium_conf,
        "high_conf_negative": high_conf_negative,
        "disputed": disputed,
    }

    # Deduplicate by (gpcr_id, family)
    for key in candidates:
        seen = set()
        unique = []
        for item in candidates[key]:
            k = (item["gpcr_id"], item["g_protein_family"])
            if k not in seen:
                seen.add(k)
                unique.append(item)
        candidates[key] = unique

    with open(CANDIDATES_FILE, "w", encoding="utf-8") as f:
        json.dump(candidates, f, indent=2, ensure_ascii=False)
    print(f"[OK] Saved candidates to {CANDIDATES_FILE}")

    print("\n--- Candidate Summary ---")
    for cat, items in candidates.items():
        print(f"{cat}: {len(items)} candidates")
        for item in items:
            print(f"  {item['gpcr_id']} ({item['g_protein_family']}): prob={item['prob']}, label={item['label']}")


if __name__ == "__main__":
    main()
