#!/usr/bin/env python3
"""Compute tiered metrics from CV prediction files for CA, MLP, SVM."""
import json, numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import warnings
warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
DATA_DIR = BASE.parent / "data"
OUTPUT_FILE = DATA_DIR / "tiered_metrics.json"
PAIRING_FILE = DATA_DIR / "pairing_matrix_raw.csv"
RS = 42

MODEL_FILES = {
    "CA": DATA_DIR / "ca_predictions.json",
    "SVM": DATA_DIR / "svm_predictions.json",
    "MLP": DATA_DIR / "mlp_predictions.json",
}
FAMILIES = ["Gq", "Gi", "Gs", "G12_13"]


def load_preds(path):
    with open(path) as f: preds = json.load(f)
    records = []
    for gid, families in preds.items():
        for gfam, info in families.items():
            records.append({"gpcr_id": gid, "g_protein_family": gfam,
                           "label": int(info.get("label", -1)),
                           "prob": float(info.get("prob", float("nan")))})
    return pd.DataFrame.from_records(records)


def main():
    print("=" * 70)
    print("  Tiered Metrics from CV Predictions")
    print("=" * 70)
    np.random.seed(RS)

    # Load pairing matrix for zero-positive GPCR detection
    df = pd.read_csv(PAIRING_FILE).dropna(subset=["cluster_id"])
    gpcr_pos_count = defaultdict(int)
    for _, row in df.iterrows():
        if row["coupling"] == 1:
            gpcr_pos_count[row["gpcr_id"]] += 1

    all_results = {}
    for model_name, pred_file in MODEL_FILES.items():
        if not pred_file.exists():
            print(f"  {model_name}: predictions not found, skipping")
            continue

        df_pred = load_preds(pred_file).dropna(subset=["prob"])
        print(f"\n  {model_name}: {len(df_pred)} pairs")

        # Determine tiers for each row in predictions
        tier1_mask = np.ones(len(df_pred), dtype=bool)
        tier2_mask = np.array([
            gpcr_pos_count.get(row["gpcr_id"], 0) > 0
            for _, row in df_pred.iterrows()
        ])
        # Tier 3: exclude pairs where gpcr has zero positive labels (inferred_negative)
        # This uses the same logic as label_audit.py
        tier3_mask = tier2_mask  # same as tier2 in this formulation

        model_results = {}
        for tier_name, mask in [("Tier1_full", tier1_mask),
                                 ("Tier2_remove_zero_positive", tier2_mask)]:
            sub = df_pred[mask]
            if len(sub) < 20: continue
            y, p = sub["label"].values, sub["prob"].values
            n_pos = int(y.sum())
            model_results[tier_name] = {
                "n": int(len(y)), "n_pos": n_pos,
                "pos_ratio": float(n_pos / len(y)),
                "auc": float(roc_auc_score(y, p)) if len(set(y)) >= 2 else None,
                "prauc": float(average_precision_score(y, p)),
                "brier": float(brier_score_loss(y, p)),
            }
            print(f"    {tier_name}: N={len(y)}, pos_ratio={n_pos/len(y):.3f}, "
                  f"AUC={model_results[tier_name]['auc']:.4f}, "
                  f"PRAUC={model_results[tier_name]['prauc']:.4f}")

        # Also per-family tier2 metrics for G12/13 focus
        per_fam_tier = {}
        for fam in FAMILIES:
            fam_df = df_pred[(df_pred["g_protein_family"] == fam)]
            # Tier 2 on family subset
            fam_t2 = fam_df[[gpcr_pos_count.get(r["gpcr_id"], 0) > 0
                            for _, r in fam_df.iterrows()]]
            if len(fam_t2) >= 10 and len(set(fam_t2["label"])) >= 2:
                yf, pf = fam_t2["label"].values, fam_t2["prob"].values
                per_fam_tier[fam] = {
                    "n": int(len(yf)),
                    "auc": float(roc_auc_score(yf, pf)),
                    "prauc": float(average_precision_score(yf, pf)),
                }

        model_results["per_family_tier2"] = per_fam_tier
        all_results[model_name] = model_results

        # Summary line
        t1 = model_results.get("Tier1_full", {})
        t2 = model_results.get("Tier2_remove_zero_positive", {})
        delta = (t1.get("auc", 0) or 0) - (t2.get("auc", 0) or 0)
        print(f"    Delta AUC (T1-T2): {delta:.4f}")

    # References from minimal family-conditioned MLP
    min_mlp_file = DATA_DIR / "minimal_family_classifier_results.json"
    min_mlp_ref = None
    if min_mlp_file.exists():
        with open(min_mlp_file) as f:
            min_mlp_ref = json.load(f)["results"]["mlp"]
        print(f"\n  Minimal family-conditioned MLP (reference): AUC={min_mlp_ref['auc_mean']:.4f}")

    out = {
        "description": "Tiered evaluation metrics from cluster-CV predictions",
        "tiers": {
            "Tier1": "Full dataset (1647 pairs)",
            "Tier2": "Remove zero-positive GPCRs (GPCRs with no positive annotations)",
        },
        "models": all_results,
        "minimal_mlp_reference": min_mlp_ref,
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
