"""
Command-line interface for gpcr-coupling tool.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        prog="gpcr-coupling",
        description="Paired GPCR-G protein coupling prediction tool",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ---- predict ----
    predict_parser = subparsers.add_parser(
        "predict", help="Predict coupling for a GPCR-G protein pair"
    )
    predict_parser.add_argument(
        "--gpcr", required=True, help="Path to GPCR FASTA file or feature .npy file"
    )
    predict_parser.add_argument(
        "--gprotein", required=True,
        help="G protein family (Gq/Gi/Gs/G12_13) or path to feature .npy file"
    )
    predict_parser.add_argument(
        "--model-dir", default=None, help="Path to pre-trained model directory"
    )
    predict_parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda"],
        help="Device for inference"
    )
    predict_parser.add_argument(
        "--output", "-o", default=None, help="Output JSON file path"
    )
    predict_parser.add_argument(
        "--all-families", action="store_true",
        help="Predict coupling to all four families"
    )

    # ---- extract-features ----
    extract_parser = subparsers.add_parser(
        "extract-features", help="Extract ESM-2 and ICL features from sequences"
    )
    extract_parser.add_argument(
        "--input", "-i", required=True, help="Input FASTA file path"
    )
    extract_parser.add_argument(
        "--model", default="esm2_t33_650M_UR50D",
        help="ESM-2 model variant"
    )
    extract_parser.add_argument(
        "--topology", default=None,
        help="UniProt topology annotation file (.dat format)"
    )
    extract_parser.add_argument(
        "--output-dir", "-o", required=True, help="Output directory for features"
    )
    extract_parser.add_argument(
        "--device", default="cuda", choices=["cpu", "cuda"],
        help="Device for feature extraction"
    )

    # ---- evaluate ----
    eval_parser = subparsers.add_parser(
        "evaluate", help="Evaluate predictions against labeled data"
    )
    eval_parser.add_argument(
        "--predictions", "-p", required=True,
        help="Path to predictions JSON file"
    )
    eval_parser.add_argument(
        "--labels", "-l", required=True,
        help="Path to labeled test data (CSV with gpcr_id, family, label columns)"
    )
    eval_parser.add_argument(
        "--output", "-o", default=None, help="Output metrics JSON file"
    )

    # ---- train ----
    train_parser = subparsers.add_parser(
        "train", help="Train cross-attention model"
    )
    train_parser.add_argument(
        "--config", "-c", required=True,
        help="Training configuration YAML/JSON file"
    )
    train_parser.add_argument(
        "--output-dir", "-o", required=True,
        help="Output directory for model checkpoints"
    )

    args = parser.parse_args()

    if args.command == "predict":
        run_predict(args)
    elif args.command == "extract-features":
        run_extract_features(args)
    elif args.command == "evaluate":
        run_evaluate(args)
    elif args.command == "train":
        run_train(args)
    else:
        parser.print_help()


def run_predict(args):
    """Execute prediction command."""
    from .predict import CouplingPredictor

    predictor = CouplingPredictor(model_dir=args.model_dir)
    gpcr_feat = _load_features(args.gpcr)

    # Determine if user passed a family name or feature file
    families = CouplingPredictor.G_PROTEIN_FAMILIES
    family_to_file = {
        f: Path(args.model_dir or "pretrained_models") / f"gprotein_{f}_650m.npy"
        for f in families
    }

    if args.all_families:
        gprot_feats = {}
        for fam in families:
            feat_path = family_to_file[fam]
            if feat_path.exists():
                gprot_feats[fam] = np.load(feat_path)
        results = predictor.predict_all_families(gpcr_feat, gprot_feats)
    else:
        if args.gprotein in families:
            feat_path = family_to_file[args.gprotein]
            gprot_feat = np.load(feat_path) if feat_path.exists() else np.zeros(1280)
        else:
            gprot_feat = _load_features(args.gprotein)
        results = {args.gprotein: predictor.predict(gpcr_feat, gprot_feat, args.gprotein)}

    output = {
        "gpcr_features_shape": gpcr_feat.shape,
        "predictions": results,
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Predictions saved to {args.output}")
    else:
        print(json.dumps(output, indent=2))


def run_extract_features(args):
    """Execute feature extraction command."""
    from .features import extract_esm_embeddings, parse_uniprot_topology

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read FASTA
    sequences = _read_fasta(args.input)
    print(f"Loaded {len(sequences)} sequences from {args.input}")

    # Extract ESM embeddings
    embeddings = extract_esm_embeddings(
        sequences, model_name=args.model, device=args.device,
        output_path=output_dir
    )
    print(f"Extracted embeddings for {len(embeddings)} sequences")

    # Save metadata
    meta = {
        "n_sequences": len(sequences),
        "model": args.model,
        "embedding_dim": embeddings[list(embeddings.keys())[0]].shape[0],
        "sequences": {s[0]: s[1] for s in sequences},
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to {output_dir / 'metadata.json'}")


def run_evaluate(args):
    """Execute evaluation command."""
    from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score

    predictions = _load_json(args.predictions)
    labels_df = pd.read_csv(args.labels)

    y_true, y_pred = [], []
    for _, row in labels_df.iterrows():
        gpcr_id = row["gpcr_id"]
        family = row["family"]
        if gpcr_id in predictions and family in predictions[gpcr_id]:
            y_true.append(row["label"])
            y_pred.append(predictions[gpcr_id][family]["probability"])

    results = {
        "n_samples": len(y_true),
        "auc_roc": float(roc_auc_score(y_true, y_pred)),
        "auc_pr": float(average_precision_score(y_true, y_pred)),
        "brier_score": float(brier_score_loss(y_true, y_pred)),
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Metrics saved to {args.output}")
    else:
        print(json.dumps(results, indent=2))


def run_train(args):
    """Execute training command."""
    print(
        "Training module: use paired_cross_validation_enhanced_v2_650m.py "
        "for reproducible cross-validation training as described in the manuscript."
    )
    print(f"Config file: {args.config}")
    print(f"Output directory: {args.output_dir}")
    print(
        "For full training, run: "
        "python paired_cross_validation_enhanced_v2_650m.py"
    )


def _load_features(path: str) -> np.ndarray:
    """Load features from .npy file or compute from FASTA."""
    p = Path(path)
    if p.suffix == ".npy":
        return np.load(p)
    elif p.suffix in {".fasta", ".fa", ".faa"}:
        from .features import extract_esm_embeddings
        seqs = _read_fasta(path)
        emb = extract_esm_embeddings(seqs, device="cpu")
        return list(emb.values())[0]
    else:
        raise ValueError(f"Unsupported file type: {p.suffix}. Use .npy or .fasta")


def _read_fasta(path: str) -> List[Tuple[str, str]]:
    """Read FASTA file and return list of (header, sequence) tuples."""
    sequences = []
    current_header, current_seq = "", []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_header:
                    sequences.append((current_header, "".join(current_seq)))
                current_header = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
    if current_header:
        sequences.append((current_header, "".join(current_seq)))
    return sequences


def _load_json(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    main()
