#!/usr/bin/env python3
"""
Extract ESM-2 embeddings for protein sequences.

Usage:
    python extract_esm_features.py \
        --input data/extended_sequences.json \
        --output data/gpcr_esm_features.json \
        --model esm2_t6_8M_UR50D
"""

import json
import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm


def load_model(model_name: str):
    """Load ESM-2 model and alphabet."""
    import esm

    if model_name == "esm2_t6_8M_UR50D":
        model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    elif model_name == "esm2_t33_650M_UR50D":
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model = model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model, alphabet


def extract_sequence_embedding(model, alphabet, sequence: str) -> np.ndarray:
    """Extract mean-pooled ESM-2 embedding for a single sequence."""
    batch_converter = alphabet.get_batch_converter()
    data = [("protein", sequence)]
    _, _, batch_tokens = batch_converter(data)

    if torch.cuda.is_available():
        batch_tokens = batch_tokens.cuda()

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[model.num_layers], return_contacts=False)

    token_representations = results["representations"][model.num_layers]
    # Mean pool over sequence length (excluding special tokens)
    seq_rep = token_representations[0, 1 : len(sequence) + 1].mean(0).cpu().numpy()
    return seq_rep


def main():
    parser = argparse.ArgumentParser(description="Extract ESM-2 embeddings")
    parser.add_argument("--input", required=True, help="Input JSON with {id: sequence}")
    parser.add_argument("--output", required=True, help="Output JSON for embeddings")
    parser.add_argument("--model", default="esm2_t6_8M_UR50D", help="ESM-2 model name")
    args = parser.parse_args()

    with open(args.input) as f:
        sequences = json.load(f)

    model, alphabet = load_model(args.model)

    embeddings = {}
    for uid, seq_rec in tqdm(sequences.items(), desc="Extracting ESM-2"):
        seq = seq_rec["sequence"] if isinstance(seq_rec, dict) else seq_rec
        embeddings[uid] = extract_sequence_embedding(model, alphabet, seq).tolist()

    with open(args.output, "w") as f:
        json.dump(embeddings, f)

    print(f"Saved {len(embeddings)} embeddings to {args.output}")


if __name__ == "__main__":
    main()
