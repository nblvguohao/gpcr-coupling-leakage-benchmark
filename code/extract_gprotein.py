#!/usr/bin/env python3
"""
Extract 650M ESM-2 (1280-d) embeddings for the 7 human G protein alpha subunits.

Output: paired_dataset/g_protein_esm_features_650m.json
"""

import json, requests, torch, numpy as np
from pathlib import Path
import esm

BASE = Path(__file__).parent
OUTPUT_FILE = BASE.parent / "data" / "g_protein_esm_features_650m.json"

UNIPROT_IDS = {
    "GNAQ": "P50148",
    "GNAI1": "P63096",
    "GNAI2": "P04899",
    "GNAI3": "P08754",
    "GNAS": "P63092",
    "GNA12": "Q03113",
    "GNA13": "Q14344",
}


def fetch_sequence(uniprot_id):
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    lines = resp.text.strip().split("\n")
    return "".join(lines[1:])


def load_esm_model():
    print("[INFO] Loading ESM-2 t33 650M model ...")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    if torch.cuda.is_available():
        model = model.eval().cuda()
        print("[INFO] Using CUDA.")
    else:
        model = model.eval().cpu()
        print("[INFO] Using CPU (will be slow).")
    batch_converter = alphabet.get_batch_converter()
    return model, batch_converter


def extract(model, batch_converter, name, seq):
    _, _, batch_tokens = batch_converter([(name, seq)])
    if torch.cuda.is_available():
        batch_tokens = batch_tokens.cuda()
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
    token_repr = results["representations"][33].cpu().numpy()
    seq_len = len(seq)
    features = token_repr[0, 1:seq_len+1, :]
    mean_pooled = features.mean(axis=0).tolist()
    return {"mean_pooling": mean_pooled}, features


def main():
    print("=" * 60)
    print("  G Protein 650M ESM-2 Feature Extraction")
    print("=" * 60)

    sequences = {}
    for name, uid in UNIPROT_IDS.items():
        seq = fetch_sequence(uid)
        sequences[name] = seq
        print(f"  {name} ({uid}): {len(seq)} residues")

    model, batch_converter = load_esm_model()

    output = {}
    for name, seq in sequences.items():
        print(f"  Extracting {name} ...")
        feats, _ = extract(model, batch_converter, name, seq)
        output[name] = {
            "uniprot_id": UNIPROT_IDS[name],
            "sequence_length": len(seq),
            "mean_pooling": feats["mean_pooling"],
        }
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f)
    print(f"[OK] Saved to {OUTPUT_FILE}")
    sample = list(output.values())[0]
    print(f"[OK] Sample dimension: {len(sample['mean_pooling'])}")


if __name__ == "__main__":
    main()
