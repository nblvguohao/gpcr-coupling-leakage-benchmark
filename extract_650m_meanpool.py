#!/usr/bin/env python3
"""
Extract esm2_t33_650M_UR50D mean-pooled features for all GPCR sequences.
Output: paired_dataset/gpcr_esm_features_650m.json  (1280-d mean-pooled vectors)

This replaces the deleted server_sync/ data.
"""
import json, torch, numpy as np, time
from pathlib import Path
from tqdm import tqdm
import esm

BASE = Path(__file__).parent
SEQUENCES_FILE = BASE / "reproducible_package" / "data" / "extended_sequences.json"
OUTPUT_FILE = BASE / "paired_dataset" / "gpcr_esm_features_650m.json"

def load_esm_model():
    print("[INFO] Loading ESM-2 650M model ...")
    t0 = time.time()
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    if torch.cuda.is_available():
        model = model.eval().cuda()
        print(f"[INFO] Using CUDA ({torch.cuda.get_device_name(0)})")
    else:
        model = model.eval().cpu()
        print("[INFO] Using CPU (will be very slow)")
    batch_converter = alphabet.get_batch_converter()
    print(f"[INFO] Model loaded in {time.time()-t0:.1f}s")
    return model, batch_converter

def extract_mean_pooled(model, batch_converter, uid: str, seq: str):
    _, _, batch_tokens = batch_converter([(uid, seq)])
    try:
        if torch.cuda.is_available():
            batch_tokens = batch_tokens.cuda()
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
    except torch.OutOfMemoryError:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            batch_tokens = batch_tokens.cpu()
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33], return_contacts=False)
        else:
            raise
    token_repr = results["representations"][33].cpu().numpy()
    seq_len = len(seq)
    feat = token_repr[0, 1:seq_len+1, :].mean(axis=0)
    return feat.tolist()

def main():
    print("=" * 60)
    print("  ESM-2 650M Mean-Pooled Feature Extraction")
    print("=" * 60)

    with open(SEQUENCES_FILE, encoding="utf-8") as f:
        sequences = json.load(f)
    print(f"[INFO] Loaded {len(sequences)} sequences")

    model, batch_converter = load_esm_model()

    features = {}
    errors = []
    for uid, entry in tqdm(list(sequences.items()), desc="Extracting"):
        seq_str = entry["sequence"] if isinstance(entry, dict) else entry
        try:
            features[uid] = extract_mean_pooled(model, batch_converter, uid, seq_str)
        except Exception as e:
            errors.append((uid, str(e)))
            print(f"[WARN] Failed on {uid}: {e}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(features, f)

    print(f"[OK] Saved {len(features)} features to {OUTPUT_FILE}")
    if errors:
        print(f"[WARN] {len(errors)} errors: {errors[:5]}")
    sample = np.array(list(features.values())[0])
    print(f"[OK] Feature dim: {sample.shape}")

if __name__ == "__main__":
    main()
