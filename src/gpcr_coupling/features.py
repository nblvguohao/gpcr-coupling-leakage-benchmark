"""
Feature extraction module: extract ESM-2 embeddings and ICL features.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    import esm
except ImportError:
    esm = None


# AAindex physicochemical properties (partial; full table in supplementary)
AAINDEX = {
    "hydrophobicity": {
        "A": 0.62, "C": 0.29, "D": -0.90, "E": -0.74, "F": 1.19,
        "G": 0.48, "H": -0.40, "I": 1.38, "K": -1.50, "L": 1.06,
        "M": 0.64, "N": -0.78, "P": 0.12, "Q": -0.85, "R": -2.53,
        "S": -0.18, "T": -0.05, "V": 1.08, "W": 0.81, "Y": 0.26,
    },
    "polarity": {
        "A": 8.1, "C": 5.5, "D": 13.0, "E": 12.3, "F": 5.2,
        "G": 9.0, "H": 10.4, "I": 5.2, "K": 11.3, "L": 4.9,
        "M": 5.7, "N": 11.6, "P": 8.0, "Q": 10.5, "R": 10.5,
        "S": 9.2, "T": 8.6, "V": 5.9, "W": 5.4, "Y": 6.2,
    },
}

CHARGED_POS = {"K": 1, "R": 1, "H": 1}
CHARGED_NEG = {"D": -1, "E": -1}
AROMATIC = {"F": 1, "W": 1, "Y": 1, "H": 1}
ALIPHATIC = {"A": 1, "I": 1, "L": 1, "M": 1, "V": 1}
SMALL = {"G": 1, "A": 1, "S": 1}


def compute_physicochemical_stats(sequence: str) -> Dict[str, float]:
    """Compute 8 physicochemical statistics for a protein sequence segment."""
    if len(sequence) == 0:
        return {k: 0.0 for k in [
            "mean_hydrophobicity", "mean_polarity", "net_charge",
            "aromatic_fraction", "aliphatic_fraction", "small_fraction",
            "proline_fraction", "length"
        ]}

    n = len(sequence)
    return {
        "mean_hydrophobicity": np.mean([AAINDEX["hydrophobicity"].get(aa, 0) for aa in sequence]),
        "mean_polarity": np.mean([AAINDEX["polarity"].get(aa, 0) for aa in sequence]),
        "net_charge": sum(CHARGED_POS.get(aa, 0) + CHARGED_NEG.get(aa, 0) for aa in sequence),
        "aromatic_fraction": sum(AROMATIC.get(aa, 0) for aa in sequence) / n,
        "aliphatic_fraction": sum(ALIPHATIC.get(aa, 0) for aa in sequence) / n,
        "small_fraction": sum(SMALL.get(aa, 0) for aa in sequence) / n,
        "proline_fraction": sequence.count("P") / n,
        "length": float(n),
    }


def extract_esm_embeddings(
    sequences: List[Tuple[str, str]],
    model_name: str = "esm2_t33_650M_UR50D",
    device: str = "cuda",
    output_path: Optional[Path] = None,
) -> Dict[str, np.ndarray]:
    """Extract mean-pooled ESM-2 embeddings for a list of sequences.

    Args:
        sequences: List of (identifier, amino_acid_sequence) tuples
        model_name: ESM-2 model variant
        device: 'cuda' or 'cpu'
        output_path: Optional path to save embeddings as .npy files

    Returns:
        Dict mapping sequence identifier to embedding array
    """
    if esm is None:
        raise ImportError(
            "ESM library not installed. Install with: pip install fair-esm"
        )

    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()

    embeddings = {}
    batch_size = 4  # Small batch for 650M model on 8 GB VRAM

    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i + batch_size]
        batch_labels, batch_strs, batch_tokens = batch_converter(batch)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[model.num_layers])
            token_repr = results["representations"][model.num_layers]

        for j, (label, _) in enumerate(batch):
            # Mean pool over sequence (exclude BOS/EOS tokens)
            seq_len = len(batch_strs[j])
            emb = token_repr[j, 1:seq_len + 1].mean(dim=0).cpu().numpy()
            embeddings[label] = emb

            if output_path is not None:
                np.save(output_path / f"{label}.npy", emb)

    return embeddings


def extract_icl_features(
    gpcr_id: str,
    sequence: str,
    topology_annotations: List[Dict],
    esm_embeddings: Optional[np.ndarray] = None,
    embedding_dim: int = 1280,
) -> Dict:
    """Extract ICL2 and ICL3 features for a GPCR sequence.

    Args:
        gpcr_id: GPCR identifier
        sequence: Full GPCR amino acid sequence
        topology_annotations: List of transmembrane topology annotations
            Each item: {'type': 'TRANSMEM', 'start': int, 'end': int}
        esm_embeddings: Per-residue ESM embeddings (n_residues x dim) or None
        embedding_dim: ESM embedding dimension

    Returns:
        Dict with keys: icl2_features, icl3_features, icl2_stats, icl3_stats
    """
    # Parse TM boundaries from topology annotations
    tm_regions = sorted(
        [(a["start"], a["end"]) for a in topology_annotations],
        key=lambda x: x[0],
    )

    # Extract ICL2 (between TM3 and TM4) and ICL3 (between TM5 and TM6)
    icl2_seq, icl3_seq = "", ""
    icl2_start, icl2_end = 0, 0
    icl3_start, icl3_end = 0, 0

    if len(tm_regions) >= 6:
        icl2_start = tm_regions[2][1] + 1  # After TM3
        icl2_end = tm_regions[3][0] - 1    # Before TM4
        icl3_start = tm_regions[4][1] + 1  # After TM5
        icl3_end = tm_regions[5][0] - 1    # Before TM6

        icl2_seq = sequence[icl2_start - 1:icl2_end]
        icl3_seq = sequence[icl3_start - 1:icl3_end]

    # Physicochemical statistics
    icl2_stats = compute_physicochemical_stats(icl2_seq)
    icl3_stats = compute_physicochemical_stats(icl3_seq)

    result = {
        "gpcr_id": gpcr_id,
        "icl2_sequence": icl2_seq,
        "icl3_sequence": icl3_seq,
        "icl2_stats": icl2_stats,
        "icl3_stats": icl3_stats,
    }

    # Local ESM embeddings for ICL regions
    if esm_embeddings is not None:
        icl2_local = np.zeros(embedding_dim)
        icl3_local = np.zeros(embedding_dim)
        if len(icl2_seq) > 0:
            icl2_local = esm_embeddings[icl2_start - 1:icl2_end].mean(axis=0)
        if len(icl3_seq) > 0:
            icl3_local = esm_embeddings[icl3_start - 1:icl3_end].mean(axis=0)
        result["icl2_embedding"] = icl2_local
        result["icl3_embedding"] = icl3_local

    return result


def parse_uniprot_topology(uniprot_dat: str) -> Dict[str, List[Dict]]:
    """Parse UniProt topology annotations from .dat file format.

    Args:
        uniprot_dat: Path to UniProt .dat file or raw text content

    Returns:
        Dict mapping UniProt accession to list of topology annotations
    """
    text = uniprot_dat
    if Path(uniprot_dat).exists():
        text = Path(uniprot_dat).read_text()

    annotations = {}
    current_ac = None

    for line in text.split("\n"):
        if line.startswith("AC"):
            current_ac = line.split()[1].rstrip(";")
            if current_ac not in annotations:
                annotations[current_ac] = []
        elif line.startswith("FT   TRANSMEM") and current_ac is not None:
            match = re.match(r"FT\s+TRANSMEM\s+(\d+)\.\.(\d+)", line)
            if match:
                annotations[current_ac].append({
                    "type": "TRANSMEM",
                    "start": int(match.group(1)),
                    "end": int(match.group(2)),
                })

    return annotations
