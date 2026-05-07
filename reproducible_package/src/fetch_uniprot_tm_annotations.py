#!/usr/bin/env python3
"""
Fetch curated transmembrane annotations from UniProt REST API.

Usage:
    python fetch_uniprot_tm_annotations.py \
        --sequences data/extended_sequences.json \
        --output data/uniprot_topology.json
"""

import json
import time
import argparse
import requests
from pathlib import Path
from typing import Dict, List, Tuple


def fetch_tm_regions(uniprot_id: str) -> List[Tuple[int, int]]:
    """Fetch Transmembrane regions from UniProt API."""
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    params = {"fields": "ft_transmem"}
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        regions = []
        for feat in data.get("features", []):
            if feat.get("type") == "Transmembrane":
                loc = feat.get("location", {})
                s = loc.get("start", {}).get("value")
                e = loc.get("end", {}).get("value")
                if s is not None and e is not None:
                    regions.append((int(s), int(e)))
        return regions
    except Exception as e:
        print(f"  [WARN] {uniprot_id} failed: {e}")
        return []


def derive_loops(tm_regions: List[Tuple[int, int]], seq_len: int) -> Dict[str, Tuple[int, int]]:
    """Derive loop regions from TM helix boundaries."""
    loops = {}
    if not tm_regions:
        return loops

    if tm_regions[0][0] > 1:
        loops["N-tail"] = (1, tm_regions[0][0] - 1)

    loop_names = ["ICL1", "ECL1", "ICL2", "ECL2", "ICL3", "ECL3"]
    for i in range(len(tm_regions) - 1):
        s = tm_regions[i][1] + 1
        e = tm_regions[i + 1][0] - 1
        if e >= s:
            name = loop_names[i] if i < len(loop_names) else f"loop_{i}"
            loops[name] = (s, e)

    if tm_regions[-1][1] < seq_len:
        loops["C-tail"] = (tm_regions[-1][1] + 1, seq_len)

    return loops


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences", required=True, help="JSON with {id: {sequence: ..., uniprot: ...}}")
    parser.add_argument("--output", required=True, help="Output JSON for topology")
    args = parser.parse_args()

    with open(args.sequences) as f:
        sequences = json.load(f)

    topology = {}
    for uid, rec in sequences.items():
        seq = rec["sequence"] if isinstance(rec, dict) else rec
        uniprot_id = rec.get("uniprot", "") if isinstance(rec, dict) else ""

        if not uniprot_id:
            topology[uid] = {"tm_regions": [], "loops": {}, "source": "none"}
            continue

        tm_regions = fetch_tm_regions(uniprot_id)
        time.sleep(0.2)  # Rate limiting

        loops = derive_loops(tm_regions, len(seq))
        topology[uid] = {
            "tm_regions": tm_regions,
            "loops": loops,
            "source": "uniprot" if tm_regions else "failed",
        }

    with open(args.output, "w") as f:
        json.dump(topology, f, indent=2)

    n_annotated = sum(1 for v in topology.values() if v["tm_regions"])
    n_7tm = sum(1 for v in topology.values() if len(v["tm_regions"]) == 7)
    print(f"Annotated: {n_annotated}/{len(topology)}, Exact 7-TM: {n_7tm}")


if __name__ == "__main__":
    main()
