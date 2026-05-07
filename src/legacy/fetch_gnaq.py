#!/usr/bin/env python3
"""Fetch human GNAQ (P50148) sequence from UniProt and save."""
import json
import requests
from pathlib import Path

OUTPUT = Path(__file__).parent / "gnaq_uniprot.json"

url = "https://rest.uniprot.org/uniprotkb/P50148.json"
resp = requests.get(url)
resp.raise_for_status()
data = resp.json()

seq = data["sequence"]["value"]
print(f">P50148 {data['proteinDescription']['recommendedName']['fullName']['value']}")
print(seq)
print(f"Length: {len(seq)}")

with open(OUTPUT, "w") as f:
    json.dump({
        "uniprot_id": "P50148",
        "protein_name": data["proteinDescription"]["recommendedName"]["fullName"]["value"],
        "sequence": seq,
        "length": len(seq),
    }, f, indent=2)

print(f"Saved to {OUTPUT}")
