"""
gpcr-coupling: Paired GPCR-G protein coupling prediction tool.

Usage:
    gpcr-coupling predict --gpcr P35462.fasta --gprotein Gq
    gpcr-coupling extract-features --input gpcr_sequences.fasta
    gpcr-coupling train --config config.yaml
    gpcr-coupling evaluate --predictions output.json --labels test.csv
"""

__version__ = "1.0.0"
__author__ = "Guohao Lü, Yuxin Xia"
