# Kod napisao: Domagoj Matković

from collections import Counter
from typing import Dict, List


def get_kmers(sequence: str, k: int) -> List[str]:
    """
    Vrati sve k-mere iz sekvence, preskačući one koji sadrže N.
    """
    sequence = sequence.upper()

    if k <= 0:
        raise ValueError("k mora biti > 0")

    if len(sequence) < k:
        return []

    kmers: List[str] = []

    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i + k]
        if "N" in kmer:
            continue
        kmers.append(kmer)

    return kmers


def count_kmers(sequence: str, k: int) -> Dict[str, int]:
    """
    Broji k-mere u sekvenci.
    """
    return dict(Counter(get_kmers(sequence, k)))


def normalize_kmer_counts(counts: Dict[str, int]) -> Dict[str, float]:
    """
    Pretvara brojanja u relativne frekvencije.
    """
    total = sum(counts.values())

    if total == 0:
        return {}

    return {kmer: count / total for kmer, count in counts.items()}