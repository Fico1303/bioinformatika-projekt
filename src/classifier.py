from typing import Dict, List, Tuple
from Bio.SeqRecord import SeqRecord

from io_utils import load_all_fasta_sequences
from kmer import count_kmers, normalize_kmer_counts
from metrics import cosine_similarity_dicts


def build_reference_profiles(reference_files: Dict[str, str], k: int) -> Dict[str, Dict[str, float]]:
    """
    Za svaku referencu izgradi normalizirani k-mer profil.
    """
    profiles: Dict[str, Dict[str, float]] = {}

    for bacterium_name, fasta_path in reference_files.items():
        sequence = load_all_fasta_sequences(fasta_path)
        counts = count_kmers(sequence, k)
        profile = normalize_kmer_counts(counts)
        profiles[bacterium_name] = profile

        print(
            f"[INFO] Profil izgrađen za {bacterium_name}: "
            f"len={len(sequence)}, unique_kmers={len(profile)}"
        )

    return profiles


def classify_sequence(
    sequence: str,
    reference_profiles: Dict[str, Dict[str, float]],
    k: int
) -> Tuple[str, Dict[str, float]]:
    """
    Klasificira jednu sekvencu prema svim referencama.
    Vraća:
    - najbolju referencu
    - score za svaku referencu
    """
    read_counts = count_kmers(sequence, k)
    read_profile = normalize_kmer_counts(read_counts)

    scores: Dict[str, float] = {}

    for bacterium_name, ref_profile in reference_profiles.items():
        score = cosine_similarity_dicts(read_profile, ref_profile)
        scores[bacterium_name] = score

    best_match = max(scores, key=scores.get)
    return best_match, scores


def classify_read(
    read: SeqRecord,
    reference_profiles: Dict[str, Dict[str, float]],
    k: int
) -> Tuple[str, Dict[str, float]]:
    """
    Klasificira jedno očitanje.
    """
    return classify_sequence(str(read.seq), reference_profiles, k)


def classify_reads(
    reads: List[SeqRecord],
    reference_profiles: Dict[str, Dict[str, float]],
    k: int
) -> List[Dict[str, object]]:
    """
    Klasificira listu očitanja.
    """
    results: List[Dict[str, object]] = []

    for read in reads:
        best_match, scores = classify_read(read, reference_profiles, k)

        row = {
            "read_id": read.id,
            "predicted_label": best_match,
            "best_score": scores[best_match],
        }

        for bacterium_name, score in scores.items():
            row[f"score_{bacterium_name}"] = score

        results.append(row)

    return results