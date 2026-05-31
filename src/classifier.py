# Kod napisao: Domagoj Matković

from typing import Dict, Iterator, Tuple
from math import sqrt
from Bio.SeqRecord import SeqRecord

from io_utils import load_all_fasta_sequences
from kmer import count_kmers, normalize_kmer_counts


Profile = Dict[str, float]
PreparedProfile = Tuple[Profile, float]


def vector_norm(profile: Dict[str, float]) -> float:
    return sqrt(sum(v * v for v in profile.values()))


def build_normalized_kmer_profile(sequence: str, k: int) -> Tuple[Profile, float]:
    """
    Izgradi normalizirani k-mer profil i odmah izračunaj njegovu normu.
    """
    counts = count_kmers(sequence, k)
    profile = normalize_kmer_counts(counts)
    norm = vector_norm(profile)
    return profile, norm


def cosine_similarity_precomputed(
    query_profile: Profile,
    query_norm: float,
    ref_profile: Profile,
    ref_norm: float,
) -> float:
    """
    Cosine similarity uz unaprijed izračunate norme.
    Računa dot produkt preko manjeg dict-a radi brzine.
    """
    if query_norm == 0.0 or ref_norm == 0.0:
        return 0.0

    if len(query_profile) <= len(ref_profile):
        dot = sum(value * ref_profile.get(kmer, 0.0) for kmer, value in query_profile.items())
    else:
        dot = sum(query_profile.get(kmer, 0.0) * value for kmer, value in ref_profile.items())

    return dot / (query_norm * ref_norm)


def build_reference_profiles(
    reference_files: Dict[str, str],
    k: int
) -> Dict[str, PreparedProfile]:
    """
    Za svaku referencu izgradi normalizirani k-mer profil + normu.
    """
    profiles: Dict[str, PreparedProfile] = {}

    for bacterium_name, fasta_path in reference_files.items():
        sequence = load_all_fasta_sequences(fasta_path)
        profile, norm = build_normalized_kmer_profile(sequence, k)
        profiles[bacterium_name] = (profile, norm)

        print(
            f"[INFO] Profil izgrađen za {bacterium_name}: "
            f"len={len(sequence)}, unique_kmers={len(profile)}, norm={norm:.6f}"
        )

    return profiles


def classify_sequence(
    sequence: str,
    reference_profiles: Dict[str, PreparedProfile],
    k: int
) -> Tuple[str, Dict[str, float]]:
    """
    Klasificira jednu sekvencu prema svim referencama.
    Vraća:
    - najbolju referencu
    - score za svaku referencu
    """
    read_profile, read_norm = build_normalized_kmer_profile(sequence, k)

    scores: Dict[str, float] = {}

    for bacterium_name, (ref_profile, ref_norm) in reference_profiles.items():
        score = cosine_similarity_precomputed(
            read_profile,
            read_norm,
            ref_profile,
            ref_norm,
        )
        scores[bacterium_name] = score

    best_match = max(scores, key=scores.get)
    return best_match, scores


def classify_read(
    read: SeqRecord,
    reference_profiles: Dict[str, PreparedProfile],
    k: int
) -> Tuple[str, Dict[str, float]]:
    """
    Klasificira jedno očitanje.
    """
    return classify_sequence(str(read.seq), reference_profiles, k)


def classify_reads_iter(
    reads: Iterator[SeqRecord],
    reference_profiles: Dict[str, PreparedProfile],
    k: int
):
    """
    Streaming klasifikacija readova.
    Ne sprema sve rezultate u memoriju odjednom.
    """
    for read in reads:
        best_match, scores = classify_read(read, reference_profiles, k)

        row = {
            "read_id": read.id,
            "predicted_label": best_match,
            "best_score": scores[best_match],
        }

        for bacterium_name, score in scores.items():
            row[f"score_{bacterium_name}"] = score

        yield row