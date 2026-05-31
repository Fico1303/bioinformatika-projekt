# Kod napisao: Filip Paić

from pathlib import Path
from typing import List, Dict
import csv


def extract_true_label(read_id: str) -> str:
    """
    Iz read ID-a tipa 'bacterium2|neki_originalni_id'
    vrati 'bacterium2'.
    """
    return read_id.split("|")[0]


def add_true_label_to_row(row: Dict[str, object]) -> Dict[str, object]:
    """
    Doda true_label jednom retku.
    """
    new_row = dict(row)
    new_row["true_label"] = extract_true_label(str(row["read_id"]))
    return new_row


def add_true_labels(results: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """
    Svakom retku doda true_label na temelju read_id.
    """
    return [
        add_true_label_to_row(row)
        for row in results
    ]


def add_correct_flag_to_row(row: Dict[str, object]) -> Dict[str, object]:
    """
    Doda is_correct jednom retku.
    """
    new_row = dict(row)
    new_row["is_correct"] = (
        str(new_row.get("true_label", "")) ==
        str(new_row.get("predicted_label", ""))
    )
    return new_row


def add_correct_flag(results: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """
    Dodaje stupac is_correct svakom retku.
    """
    return [
        add_correct_flag_to_row(row)
        for row in results
    ]


def save_results_csv(results: List[Dict[str, object]], output_path: Path) -> None:
    """
    Spremi listu rječnika u CSV.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not results:
        raise ValueError("Nema rezultata za spremanje u CSV.")

    all_keys = set()

    for row in results:
        all_keys.update(row.keys())

    preferred_order = [
        "read_id",
        "true_label",
        "predicted_label",
        "is_correct",
        "best_score",
        "target_name",
        "mapq",
        "n_match",
        "aln_block",
        "identity",
    ]

    remaining = sorted(
        key
        for key in all_keys
        if key not in preferred_order
    )

    fieldnames = [
        key
        for key in preferred_order
        if key in all_keys
    ] + remaining

    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for row in results:
            writer.writerow(row)


def load_results_csv(input_path: Path) -> List[Dict[str, str]]:
    """
    Učitaj CSV rezultata kao listu dictova.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Ne postoji datoteka: {input_path}")

    with open(input_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def compare_methods(
    kmer_results: List[Dict[str, object]],
    minimap_results: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    """
    Spoji rezultate k-mer i minimap metode po read_id.

    Rezultat sadrži:
    - read_id
    - true_label
    - kmer_predicted
    - kmer_correct
    - kmer_best_score
    - minimap_predicted
    - minimap_correct
    - minimap_mapq
    - minimap_identity
    - methods_agree
    """
    minimap_by_read = {
        str(row["read_id"]): row
        for row in minimap_results
    }

    comparison = []

    for kmer_row in kmer_results:
        read_id = str(kmer_row["read_id"])
        minimap_row = minimap_by_read.get(read_id)

        row = {
            "read_id": read_id,
            "true_label": kmer_row.get("true_label", ""),
            "kmer_predicted": kmer_row.get("predicted_label", ""),
            "kmer_correct": kmer_row.get("is_correct", ""),
            "kmer_best_score": kmer_row.get("best_score", ""),
            "minimap_predicted": "",
            "minimap_correct": "",
            "minimap_mapq": "",
            "minimap_identity": "",
            "methods_agree": "",
        }

        if minimap_row is not None:
            row["minimap_predicted"] = minimap_row.get("predicted_label", "")
            row["minimap_correct"] = minimap_row.get("is_correct", "")
            row["minimap_mapq"] = minimap_row.get("mapq", "")
            row["minimap_identity"] = minimap_row.get("identity", "")
            row["methods_agree"] = (
                str(kmer_row.get("predicted_label", "")) ==
                str(minimap_row.get("predicted_label", ""))
            )

        comparison.append(row)

    return comparison


def save_comparison_csv(
    comparison_rows: List[Dict[str, object]],
    output_path: Path,
) -> None:
    """
    Spremi usporedbu k-mer i minimap metode u CSV.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not comparison_rows:
        raise ValueError("Nema usporednih rezultata za spremanje u CSV.")

    fieldnames = [
        "read_id",
        "true_label",
        "kmer_predicted",
        "kmer_correct",
        "kmer_best_score",
        "minimap_predicted",
        "minimap_correct",
        "minimap_mapq",
        "minimap_identity",
        "methods_agree",
    ]

    all_keys = set()

    for row in comparison_rows:
        all_keys.update(row.keys())

    remaining = sorted(
        key
        for key in all_keys
        if key not in fieldnames
    )

    fieldnames = [
        key
        for key in fieldnames
        if key in all_keys
    ] + remaining

    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparison_rows)