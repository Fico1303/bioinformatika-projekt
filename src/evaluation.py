from pathlib import Path
from typing import List, Dict
import csv


def extract_true_label(read_id: str) -> str:
    """
    Iz read ID-a tipa 'bacterium2|neki_originalni_id'
    vrati 'bacterium2'.
    """
    return read_id.split("|")[0]


def add_true_labels(results: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """
    Svakom retku doda true_label na temelju read_id.
    """
    updated = []

    for row in results:
        new_row = dict(row)
        new_row["true_label"] = extract_true_label(str(row["read_id"]))
        updated.append(new_row)

    return updated


def add_correct_flag(results: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """
    Dodaje stupac is_correct.
    """
    updated = []

    for row in results:
        new_row = dict(row)
        new_row["is_correct"] = (
            str(new_row.get("true_label", "")) == str(new_row.get("predicted_label", ""))
        )
        updated.append(new_row)

    return updated


def compute_accuracy(results: List[Dict[str, object]]) -> float:
    """
    Izračuna točnost klasifikacije.
    """
    if not results:
        return 0.0

    correct = 0
    total = len(results)

    for row in results:
        if str(row["true_label"]) == str(row["predicted_label"]):
            correct += 1

    return correct / total


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

    remaining = sorted(k for k in all_keys if k not in preferred_order)
    fieldnames = [k for k in preferred_order if k in all_keys] + remaining

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
    minimap_results: List[Dict[str, object]]
) -> List[Dict[str, object]]:
    """
    Spoji rezultate k-mer i minimap metode po read_id.
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


def compute_agreement(comparison_rows: List[Dict[str, object]]) -> float:
    """
    Koliki je udio readova na kojima se metode slažu.
    """
    valid = [row for row in comparison_rows if row.get("methods_agree") != ""]
    if not valid:
        return 0.0

    agree = sum(1 for row in valid if row["methods_agree"] in [True, "True", "true", "TRUE"])
    return agree / len(valid)