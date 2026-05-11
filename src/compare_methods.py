from pathlib import Path
import csv
from collections import defaultdict, Counter

from evaluation import (
    load_results_csv,
    compare_methods,
    compute_accuracy,
    compute_agreement,
)


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "data" / "results"

KMER_CSV = RESULTS_DIR / "kmer_assignments.csv"
MINIMAP_CSV = RESULTS_DIR / "minimap2_assignments.csv"

SUMMARY_OVERALL_CSV = RESULTS_DIR / "summary_overall.csv"
SUMMARY_BY_BACTERIUM_CSV = RESULTS_DIR / "summary_by_bacterium.csv"


def as_bool(value) -> bool:
    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in {"true", "1", "yes", "da"}


def format_counts(counter: Counter) -> str:
    return "; ".join(f"{label}={count}" for label, count in counter.items())


def save_aggregate_results(kmer_results, minimap_results, comparison_rows) -> None:
    kmer_accuracy = compute_accuracy(kmer_results)
    minimap_accuracy = compute_accuracy(minimap_results)
    agreement = compute_agreement(comparison_rows)

    overall_rows = [
        {"metric": "kmer_accuracy", "value": f"{kmer_accuracy:.6f}"},
        {"metric": "minimap2_accuracy", "value": f"{minimap_accuracy:.6f}"},
        {"metric": "methods_agreement", "value": f"{agreement:.6f}"},
        {"metric": "kmer_n_results", "value": str(len(kmer_results))},
        {"metric": "minimap2_n_results", "value": str(len(minimap_results))},
        {"metric": "comparison_n_rows", "value": str(len(comparison_rows))},
    ]

    with open(SUMMARY_OVERALL_CSV, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(overall_rows)

    by_bacterium = defaultdict(lambda: {
        "n_reads": 0,
        "kmer_correct": 0,
        "minimap_correct": 0,
        "methods_agree": 0,
        "kmer_predictions": Counter(),
        "minimap_predictions": Counter(),
    })

    for row in comparison_rows:
        true_label = row.get("true_label", "").strip() or "UNKNOWN"
        kmer_predicted = row.get("kmer_predicted", "").strip()
        minimap_predicted = row.get("minimap_predicted", "").strip()

        stats = by_bacterium[true_label]
        stats["n_reads"] += 1

        if as_bool(row.get("kmer_correct")):
            stats["kmer_correct"] += 1

        if as_bool(row.get("minimap_correct")):
            stats["minimap_correct"] += 1

        if as_bool(row.get("methods_agree")):
            stats["methods_agree"] += 1

        if kmer_predicted:
            stats["kmer_predictions"][kmer_predicted] += 1

        if minimap_predicted:
            stats["minimap_predictions"][minimap_predicted] += 1
        else:
            stats["minimap_predictions"]["UNMAPPED"] += 1

    bacteria_rows = []

    for bacterium, stats in sorted(by_bacterium.items()):
        n_reads = stats["n_reads"]

        bacteria_rows.append({
            "bacterium": bacterium,
            "n_reads": n_reads,
            "kmer_correct": stats["kmer_correct"],
            "kmer_accuracy": f"{stats['kmer_correct'] / n_reads:.6f}",
            "minimap2_correct": stats["minimap_correct"],
            "minimap2_accuracy": f"{stats['minimap_correct'] / n_reads:.6f}",
            "methods_agree": stats["methods_agree"],
            "methods_agreement": f"{stats['methods_agree'] / n_reads:.6f}",
            "kmer_predicted_counts": format_counts(stats["kmer_predictions"]),
            "minimap2_predicted_counts": format_counts(stats["minimap_predictions"]),
        })

    fieldnames = [
        "bacterium",
        "n_reads",
        "kmer_correct",
        "kmer_accuracy",
        "minimap2_correct",
        "minimap2_accuracy",
        "methods_agree",
        "methods_agreement",
        "kmer_predicted_counts",
        "minimap2_predicted_counts",
    ]

    with open(SUMMARY_BY_BACTERIUM_CSV, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(bacteria_rows)


def main():
    print("[INFO] Učitavam k-mer rezultate...")
    kmer_results = load_results_csv(KMER_CSV)

    print("[INFO] Učitavam minimap2 rezultate...")
    minimap_results = load_results_csv(MINIMAP_CSV)

    comparison = compare_methods(kmer_results, minimap_results)

    kmer_accuracy = compute_accuracy(kmer_results)
    minimap_accuracy = compute_accuracy(minimap_results)
    agreement = compute_agreement(comparison)

    save_aggregate_results(kmer_results, minimap_results, comparison)

    print(f"[RESULT] k-mer accuracy:    {kmer_accuracy:.4f}")
    print(f"[RESULT] minimap2 accuracy: {minimap_accuracy:.4f}")
    print(f"[RESULT] slaganje metoda:   {agreement:.4f}")

    print(f"[INFO] Ukupni rezultati spremljeni u: {SUMMARY_OVERALL_CSV}")
    print(f"[INFO] Rezultati po bakteriji spremljeni u: {SUMMARY_BY_BACTERIUM_CSV}")


if __name__ == "__main__":
    main()