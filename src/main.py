from pathlib import Path
import csv
import shutil
from collections import defaultdict, Counter

from io_utils import find_reference_files, iter_fastq_reads
from classifier import build_reference_profiles, classify_reads_iter
from evaluation import (
    add_true_labels,
    add_correct_flag,
    add_true_label_to_row,
    add_correct_flag_to_row,
    compute_accuracy,
    save_results_csv,
    compare_methods,
    compute_agreement,
)
from minimap_baseline import (
    build_combined_reference,
    run_minimap2,
    parse_paf_best_hits,
    summarize_assignments,
)
from metrics import (
    compute_classification_report,
    format_classification_report,
)
from plots import (
    save_text,
    save_report_csv,
    plot_accuracy_comparison,
    plot_assignments_bar,
    plot_confusion_matrix,
)


# =========================================================
# CONFIG
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "data" / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

INPUT_READS_PATH = PROCESSED_DIR / "uzorak.fastq"

K = 7
N_TEST_READS = None

KMER_ASSIGNMENTS_CSV = RESULTS_DIR / "kmer_assignments.csv"
MINIMAP_ASSIGNMENTS_CSV = RESULTS_DIR / "minimap2_assignments.csv"

SUMMARY_OVERALL_CSV = RESULTS_DIR / "summary_overall.csv"
SUMMARY_BY_BACTERIUM_CSV = RESULTS_DIR / "summary_by_bacterium.csv"

MINIMAP_DIR = PROCESSED_DIR / "minimap"
COMBINED_REFERENCE_PATH = MINIMAP_DIR / "all_references.fasta"
PAF_PATH = MINIMAP_DIR / "mappings.paf"
MINIMAP2_PRESET = "map-ont"


# =========================================================
# HELPERS
# =========================================================

def ensure_directories() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    MINIMAP_DIR.mkdir(parents=True, exist_ok=True)


def print_section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def as_bool(value) -> bool:
    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in {"true", "1", "yes", "da"}


def format_counts(counter: Counter) -> str:
    return "; ".join(f"{label}={count}" for label, count in counter.items())


# =========================================================
# PIPELINE STEPS
# =========================================================

def run_kmer_classification():
    print_section("1) K-MER KLASIFIKACIJA")

    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError(f"Ne postoji folder s raw podacima: {RAW_DATA_DIR}")

    if not INPUT_READS_PATH.exists():
        raise FileNotFoundError(f"Ne postoji FASTQ uzorak: {INPUT_READS_PATH}")

    reference_files = find_reference_files(str(RAW_DATA_DIR))

    if not reference_files:
        raise ValueError(f"Nisam našao reference u folderu: {RAW_DATA_DIR}")

    reference_profiles = build_reference_profiles(reference_files, K)
    reads_iter = iter_fastq_reads(str(INPUT_READS_PATH), limit=N_TEST_READS)

    results = []
    writer = None
    handle = None

    try:
        for i, row in enumerate(classify_reads_iter(reads_iter, reference_profiles, K), start=1):
            row = add_true_label_to_row(row)
            row = add_correct_flag_to_row(row)

            if writer is None:
                fieldnames = list(row.keys())
                KMER_ASSIGNMENTS_CSV.parent.mkdir(parents=True, exist_ok=True)
                handle = open(KMER_ASSIGNMENTS_CSV, "w", newline="", encoding="utf-8")
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()

            writer.writerow(row)
            results.append(row)

            if i % 1000 == 0:
                print(f"[INFO] Obrađeno {i} readova...")

    finally:
        if handle is not None:
            handle.close()

    if not results:
        raise ValueError(f"Nisam učitao nijedan read iz: {INPUT_READS_PATH}")

    accuracy = compute_accuracy(results)

    print(f"[RESULT] k-mer broj readova: {len(results)}")
    print(f"[RESULT] k-mer accuracy: {accuracy:.4f}")

    return results


def run_minimap2_baseline():
    print_section("2) MINIMAP2 BASELINE")

    if shutil.which("minimap2") is None:
        raise RuntimeError(
            "Ne nalazim 'minimap2' u PATH-u. "
            "Provjeri instalaciju i da je naredba dostupna iz terminala."
        )

    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError(f"Ne postoji direktorij s referencama: {RAW_DATA_DIR}")

    if not INPUT_READS_PATH.exists():
        raise FileNotFoundError(f"Ne postoji FASTQ uzorak: {INPUT_READS_PATH}")

    reference_files = find_reference_files(str(RAW_DATA_DIR))

    if not reference_files:
        raise ValueError(f"Nisam našao reference u: {RAW_DATA_DIR}")

    build_combined_reference(reference_files, COMBINED_REFERENCE_PATH)
    run_minimap2(
        COMBINED_REFERENCE_PATH,
        INPUT_READS_PATH,
        PAF_PATH,
        preset=MINIMAP2_PRESET,
    )

    best_hits = parse_paf_best_hits(PAF_PATH)

    if not best_hits:
        raise ValueError("PAF je prazan ili nijedan read nije mapiran.")

    results = list(best_hits.values())
    results = add_true_labels(results)
    results = add_correct_flag(results)

    accuracy = compute_accuracy(results)
    counts = summarize_assignments(results)

    save_results_csv(results, MINIMAP_ASSIGNMENTS_CSV)

    print("[RESULT] Minimap2 broj readova po predikciji:")
    for label, count in counts.items():
        print(f"  - {label}: {count}")

    print(f"[RESULT] Minimap2 broj klasificiranih readova: {len(results)}")
    print(f"[RESULT] Minimap2 accuracy: {accuracy:.4f}")

    return results


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

    print(f"[INFO] Ukupni rezultati spremljeni u: {SUMMARY_OVERALL_CSV}")
    print(f"[INFO] Rezultati po bakteriji spremljeni u: {SUMMARY_BY_BACTERIUM_CSV}")


def run_method_comparison(kmer_results, minimap_results):
    print_section("3) USPOREDBA METODA")

    comparison = compare_methods(kmer_results, minimap_results)

    kmer_accuracy = compute_accuracy(kmer_results)
    minimap_accuracy = compute_accuracy(minimap_results)
    agreement = compute_agreement(comparison)

    print(f"[RESULT] k-mer accuracy:    {kmer_accuracy:.4f}")
    print(f"[RESULT] minimap2 accuracy: {minimap_accuracy:.4f}")
    print(f"[RESULT] slaganje metoda:   {agreement:.4f}")

    save_aggregate_results(kmer_results, minimap_results, comparison)

    return comparison


def run_reports_and_plots(kmer_results, minimap_results):
    print_section("4) REPORTOVI I GRAFOVI")

    kmer_report = compute_classification_report(kmer_results)
    minimap_report = compute_classification_report(minimap_results)

    kmer_report_text = format_classification_report(kmer_report)
    minimap_report_text = format_classification_report(minimap_report)

    save_text(kmer_report_text, PLOTS_DIR / "kmer_classification_report.txt")
    save_text(minimap_report_text, PLOTS_DIR / "minimap2_classification_report.txt")

    save_report_csv(kmer_report, PLOTS_DIR / "kmer_classification_report.csv")
    save_report_csv(minimap_report, PLOTS_DIR / "minimap2_classification_report.csv")

    plot_accuracy_comparison(
        kmer_results,
        minimap_results,
        PLOTS_DIR / "accuracy_comparison.png",
    )

    plot_assignments_bar(
        kmer_results,
        "k-mer: broj readova po predikciji",
        PLOTS_DIR / "kmer_assignments_bar.png",
    )

    plot_assignments_bar(
        minimap_results,
        "minimap2: broj readova po predikciji",
        PLOTS_DIR / "minimap2_assignments_bar.png",
    )

    plot_confusion_matrix(
        kmer_results,
        "k-mer confusion matrix",
        PLOTS_DIR / "kmer_confusion_matrix.png",
    )

    plot_confusion_matrix(
        minimap_results,
        "minimap2 confusion matrix",
        PLOTS_DIR / "minimap2_confusion_matrix.png",
    )

    print(f"[INFO] Reportovi i grafovi spremljeni u: {PLOTS_DIR}")


# =========================================================
# MAIN
# =========================================================

def main():
    ensure_directories()

    kmer_results = run_kmer_classification()
    minimap_results = run_minimap2_baseline()
    run_method_comparison(kmer_results, minimap_results)
    run_reports_and_plots(kmer_results, minimap_results)

    print_section("GOTOVO")
    print("[INFO] Pipeline je uspješno završen.")
    print(f"[INFO] Ulazni FASTQ:           {INPUT_READS_PATH}")
    print(f"[INFO] Ukupni rezultati:       {SUMMARY_OVERALL_CSV}")
    print(f"[INFO] Rezultati po bakteriji: {SUMMARY_BY_BACTERIUM_CSV}")
    print(f"[INFO] Grafovi i reportovi:    {PLOTS_DIR}")


if __name__ == "__main__":
    main()