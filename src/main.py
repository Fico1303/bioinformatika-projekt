from pathlib import Path
import csv
import shutil

from io_utils import find_reference_files, find_fastq_files, load_fastq_reads
from preprocessing import merge_reads
from classifier import build_reference_profiles, classify_reads
from evaluation import (
    add_true_labels,
    add_correct_flag,
    compute_accuracy,
    save_results_csv,
    load_results_csv,
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
    count_predictions_by_label,
)
from plots import (
    save_text,
    save_report_csv,
    plot_accuracy_comparison,
    plot_assignments_bar,
    plot_confusion_matrix,
    plot_minimap_mapping_status,
)


# =========================================================
# CONFIG
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "data" / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

MIXED_READS_PATH = PROCESSED_DIR / "mixed_metagenome.fastq"

K = 7
N_READS_PER_FILE = 100
N_TEST_READS = None
FILE_FORMAT = "fastq"

KMER_ASSIGNMENTS_CSV = RESULTS_DIR / "kmer_assignments.csv"
MINIMAP_ASSIGNMENTS_CSV = RESULTS_DIR / "minimap2_assignments.csv"
COMPARISON_CSV = RESULTS_DIR / "comparison.csv"

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


def run_preprocessing() -> None:
    """
    Nađe FASTQ fileove po bakteriji i spoji ih u mixed_metagenome.fastq.
    """
    print_section("1) PREPROCESSING - GENERIRANJE MIXED METAGENOME FASTQ")

    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError(f"Ne postoji folder s raw podacima: {RAW_DATA_DIR}")

    input_files = find_fastq_files(str(RAW_DATA_DIR))

    if not input_files:
        raise ValueError(f"Nisam našao nijedan FASTQ file u: {RAW_DATA_DIR}")

    print("[INFO] Nađeni FASTQ fileovi:")
    for bacterium, paths in input_files.items():
        print(f"  - {bacterium}:")
        if isinstance(paths, str):
            print(f"      {paths}")
        else:
            for path in paths:
                print(f"      {path}")

    merge_reads(
        input_files=input_files,
        output_path=str(MIXED_READS_PATH),
        n_reads_per_file=N_READS_PER_FILE,
        file_format=FILE_FORMAT,
    )

    print(f"[INFO] Mixed metagenome spremljen u: {MIXED_READS_PATH}")


def run_kmer_classification():
    """
    K-mer klasifikacija mixed FASTQ readova.
    """
    print_section("2) K-MER KLASIFIKACIJA")

    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError(f"Ne postoji folder s raw podacima: {RAW_DATA_DIR}")

    if not MIXED_READS_PATH.exists():
        raise FileNotFoundError(
            f"Ne postoji mixed FASTQ file: {MIXED_READS_PATH}\n"
            "Prvo ga generiraj preprocessing korakom."
        )

    print("[INFO] Tražim reference...")
    reference_files = find_reference_files(str(RAW_DATA_DIR))

    if not reference_files:
        raise ValueError(f"Nisam našao reference u folderu: {RAW_DATA_DIR}")

    print("[INFO] Nađene reference:")
    for name, path in reference_files.items():
        print(f"  - {name}: {path}")

    print("\n[INFO] Gradim reference profile...")
    reference_profiles = build_reference_profiles(reference_files, K)

    print(f"\n[INFO] Učitavam readove iz mixed FASTQ (limit={N_TEST_READS})...")
    reads = load_fastq_reads(str(MIXED_READS_PATH), limit=N_TEST_READS)

    if not reads:
        raise ValueError(f"Nisam učitao nijedan read iz: {MIXED_READS_PATH}")

    print("[INFO] Klasificiram readove...")
    results = classify_reads(reads, reference_profiles, K)

    print("[INFO] Dodajem true labelove...")
    results = add_true_labels(results)
    results = add_correct_flag(results)

    accuracy = compute_accuracy(results)

    save_results_csv(results, KMER_ASSIGNMENTS_CSV)
    print(f"[INFO] Spremljeni k-mer assignmenti: {KMER_ASSIGNMENTS_CSV}")
    print(f"[RESULT] k-mer accuracy: {accuracy:.4f}")

    print("\n[INFO] Primjer nekoliko readova:")
    for row in results[:10]:
        print(f"Read: {row['read_id']}")
        print(f"  True: {row['true_label']}")
        print(f"  Predicted: {row['predicted_label']}")
        print(f"  Correct: {row['is_correct']}")
        print(f"  Best score: {row['best_score']:.6f}")

        score_keys = sorted(key for key in row if key.startswith("score_"))
        for key in score_keys:
            print(f"  {key}: {row[key]:.6f}")
        print()

    return results


def run_minimap2_baseline():
    """
    Minimap2 baseline nad istim mixed FASTQ readovima.
    """
    print_section("3) MINIMAP2 BASELINE")

    if shutil.which("minimap2") is None:
        raise RuntimeError(
            "Ne nalazim 'minimap2' u PATH-u. "
            "Provjeri instalaciju i da je naredba dostupna iz terminala."
        )

    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError(f"Ne postoji direktorij s referencama: {RAW_DATA_DIR}")

    if not MIXED_READS_PATH.exists():
        raise FileNotFoundError(f"Ne postoji reads datoteka: {MIXED_READS_PATH}")

    print("[INFO] Tražim reference...")
    reference_files = find_reference_files(str(RAW_DATA_DIR))

    if not reference_files:
        raise ValueError(f"Nisam našao reference u: {RAW_DATA_DIR}")

    print("[INFO] Nađene reference:")
    for label, path in reference_files.items():
        print(f"  - {label}: {path}")

    print("\n[INFO] Gradim combined reference FASTA...")
    build_combined_reference(reference_files, COMBINED_REFERENCE_PATH)
    print(f"[INFO] Combined FASTA: {COMBINED_REFERENCE_PATH}")

    print("\n[INFO] Pokrećem minimap2 mapiranje...")
    run_minimap2(COMBINED_REFERENCE_PATH, MIXED_READS_PATH, PAF_PATH, preset=MINIMAP2_PRESET)

    print("\n[INFO] Čitam najbolja mapiranja po readu...")
    best_hits = parse_paf_best_hits(PAF_PATH)

    if not best_hits:
        raise ValueError("PAF je prazan ili nijedan read nije mapiran.")

    results = list(best_hits.values())
    results = add_true_labels(results)
    results = add_correct_flag(results)

    accuracy = compute_accuracy(results)
    counts = summarize_assignments(results)

    save_results_csv(results, MINIMAP_ASSIGNMENTS_CSV)
    print(f"[INFO] Spremljeni minimap2 assignmenti: {MINIMAP_ASSIGNMENTS_CSV}")

    print("\n[RESULT] Broj readova po genomu:")
    total = 0
    for label, count in counts.items():
        print(f"  - {label}: {count}")
        total += count

    print(f"\n[RESULT] Ukupno klasificiranih readova: {total}")
    print(f"[RESULT] Minimap2 accuracy: {accuracy:.4f}")

    print("\n[INFO] Primjer nekoliko najboljih hitova:")
    for row in results[:10]:
        print(
            f"Read: {row['read_id']}\n"
            f"  True: {row['true_label']}\n"
            f"  Predicted: {row['predicted_label']}\n"
            f"  Correct: {row['is_correct']}\n"
            f"  Target: {row['target_name']}\n"
            f"  MAPQ: {row['mapq']}\n"
            f"  Matches: {row['n_match']}\n"
            f"  Aln block: {row['aln_block']}\n"
            f"  Identity: {row['identity']:.4f}\n"
        )

    return results


def run_method_comparison(kmer_results, minimap_results):
    """
    Usporedba k-mer i minimap2 metode.
    """
    print_section("4) USPOREDBA METODA")

    kmer_accuracy = compute_accuracy(kmer_results)
    minimap_accuracy = compute_accuracy(minimap_results)

    comparison = compare_methods(kmer_results, minimap_results)
    agreement = compute_agreement(comparison)

    save_results_csv(comparison, COMPARISON_CSV)

    print(f"[INFO] Spremljena usporedba: {COMPARISON_CSV}")
    print(f"[RESULT] k-mer accuracy:    {kmer_accuracy:.4f}")
    print(f"[RESULT] minimap2 accuracy: {minimap_accuracy:.4f}")
    print(f"[RESULT] slaganje metoda:   {agreement:.4f}")

    print("\n[INFO] Primjer nekoliko redaka usporedbe:")
    for row in comparison[:10]:
        print(
            f"Read: {row['read_id']}\n"
            f"  True: {row['true_label']}\n"
            f"  k-mer: {row['kmer_predicted']} (correct={row['kmer_correct']})\n"
            f"  minimap2: {row['minimap_predicted']} (correct={row['minimap_correct']})\n"
            f"  agree: {row['methods_agree']}\n"
        )

    return comparison


def run_plots(kmer_results, minimap_results, comparison_rows):
    """
    Reportovi i grafovi.
    """
    print_section("5) REPORTOVI I PLOTOVI")

    print("[INFO] Računam classification report za k-mer...")
    kmer_report = compute_classification_report(kmer_results)
    kmer_report_text = format_classification_report(kmer_report)

    print("[INFO] Računam classification report za minimap2...")
    minimap_report = compute_classification_report(minimap_results)
    minimap_report_text = format_classification_report(minimap_report)

    save_text(kmer_report_text, PLOTS_DIR / "kmer_classification_report.txt")
    save_text(minimap_report_text, PLOTS_DIR / "minimap2_classification_report.txt")

    save_report_csv(kmer_report, PLOTS_DIR / "kmer_classification_report.csv")
    save_report_csv(minimap_report, PLOTS_DIR / "minimap2_classification_report.csv")

    print("[INFO] Crtam grafove...")
    plot_accuracy_comparison(
        kmer_results,
        minimap_results,
        PLOTS_DIR / "accuracy_comparison.png"
    )
    plot_assignments_bar(
        kmer_results,
        "k-mer: broj readova po predikciji",
        PLOTS_DIR / "kmer_assignments_bar.png"
    )
    plot_assignments_bar(
        minimap_results,
        "minimap2: broj readova po predikciji",
        PLOTS_DIR / "minimap_assignments_bar.png"
    )
    plot_confusion_matrix(
        kmer_results,
        "k-mer confusion matrix",
        PLOTS_DIR / "kmer_confusion_matrix.png"
    )
    plot_confusion_matrix(
        minimap_results,
        "minimap2 confusion matrix",
        PLOTS_DIR / "minimap_confusion_matrix.png"
    )
    plot_minimap_mapping_status(
        comparison_rows,
        PLOTS_DIR / "minimap_mapping_status.png"
    )

    print(f"[INFO] Sve spremljeno u: {PLOTS_DIR}")

    print("\n[INFO] k-mer classification report:\n")
    print(kmer_report_text)

    print("\n[INFO] minimap2 classification report:\n")
    print(minimap_report_text)


def save_summary_csv(kmer_results, minimap_results, comparison_rows) -> None:
    """
    Mali sažetak u jedan CSV da imaš odmah glavne metrike na jednom mjestu.
    """
    summary_csv = RESULTS_DIR / "summary_metrics.csv"

    kmer_accuracy = compute_accuracy(kmer_results)
    minimap_accuracy = compute_accuracy(minimap_results)
    agreement = compute_agreement(comparison_rows)

    rows = [
        {"metric": "kmer_accuracy", "value": f"{kmer_accuracy:.6f}"},
        {"metric": "minimap2_accuracy", "value": f"{minimap_accuracy:.6f}"},
        {"metric": "methods_agreement", "value": f"{agreement:.6f}"},
        {"metric": "kmer_n_results", "value": str(len(kmer_results))},
        {"metric": "minimap_n_results", "value": str(len(minimap_results))},
        {"metric": "comparison_n_rows", "value": str(len(comparison_rows))},
    ]

    with open(summary_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] Sažetak spremljen u: {summary_csv}")


# =========================================================
# MAIN
# =========================================================

def main():
    ensure_directories()

    run_preprocessing()
    kmer_results = run_kmer_classification()
    minimap_results = run_minimap2_baseline()
    comparison_rows = run_method_comparison(kmer_results, minimap_results)
    run_plots(kmer_results, minimap_results, comparison_rows)
    save_summary_csv(kmer_results, minimap_results, comparison_rows)

    print_section("GOTOVO")
    print("[INFO] Cijeli pipeline je uspješno završen.")
    print(f"[INFO] Mixed reads:        {MIXED_READS_PATH}")
    print(f"[INFO] k-mer CSV:          {KMER_ASSIGNMENTS_CSV}")
    print(f"[INFO] minimap2 CSV:      {MINIMAP_ASSIGNMENTS_CSV}")
    print(f"[INFO] comparison CSV:    {COMPARISON_CSV}")
    print(f"[INFO] plots dir:         {PLOTS_DIR}")


if __name__ == "__main__":
    main()