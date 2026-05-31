# Kod napisao: Filip Paić

from pathlib import Path
import csv
import shutil

from io_utils import (
    find_reference_files,
    iter_fastq_reads,
    ensure_directories,
    print_section,
    save_aggregate_results,
)

from classifier import (
    build_reference_profiles,
    classify_reads_iter,
)

from evaluation import (
    add_true_labels,
    add_correct_flag,
    add_true_label_to_row,
    add_correct_flag_to_row,
    save_results_csv,
    compare_methods,
)

from metrics import (
    compute_accuracy,
    compute_agreement,
    compute_classification_report,
    format_classification_report,
    count_predictions_by_label,
)

from minimap_baseline import (
    build_combined_reference,
    run_minimap2,
    parse_paf_best_hits,
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
# PIPELINE STEPS
# =========================================================

def run_kmer_classification():
    """
    Pokreće k-mer klasifikaciju nad ulaznim FASTQ očitanjima.

    Vraća listu rezultata s predikcijama, stvarnim labelama
    i oznakom točnosti za svaki read.
    """
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
    """
    Pokreće minimap2 baseline metodu nad istim ulaznim FASTQ očitanjima.

    Vraća listu rezultata s najboljim mapiranjem za svaki read.
    """
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
    counts = count_predictions_by_label(results)

    save_results_csv(results, MINIMAP_ASSIGNMENTS_CSV)

    print("[RESULT] Minimap2 broj readova po predikciji:")
    for label, count in counts.items():
        print(f"  - {label}: {count}")

    print(f"[RESULT] Minimap2 broj klasificiranih readova: {len(results)}")
    print(f"[RESULT] Minimap2 accuracy: {accuracy:.4f}")

    return results


def run_method_comparison(kmer_results, minimap_results):
    """
    Uspoređuje rezultate k-mer metode i minimap2 metode.

    Računa accuracy za obje metode, slaganje metoda
    i sprema agregirane CSV sažetke.
    """
    print_section("3) USPOREDBA METODA")

    comparison = compare_methods(kmer_results, minimap_results)

    kmer_accuracy = compute_accuracy(kmer_results)
    minimap_accuracy = compute_accuracy(minimap_results)
    agreement = compute_agreement(comparison)

    print(f"[RESULT] k-mer accuracy:    {kmer_accuracy:.4f}")
    print(f"[RESULT] minimap2 accuracy: {minimap_accuracy:.4f}")
    print(f"[RESULT] slaganje metoda:   {agreement:.4f}")

    save_aggregate_results(
        kmer_results,
        minimap_results,
        comparison,
        SUMMARY_OVERALL_CSV,
        SUMMARY_BY_BACTERIUM_CSV,
    )

    return comparison


def run_reports_and_plots(kmer_results, minimap_results):
    """
    Generira tekstualne reportove, CSV reportove i grafove za obje metode.
    """
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
    """
    Glavna funkcija koja pokreće cijeli pipeline.

    Redoslijed:
    1. priprema direktorija
    2. k-mer klasifikacija
    3. minimap2 baseline
    4. usporedba metoda
    5. reportovi i grafovi
    """
    ensure_directories(
        PROCESSED_DIR,
        RESULTS_DIR,
        PLOTS_DIR,
        MINIMAP_DIR,
    )

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