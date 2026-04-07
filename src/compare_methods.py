from pathlib import Path

from evaluation import (
    load_results_csv,
    compare_methods,
    compute_accuracy,
    compute_agreement,
    save_results_csv,
)


BASE_DIR = Path(__file__).resolve().parent.parent
KMER_CSV = BASE_DIR / "data" / "results" / "kmer_assignments.csv"
MINIMAP_CSV = BASE_DIR / "data" / "results" / "minimap2_assignments.csv"
COMPARISON_CSV = BASE_DIR / "data" / "results" / "comparison.csv"


def main():
    print("[INFO] Učitavam k-mer rezultate...")
    kmer_results = load_results_csv(KMER_CSV)

    print("[INFO] Učitavam minimap2 rezultate...")
    minimap_results = load_results_csv(MINIMAP_CSV)

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


if __name__ == "__main__":
    main()