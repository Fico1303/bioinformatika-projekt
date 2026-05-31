# Kod napisao: Domagoj Matković

from pathlib import Path
import csv

import matplotlib.pyplot as plt

from evaluation import load_results_csv
from metrics import (
    compute_classification_report,
    format_classification_report,
    compute_confusion_matrix,
    count_predictions_by_label,
)


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "data" / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

KMER_CSV = RESULTS_DIR / "kmer_assignments.csv"
MINIMAP_CSV = RESULTS_DIR / "minimap2_assignments.csv"


def save_text(text: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(text)


def save_report_csv(report: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["label", "precision", "recall", "f1-score", "support"]

    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for label, row in report.items():
            writer.writerow({
                "label": label,
                "precision": row["precision"],
                "recall": row["recall"],
                "f1-score": row["f1-score"],
                "support": int(row["support"]),
            })


def plot_accuracy_comparison(kmer_results, minimap_results, output_path: Path) -> None:
    from metrics import compute_accuracy

    kmer_acc = compute_accuracy(kmer_results)
    minimap_acc = compute_accuracy(minimap_results)

    labels = ["k-mer", "minimap2"]
    values = [kmer_acc, minimap_acc]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, values)
    plt.ylim(0, 1)

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.3f}",
            ha="center"
        )

    plt.ylabel("Accuracy")
    plt.title("Usporedba točnosti metoda")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_assignments_bar(results, title: str, output_path: Path) -> None:
    counts = count_predictions_by_label(results)

    labels = list(counts.keys())
    values = list(counts.values())

    plt.figure(figsize=(8, 4))
    bars = plt.bar(labels, values)

    max_value = max(values) if values else 1

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + max_value * 0.01,
            str(value),
            ha="center"
        )

    plt.ylabel("Broj readova")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_confusion_matrix(results, title: str, output_path: Path) -> None:
    matrix, labels = compute_confusion_matrix(results)

    plt.figure(figsize=(7, 6))
    im = plt.imshow(matrix)
    plt.colorbar(im)

    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, str(matrix[i, j]), ha="center", va="center")

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("[INFO] Učitavam CSV rezultate...")
    kmer_results = load_results_csv(KMER_CSV)
    minimap_results = load_results_csv(MINIMAP_CSV)

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
        PLOTS_DIR / "minimap2_assignments_bar.png"
    )

    plot_confusion_matrix(
        kmer_results,
        "k-mer confusion matrix",
        PLOTS_DIR / "kmer_confusion_matrix.png"
    )

    plot_confusion_matrix(
        minimap_results,
        "minimap2 confusion matrix",
        PLOTS_DIR / "minimap2_confusion_matrix.png"
    )

    print(f"[INFO] Gotovo. Spremljeno u: {PLOTS_DIR}")


if __name__ == "__main__":
    main()