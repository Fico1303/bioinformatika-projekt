# Kod napisao: Filip Paić

from typing import Dict, Tuple, List, Optional
from collections import defaultdict

import numpy as np


def compute_accuracy(results: List[Dict[str, object]]) -> float:
    """
    Izračunaj accuracy iz liste rezultata.

    Očekuje da svaki red ima:
    - true_label
    - predicted_label

    Ili, ako postoji is_correct, koristi njega.
    """
    if not results:
        return 0.0

    correct = 0

    for row in results:
        if "is_correct" in row:
            value = row["is_correct"]

            if isinstance(value, bool):
                is_correct = value
            else:
                is_correct = str(value).strip().lower() in {"true", "1", "yes", "da"}

            if is_correct:
                correct += 1

        else:
            if str(row.get("true_label", "")) == str(row.get("predicted_label", "")):
                correct += 1

    return correct / len(results)


def compute_agreement(rows: List[Dict[str, object]]) -> float:
    """
    Izračunaj agreement između dvije metode.

    Očekuje jedan od ova dva formata:

    1) row["methods_agree"]
    2) row["kmer_predicted"] i row["minimap_predicted"]
    """
    if not rows:
        return 0.0

    agree = 0

    for row in rows:
        if "methods_agree" in row:
            value = row["methods_agree"]

            if isinstance(value, bool):
                methods_agree = value
            else:
                methods_agree = str(value).strip().lower() in {"true", "1", "yes", "da"}

            if methods_agree:
                agree += 1

        else:
            kmer_predicted = str(row.get("kmer_predicted", ""))
            minimap_predicted = str(row.get("minimap_predicted", ""))

            if kmer_predicted == minimap_predicted:
                agree += 1

    return agree / len(rows)


def dicts_to_vectors(
    dict1: Dict[str, float],
    dict2: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pretvori dva rječnika u dva vektora nad istim skupom ključeva.
    """
    all_keys = sorted(set(dict1.keys()) | set(dict2.keys()))

    vec1 = np.array(
        [dict1.get(key, 0.0) for key in all_keys],
        dtype=float,
    )

    vec2 = np.array(
        [dict2.get(key, 0.0) for key in all_keys],
        dtype=float,
    )

    return vec1, vec2


def cosine_similarity_dicts(
    dict1: Dict[str, float],
    dict2: Dict[str, float],
) -> float:
    """
    Cosine similarity između dva rječnika frekvencija.
    """
    vec1, vec2 = dicts_to_vectors(dict1, dict2)

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def get_labels_from_results(results: List[Dict[str, object]]) -> List[str]:
    """
    Vrati sortirani popis svih labela iz true_label i predicted_label stupaca.
    """
    labels = set()

    for row in results:
        true_label = str(row.get("true_label", "")).strip()
        predicted_label = str(row.get("predicted_label", "")).strip()

        if true_label:
            labels.add(true_label)

        if predicted_label:
            labels.add(predicted_label)

    return sorted(labels)


def compute_classification_report(
    results: List[Dict[str, object]],
) -> Dict[str, Dict[str, float]]:
    """
    Ručno izračunaj precision, recall, f1 i support po klasi.

    Također vraća:
    - accuracy
    - macro avg
    - weighted avg
    """
    if not results:
        return {}

    labels = get_labels_from_results(results)

    stats = {
        label: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "support": 0,
        }
        for label in labels
    }

    total_correct = 0
    total = len(results)

    for row in results:
        true_label = str(row["true_label"])
        predicted_label = str(row["predicted_label"])

        stats[true_label]["support"] += 1

        if true_label == predicted_label:
            stats[true_label]["tp"] += 1
            total_correct += 1
        else:
            stats[predicted_label]["fp"] += 1
            stats[true_label]["fn"] += 1

    report = {}

    for label in labels:
        tp = stats[label]["tp"]
        fp = stats[label]["fp"]
        fn = stats[label]["fn"]
        support = stats[label]["support"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        report[label] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "support": float(support),
        }

    accuracy = total_correct / total if total > 0 else 0.0

    macro_precision = (
        float(np.mean([report[label]["precision"] for label in labels]))
        if labels
        else 0.0
    )

    macro_recall = (
        float(np.mean([report[label]["recall"] for label in labels]))
        if labels
        else 0.0
    )

    macro_f1 = (
        float(np.mean([report[label]["f1-score"] for label in labels]))
        if labels
        else 0.0
    )

    total_support = sum(report[label]["support"] for label in labels)

    weighted_precision = (
        sum(report[label]["precision"] * report[label]["support"] for label in labels)
        / total_support
        if total_support > 0
        else 0.0
    )

    weighted_recall = (
        sum(report[label]["recall"] * report[label]["support"] for label in labels)
        / total_support
        if total_support > 0
        else 0.0
    )

    weighted_f1 = (
        sum(report[label]["f1-score"] * report[label]["support"] for label in labels)
        / total_support
        if total_support > 0
        else 0.0
    )

    report["accuracy"] = {
        "precision": accuracy,
        "recall": accuracy,
        "f1-score": accuracy,
        "support": float(total),
    }

    report["macro avg"] = {
        "precision": macro_precision,
        "recall": macro_recall,
        "f1-score": macro_f1,
        "support": float(total_support),
    }

    report["weighted avg"] = {
        "precision": weighted_precision,
        "recall": weighted_recall,
        "f1-score": weighted_f1,
        "support": float(total_support),
    }

    return report


def format_classification_report(report: Dict[str, Dict[str, float]]) -> str:
    """
    Pretvori classification report u lijepo formatiran tekst.
    """
    if not report:
        return "Nema rezultata."

    lines = []

    header = (
        f"{'label':<16}"
        f"{'precision':>12}"
        f"{'recall':>10}"
        f"{'f1-score':>12}"
        f"{'support':>10}"
    )

    lines.append(header)
    lines.append("-" * len(header))

    class_labels = [
        label
        for label in report.keys()
        if label not in {"accuracy", "macro avg", "weighted avg"}
    ]

    for label in class_labels + ["accuracy", "macro avg", "weighted avg"]:
        if label not in report:
            continue

        row = report[label]

        lines.append(
            f"{label:<16}"
            f"{row['precision']:>12.4f}"
            f"{row['recall']:>10.4f}"
            f"{row['f1-score']:>12.4f}"
            f"{int(row['support']):>10}"
        )

    return "\n".join(lines)


def compute_confusion_matrix(
    results: List[Dict[str, object]],
    labels: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Izračunaj confusion matrix.

    Redovi = true_label
    Stupci = predicted_label
    """
    if not results:
        return np.zeros((0, 0), dtype=int), []

    if labels is None:
        labels = get_labels_from_results(results)

    label_to_idx = {
        label: i
        for i, label in enumerate(labels)
    }

    matrix = np.zeros((len(labels), len(labels)), dtype=int)

    for row in results:
        true_label = str(row["true_label"])
        predicted_label = str(row["predicted_label"])

        if true_label not in label_to_idx or predicted_label not in label_to_idx:
            continue

        i = label_to_idx[true_label]
        j = label_to_idx[predicted_label]

        matrix[i, j] += 1

    return matrix, labels


def count_predictions_by_label(results: List[Dict[str, object]]) -> Dict[str, int]:
    """
    Prebroji koliko je readova predviđeno po labeli.
    """
    counts = defaultdict(int)

    for row in results:
        predicted_label = str(row["predicted_label"])
        counts[predicted_label] += 1

    return dict(sorted(counts.items()))


def count_true_labels(results: List[Dict[str, object]]) -> Dict[str, int]:
    """
    Prebroji koliko readova stvarno pripada kojoj labeli.
    """
    counts = defaultdict(int)

    for row in results:
        true_label = str(row["true_label"])
        counts[true_label] += 1

    return dict(sorted(counts.items()))# Kod napisao: Filip Paić

from typing import Dict, Tuple, List, Optional
from collections import defaultdict

import numpy as np


def compute_accuracy(results: List[Dict[str, object]]) -> float:
    """
    Izračunaj accuracy iz liste rezultata.

    Očekuje da svaki red ima:
    - true_label
    - predicted_label

    Ili, ako postoji is_correct, koristi njega.
    """
    if not results:
        return 0.0

    correct = 0

    for row in results:
        if "is_correct" in row:
            value = row["is_correct"]

            if isinstance(value, bool):
                is_correct = value
            else:
                is_correct = str(value).strip().lower() in {"true", "1", "yes", "da"}

            if is_correct:
                correct += 1

        else:
            if str(row.get("true_label", "")) == str(row.get("predicted_label", "")):
                correct += 1

    return correct / len(results)


def compute_agreement(rows: List[Dict[str, object]]) -> float:
    """
    Izračunaj agreement između dvije metode.

    Očekuje jedan od ova dva formata:

    1) row["methods_agree"]
    2) row["kmer_predicted"] i row["minimap_predicted"]
    """
    if not rows:
        return 0.0

    agree = 0

    for row in rows:
        if "methods_agree" in row:
            value = row["methods_agree"]

            if isinstance(value, bool):
                methods_agree = value
            else:
                methods_agree = str(value).strip().lower() in {"true", "1", "yes", "da"}

            if methods_agree:
                agree += 1

        else:
            kmer_predicted = str(row.get("kmer_predicted", ""))
            minimap_predicted = str(row.get("minimap_predicted", ""))

            if kmer_predicted == minimap_predicted:
                agree += 1

    return agree / len(rows)


def dicts_to_vectors(
    dict1: Dict[str, float],
    dict2: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pretvori dva rječnika u dva vektora nad istim skupom ključeva.
    """
    all_keys = sorted(set(dict1.keys()) | set(dict2.keys()))

    vec1 = np.array(
        [dict1.get(key, 0.0) for key in all_keys],
        dtype=float,
    )

    vec2 = np.array(
        [dict2.get(key, 0.0) for key in all_keys],
        dtype=float,
    )

    return vec1, vec2


def cosine_similarity_dicts(
    dict1: Dict[str, float],
    dict2: Dict[str, float],
) -> float:
    """
    Cosine similarity između dva rječnika frekvencija.
    """
    vec1, vec2 = dicts_to_vectors(dict1, dict2)

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def get_labels_from_results(results: List[Dict[str, object]]) -> List[str]:
    """
    Vrati sortirani popis svih labela iz true_label i predicted_label stupaca.
    """
    labels = set()

    for row in results:
        true_label = str(row.get("true_label", "")).strip()
        predicted_label = str(row.get("predicted_label", "")).strip()

        if true_label:
            labels.add(true_label)

        if predicted_label:
            labels.add(predicted_label)

    return sorted(labels)


def compute_classification_report(
    results: List[Dict[str, object]],
) -> Dict[str, Dict[str, float]]:
    """
    Ručno izračunaj precision, recall, f1 i support po klasi.

    Također vraća:
    - accuracy
    - macro avg
    - weighted avg
    """
    if not results:
        return {}

    labels = get_labels_from_results(results)

    stats = {
        label: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "support": 0,
        }
        for label in labels
    }

    total_correct = 0
    total = len(results)

    for row in results:
        true_label = str(row["true_label"])
        predicted_label = str(row["predicted_label"])

        stats[true_label]["support"] += 1

        if true_label == predicted_label:
            stats[true_label]["tp"] += 1
            total_correct += 1
        else:
            stats[predicted_label]["fp"] += 1
            stats[true_label]["fn"] += 1

    report = {}

    for label in labels:
        tp = stats[label]["tp"]
        fp = stats[label]["fp"]
        fn = stats[label]["fn"]
        support = stats[label]["support"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        report[label] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "support": float(support),
        }

    accuracy = total_correct / total if total > 0 else 0.0

    macro_precision = (
        float(np.mean([report[label]["precision"] for label in labels]))
        if labels
        else 0.0
    )

    macro_recall = (
        float(np.mean([report[label]["recall"] for label in labels]))
        if labels
        else 0.0
    )

    macro_f1 = (
        float(np.mean([report[label]["f1-score"] for label in labels]))
        if labels
        else 0.0
    )

    total_support = sum(report[label]["support"] for label in labels)

    weighted_precision = (
        sum(report[label]["precision"] * report[label]["support"] for label in labels)
        / total_support
        if total_support > 0
        else 0.0
    )

    weighted_recall = (
        sum(report[label]["recall"] * report[label]["support"] for label in labels)
        / total_support
        if total_support > 0
        else 0.0
    )

    weighted_f1 = (
        sum(report[label]["f1-score"] * report[label]["support"] for label in labels)
        / total_support
        if total_support > 0
        else 0.0
    )

    report["accuracy"] = {
        "precision": accuracy,
        "recall": accuracy,
        "f1-score": accuracy,
        "support": float(total),
    }

    report["macro avg"] = {
        "precision": macro_precision,
        "recall": macro_recall,
        "f1-score": macro_f1,
        "support": float(total_support),
    }

    report["weighted avg"] = {
        "precision": weighted_precision,
        "recall": weighted_recall,
        "f1-score": weighted_f1,
        "support": float(total_support),
    }

    return report


def format_classification_report(report: Dict[str, Dict[str, float]]) -> str:
    """
    Pretvori classification report u lijepo formatiran tekst.
    """
    if not report:
        return "Nema rezultata."

    lines = []

    header = (
        f"{'label':<16}"
        f"{'precision':>12}"
        f"{'recall':>10}"
        f"{'f1-score':>12}"
        f"{'support':>10}"
    )

    lines.append(header)
    lines.append("-" * len(header))

    class_labels = [
        label
        for label in report.keys()
        if label not in {"accuracy", "macro avg", "weighted avg"}
    ]

    for label in class_labels + ["accuracy", "macro avg", "weighted avg"]:
        if label not in report:
            continue

        row = report[label]

        lines.append(
            f"{label:<16}"
            f"{row['precision']:>12.4f}"
            f"{row['recall']:>10.4f}"
            f"{row['f1-score']:>12.4f}"
            f"{int(row['support']):>10}"
        )

    return "\n".join(lines)


def compute_confusion_matrix(
    results: List[Dict[str, object]],
    labels: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Izračunaj confusion matrix.

    Redovi = true_label
    Stupci = predicted_label
    """
    if not results:
        return np.zeros((0, 0), dtype=int), []

    if labels is None:
        labels = get_labels_from_results(results)

    label_to_idx = {
        label: i
        for i, label in enumerate(labels)
    }

    matrix = np.zeros((len(labels), len(labels)), dtype=int)

    for row in results:
        true_label = str(row["true_label"])
        predicted_label = str(row["predicted_label"])

        if true_label not in label_to_idx or predicted_label not in label_to_idx:
            continue

        i = label_to_idx[true_label]
        j = label_to_idx[predicted_label]

        matrix[i, j] += 1

    return matrix, labels


def count_predictions_by_label(results: List[Dict[str, object]]) -> Dict[str, int]:
    """
    Prebroji koliko je readova predviđeno po labeli.
    """
    counts = defaultdict(int)

    for row in results:
        predicted_label = str(row["predicted_label"])
        counts[predicted_label] += 1

    return dict(sorted(counts.items()))


def count_true_labels(results: List[Dict[str, object]]) -> Dict[str, int]:
    """
    Prebroji koliko readova stvarno pripada kojoj labeli.
    """
    counts = defaultdict(int)

    for row in results:
        true_label = str(row["true_label"])
        counts[true_label] += 1

    return dict(sorted(counts.items()))