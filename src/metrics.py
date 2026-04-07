import numpy as np
from typing import Dict, Tuple, List
from collections import defaultdict


def dicts_to_vectors(
    dict1: Dict[str, float],
    dict2: Dict[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pretvori dva rječnika u dva vektora nad istim skupom ključeva.
    """
    all_keys = sorted(set(dict1.keys()) | set(dict2.keys()))

    vec1 = np.array([dict1.get(key, 0.0) for key in all_keys], dtype=float)
    vec2 = np.array([dict2.get(key, 0.0) for key in all_keys], dtype=float)

    return vec1, vec2


def cosine_similarity_dicts(
    dict1: Dict[str, float],
    dict2: Dict[str, float]
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
    Vrati sortirani popis svih labela iz true i predicted stupaca.
    """
    labels = set()

    for row in results:
        true_label = str(row.get("true_label", ""))
        pred_label = str(row.get("predicted_label", ""))

        if true_label:
            labels.add(true_label)
        if pred_label:
            labels.add(pred_label)

    return sorted(labels)


def compute_classification_report(results: List[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    """
    Ručno izračunaj precision, recall, f1 i support po klasi.
    Također vrati macro avg, weighted avg i accuracy.
    """
    if not results:
        return {}

    labels = get_labels_from_results(results)
    stats = {label: {"tp": 0, "fp": 0, "fn": 0, "support": 0} for label in labels}

    total_correct = 0
    total = len(results)

    for row in results:
        true_label = str(row["true_label"])
        pred_label = str(row["predicted_label"])

        stats[true_label]["support"] += 1

        if true_label == pred_label:
            stats[true_label]["tp"] += 1
            total_correct += 1
        else:
            stats[pred_label]["fp"] += 1
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
            if (precision + recall) > 0 else 0.0
        )

        report[label] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "support": float(support),
        }

    accuracy = total_correct / total if total > 0 else 0.0

    macro_precision = np.mean([report[label]["precision"] for label in labels]) if labels else 0.0
    macro_recall = np.mean([report[label]["recall"] for label in labels]) if labels else 0.0
    macro_f1 = np.mean([report[label]["f1-score"] for label in labels]) if labels else 0.0

    total_support = sum(report[label]["support"] for label in labels)

    weighted_precision = (
        sum(report[label]["precision"] * report[label]["support"] for label in labels) / total_support
        if total_support > 0 else 0.0
    )
    weighted_recall = (
        sum(report[label]["recall"] * report[label]["support"] for label in labels) / total_support
        if total_support > 0 else 0.0
    )
    weighted_f1 = (
        sum(report[label]["f1-score"] * report[label]["support"] for label in labels) / total_support
        if total_support > 0 else 0.0
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
    header = f"{'label':<16}{'precision':>12}{'recall':>10}{'f1-score':>12}{'support':>10}"
    lines.append(header)
    lines.append("-" * len(header))

    preferred_order = [
        label for label in report.keys()
        if label not in {"accuracy", "macro avg", "weighted avg"}
    ]

    for label in preferred_order + ["accuracy", "macro avg", "weighted avg"]:
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
    labels: List[str] | None = None
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

    label_to_idx = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)

    for row in results:
        true_label = str(row["true_label"])
        pred_label = str(row["predicted_label"])

        if true_label not in label_to_idx or pred_label not in label_to_idx:
            continue

        i = label_to_idx[true_label]
        j = label_to_idx[pred_label]
        matrix[i, j] += 1

    return matrix, labels


def count_predictions_by_label(results: List[Dict[str, object]]) -> Dict[str, int]:
    """
    Prebroji koliko je readova predviđeno po labeli.
    """
    counts = defaultdict(int)
    for row in results:
        counts[str(row["predicted_label"])] += 1
    return dict(sorted(counts.items()))