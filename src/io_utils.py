# Kod napisao: Filip Paić

import gzip
import csv
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union
from collections import defaultdict, Counter

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

from metrics import compute_accuracy, compute_agreement


PathLike = Union[str, Path]


def open_textfile(path: PathLike):
    """
    Otvara običan ili gz-komprimirani tekstualni file.

    Podržava:
    - .txt, .fasta, .fa, .fna, .fastq, .fq
    - .gz varijante tih datoteka
    """
    path = Path(path)

    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rt")

    return open(path, "r", encoding="utf-8")


def load_all_fasta_sequences(path: PathLike) -> str:
    """
    Učita sve FASTA zapise iz filea i spoji ih u jednu sekvencu.
    Korisno ako referenca ima više contigova.
    """
    sequences: List[str] = []

    with open_textfile(path) as handle:
        for record in SeqIO.parse(handle, "fasta"):
            sequences.append(str(record.seq))

    if not sequences:
        raise ValueError(f"Nema FASTA sekvenci u fileu: {path}")

    return "".join(sequences)


def load_first_fastq_read(path: PathLike) -> SeqRecord:
    """
    Učita prvo očitanje iz FASTQ filea.
    """
    with open_textfile(path) as handle:
        return next(SeqIO.parse(handle, "fastq"))


def load_fastq_reads(path: PathLike, limit: Optional[int] = None) -> List[SeqRecord]:
    """
    Učita prvih 'limit' očitanja iz FASTQ filea.

    Ako je limit=None, učita sva očitanja.
    """
    reads: List[SeqRecord] = []

    with open_textfile(path) as handle:
        for i, record in enumerate(SeqIO.parse(handle, "fastq")):
            reads.append(record)

            if limit is not None and i + 1 >= limit:
                break

    return reads


def iter_fastq_reads(path: PathLike, limit: Optional[int] = None) -> Iterator[SeqRecord]:
    """
    Streaming učitavanje FASTQ readova.

    Ne drži sve readove u memoriji.
    """
    count = 0

    with open_textfile(path) as handle:
        for record in SeqIO.parse(handle, "fastq"):
            yield record
            count += 1

            if limit is not None and count >= limit:
                break


def find_reference_files(raw_data_dir: PathLike = "data/raw") -> Dict[str, str]:
    """
    Traži referencu u svakom bacterium folderu i vraća:

    {
        "bacterium1": "data/raw/bacterium1/reference.fasta",
        ...
    }

    Prepoznaje ekstenzije:
    - .fasta
    - .fa
    - .fna
    - .fasta.gz
    - .fa.gz
    - .fna.gz
    """
    raw_path = Path(raw_data_dir)

    if not raw_path.exists():
        raise FileNotFoundError(f"Ne postoji direktorij: {raw_data_dir}")

    allowed_suffixes = {
        ".fasta",
        ".fa",
        ".fna",
        ".fasta.gz",
        ".fa.gz",
        ".fna.gz",
    }

    references: Dict[str, str] = {}

    for bacterium_dir in sorted(p for p in raw_path.iterdir() if p.is_dir()):
        candidates = []

        for file_path in sorted(bacterium_dir.iterdir()):
            name = file_path.name.lower()

            if any(name.endswith(suffix) for suffix in allowed_suffixes):
                candidates.append(file_path)

        if not candidates:
            continue

        if len(candidates) > 1:
            print(
                f"[WARN] Više FASTA fileova u {bacterium_dir}. "
                f"Uzimam prvi: {candidates[0].name}"
            )

        references[bacterium_dir.name] = str(candidates[0])

    if not references:
        raise ValueError(f"Nisam našao nijednu referencu u {raw_data_dir}")

    return references


def find_fastq_files(raw_data_dir: PathLike = "data/raw") -> Dict[str, List[str]]:
    """
    Automatski pronađe sve FASTQ fileove u svakom folderu bakterije.

    Vraća:

    {
        "Escherichia_coli": [
            "data/raw/Escherichia_coli/file1.fastq.gz",
            "data/raw/Escherichia_coli/file2.fastq.gz"
        ],
        "Staphylococcus_aureus": [
            "data/raw/Staphylococcus_aureus/reads.fastq"
        ]
    }
    """
    raw_path = Path(raw_data_dir)

    if not raw_path.exists():
        raise FileNotFoundError(f"Ne postoji direktorij: {raw_data_dir}")

    allowed_suffixes = {
        ".fastq",
        ".fq",
        ".fastq.gz",
        ".fq.gz",
    }

    fastq_files: Dict[str, List[str]] = {}

    for bacterium_dir in sorted(p for p in raw_path.iterdir() if p.is_dir()):
        candidates: List[str] = []

        for file_path in sorted(bacterium_dir.iterdir()):
            name = file_path.name.lower()

            if any(name.endswith(suffix) for suffix in allowed_suffixes):
                candidates.append(str(file_path))

        if not candidates:
            continue

        fastq_files[bacterium_dir.name] = candidates

    if not fastq_files:
        raise ValueError(f"Nisam našao nijedan FASTQ u {raw_data_dir}")

    return fastq_files


def ensure_directories(*directories: PathLike) -> None:
    """
    Osigurava da svi potrebni direktoriji postoje.

    Ako direktorij ne postoji, automatski ga kreira.
    """
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def print_section(title: str) -> None:
    """
    Ispiši naslov sekcije u konzolu radi preglednijeg ispisa pipelinea.
    """
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def as_bool(value) -> bool:
    """
    Pretvori vrijednost u boolean.

    Podržava:
    - stvarne bool vrijednosti
    - tekstualne vrijednosti poput true, 1, yes i da
    """
    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in {"true", "1", "yes", "da"}


def format_counts(counter: Counter) -> str:
    """
    Formatira Counter u tekstualni oblik pogodan za spremanje u CSV.

    Primjer:

        label1=10; label2=5
    """
    return "; ".join(
        f"{label}={count}"
        for label, count in counter.items()
    )


def save_aggregate_results(
    kmer_results,
    minimap_results,
    comparison_rows,
    summary_overall_csv: PathLike,
    summary_by_bacterium_csv: PathLike,
) -> None:
    """
    Sprema agregirane rezultate evaluacije u CSV datoteke.

    Sprema:
    - ukupne metrike za obje metode
    - rezultate grupirane po stvarnoj bakteriji
    """
    summary_overall_csv = Path(summary_overall_csv)
    summary_by_bacterium_csv = Path(summary_by_bacterium_csv)

    kmer_accuracy = compute_accuracy(kmer_results)
    minimap_accuracy = compute_accuracy(minimap_results)
    agreement = compute_agreement(comparison_rows)

    overall_rows = [
        {
            "metric": "kmer_accuracy",
            "value": f"{kmer_accuracy:.6f}",
        },
        {
            "metric": "minimap2_accuracy",
            "value": f"{minimap_accuracy:.6f}",
        },
        {
            "metric": "methods_agreement",
            "value": f"{agreement:.6f}",
        },
        {
            "metric": "kmer_n_results",
            "value": str(len(kmer_results)),
        },
        {
            "metric": "minimap2_n_results",
            "value": str(len(minimap_results)),
        },
        {
            "metric": "comparison_n_rows",
            "value": str(len(comparison_rows)),
        },
    ]

    ensure_directories(summary_overall_csv.parent)

    with open(summary_overall_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(overall_rows)

    by_bacterium = defaultdict(
        lambda: {
            "n_reads": 0,
            "kmer_correct": 0,
            "minimap_correct": 0,
            "methods_agree": 0,
            "kmer_predictions": Counter(),
            "minimap_predictions": Counter(),
        }
    )

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

        bacteria_rows.append(
            {
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
            }
        )

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

    ensure_directories(summary_by_bacterium_csv.parent)

    with open(summary_by_bacterium_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(bacteria_rows)

    print(f"[INFO] Ukupni rezultati spremljeni u: {summary_overall_csv}")
    print(f"[INFO] Rezultati po bakteriji spremljeni u: {summary_by_bacterium_csv}")