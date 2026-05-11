import gzip
from pathlib import Path
from typing import Dict, Iterator, List, Optional
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord


def open_textfile(path: str):
    """
    Otvara običan ili gz-komprimirani tekstualni file.
    """
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "r", encoding="utf-8")


def load_all_fasta_sequences(path: str) -> str:
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


def load_first_fastq_read(path: str) -> SeqRecord:
    """
    Učita prvo očitanje iz FASTQ filea.
    """
    with open_textfile(path) as handle:
        return next(SeqIO.parse(handle, "fastq"))


def load_fastq_reads(path: str, limit: Optional[int] = None) -> List[SeqRecord]:
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


def iter_fastq_reads(path: str, limit: Optional[int] = None) -> Iterator[SeqRecord]:
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


def find_reference_files(raw_data_dir: str = "data/raw") -> Dict[str, str]:
    """
    Traži referencu u svakom bacterium folderu i vraća:
    {
        "bacterium1": "data/raw/bacterium1/reference.fasta",
        ...
    }

    Prepoznaje ekstenzije:
    .fasta, .fa, .fna i njihove .gz varijante
    """
    raw_path = Path(raw_data_dir)

    if not raw_path.exists():
        raise FileNotFoundError(f"Ne postoji direktorij: {raw_data_dir}")

    allowed_suffixes = {
        ".fasta", ".fa", ".fna",
        ".fasta.gz", ".fa.gz", ".fna.gz",
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
            print(f"[WARN] Više FASTA fileova u {bacterium_dir}. Uzimam prvi: {candidates[0].name}")

        references[bacterium_dir.name] = str(candidates[0])

    if not references:
        raise ValueError(f"Nisam našao nijednu referencu u {raw_data_dir}")

    return references


def find_fastq_files(raw_data_dir: str = "data/raw") -> Dict[str, List[str]]:
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
        ".fastq", ".fq",
        ".fastq.gz", ".fq.gz",
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