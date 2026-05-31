# Kod napisao: Domagoj Matković

import gzip
import os
from typing import Dict
from Bio import SeqIO


def open_textfile(path: str, mode: str = "rt"):
    """
    Otvara običan ili gz-komprimirani tekstualni file.
    """
    return gzip.open(path, mode) if path.endswith(".gz") else open(path, mode)


def subsample_reads(input_path, output_path, n_reads=1000, file_format="fastq"):
    """
    Izvuče prvih n_reads sekvenci iz FASTQ/FASTA (.gz ili običnog) filea.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    count = 0

    with open_textfile(input_path, "rt") as infile, open(output_path, "w") as outfile:
        for record in SeqIO.parse(infile, file_format):
            if count >= n_reads:
                break

            SeqIO.write(record, outfile, file_format)
            count += 1

    print(f"[INFO] Spremljeno {count} sekvenci u: {output_path}")

from typing import Dict, List
# modificiraj kasnije
def merge_reads(
    input_files: Dict[str, List[str]],
    output_path: str,
    n_reads_per_file: int = 10,
    file_format: str = "fastq"
):
    """
    Spaja očitanja iz više FASTQ datoteka u jedan izlazni FASTQ.

    Parameters:
        input_files (Dict[str, List[str]]):
            npr.
            {
                "Escherichia_coli": [
                    "data/raw/Escherichia_coli/reads1.fastq.gz",
                    "data/raw/Escherichia_coli/reads2.fastq.gz"
                ],
                "Staphylococcus_aureus": [
                    "data/raw/Staphylococcus_aureus/reads.fastq.gz"
                ]
            }

        output_path (str):
            putanja izlaznog FASTQ filea

        n_reads_per_file (int):
            koliko očitanja uzeti iz svakog filea

        file_format (str):
            "fastq" ili "fasta"
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    total_written = 0

    with open(output_path, "w") as outfile:
        for bacterium_name, input_paths in input_files.items():
            bacterium_total = 0

            for input_path in input_paths:
                count = 0

                with open_textfile(input_path, "rt") as infile:
                    for record in SeqIO.parse(infile, file_format):
                        if count >= n_reads_per_file:
                            break

                        original_id = record.id
                        record.id = f"{bacterium_name}|{original_id}"
                        record.name = record.id
                        record.description = ""

                        SeqIO.write(record, outfile, file_format)

                        count += 1
                        bacterium_total += 1
                        total_written += 1

                print(f"[INFO] {bacterium_name}: dodano {count} očitanja iz {input_path}")

            print(f"[INFO] {bacterium_name}: ukupno dodano {bacterium_total} očitanja")

    print(f"[INFO] Ukupno spremljeno {total_written} očitanja u: {output_path}")