# Kod napisao: Filip Paić

from pathlib import Path
import subprocess
from typing import Dict, Any

from Bio import SeqIO

from io_utils import (
    open_textfile,
    find_reference_files,
    ensure_directories,
    print_section,
)

from evaluation import (
    add_true_labels,
    add_correct_flag,
    save_results_csv,
)

from metrics import (
    compute_accuracy,
    count_predictions_by_label,
)


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
READS_PATH = BASE_DIR / "data" / "processed" / "mixed_metagenome.fastq"

OUTPUT_DIR = BASE_DIR / "data" / "processed" / "minimap"
COMBINED_REFERENCE_PATH = OUTPUT_DIR / "all_references.fasta"
PAF_PATH = OUTPUT_DIR / "mappings.paf"

RESULTS_DIR = BASE_DIR / "data" / "results"
MINIMAP_ASSIGNMENTS_CSV = RESULTS_DIR / "minimap2_assignments.csv"

MINIMAP2_PRESET = "map-ont"


def build_combined_reference(reference_files: Dict[str, str], output_fasta: Path) -> Path:
    """
    Spoji sve referentne genome u jedan FASTA.

    Header svakog contiga označi prefiksom bakterije:

        >bacterium1|original_header

    Tako se kasnije iz PAF target_name može dobiti predicted_label.
    """
    ensure_directories(output_fasta.parent)

    written = 0

    with open(output_fasta, "w", encoding="utf-8") as out_handle:
        for label, fasta_path in reference_files.items():
            with open_textfile(fasta_path) as in_handle:
                for record in SeqIO.parse(in_handle, "fasta"):
                    record.id = f"{label}|{record.id}"
                    record.description = ""
                    SeqIO.write(record, out_handle, "fasta")
                    written += 1

    if written == 0:
        raise ValueError("Nijedna referentna sekvenca nije upisana u combined FASTA.")

    return output_fasta


def run_minimap2(
    reference_fasta: Path,
    reads_path: Path,
    paf_path: Path,
    preset: str = "map-ont",
) -> Path:
    """
    Pokreni minimap2 i spremi izlaz u PAF.
    """
    ensure_directories(paf_path.parent)

    cmd = [
        "minimap2",
        "-x",
        preset,
        str(reference_fasta),
        str(reads_path),
    ]

    print("[INFO] Pokrećem minimap2...")
    print("[CMD]", " ".join(cmd))

    with open(paf_path, "w", encoding="utf-8") as out_handle:
        result = subprocess.run(
            cmd,
            stdout=out_handle,
            stderr=subprocess.PIPE,
            text=True,
        )

    if result.returncode != 0:
        raise RuntimeError(
            "minimap2 nije uspješno završio.\n"
            f"Return code: {result.returncode}\n"
            f"Stderr:\n{result.stderr}"
        )

    print(f"[INFO] PAF spremljen u: {paf_path}")
    return paf_path


def parse_paf_best_hits(paf_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Pročitaj PAF i za svaki read zadrži najbolji hit.

    Kriterij:
    1) veći MAPQ
    2) više matching baza
    3) veći alignment block
    4) veći identity
    """
    best_hits = {}

    with open(paf_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()

            if not line:
                continue

            parts = line.split("\t")

            if len(parts) < 12:
                continue

            query_name = parts[0]
            target_name = parts[5]
            n_match = int(parts[9])
            aln_block = int(parts[10])
            mapq = int(parts[11])

            identity = n_match / aln_block if aln_block > 0 else 0.0
            predicted_label = target_name.split("|", 1)[0]

            candidate = {
                "read_id": query_name,
                "target_name": target_name,
                "predicted_label": predicted_label,
                "n_match": n_match,
                "aln_block": aln_block,
                "identity": round(identity, 6),
                "mapq": mapq,
            }

            rank = (mapq, n_match, aln_block, identity)

            if query_name not in best_hits or rank > best_hits[query_name][0]:
                best_hits[query_name] = (rank, candidate)

    return {
        read_id: candidate
        for read_id, (_, candidate) in best_hits.items()
    }


def main() -> None:
    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError(f"Ne postoji direktorij s referencama: {RAW_DATA_DIR}")

    if not READS_PATH.exists():
        raise FileNotFoundError(f"Ne postoji reads datoteka: {READS_PATH}")

    ensure_directories(OUTPUT_DIR, RESULTS_DIR)

    print_section("Traženje referenci")

    reference_files = find_reference_files(RAW_DATA_DIR)

    print("[INFO] Nađene reference:")
    for label, path in reference_files.items():
        print(f"  - {label}: {path}")

    print_section("Izrada combined reference FASTA")

    build_combined_reference(reference_files, COMBINED_REFERENCE_PATH)

    print(f"[INFO] Combined FASTA: {COMBINED_REFERENCE_PATH}")

    print_section("Minimap2 mapiranje")

    run_minimap2(
        COMBINED_REFERENCE_PATH,
        READS_PATH,
        PAF_PATH,
        preset=MINIMAP2_PRESET,
    )

    print_section("Parsiranje najboljih mapiranja")

    best_hits = parse_paf_best_hits(PAF_PATH)

    if not best_hits:
        raise ValueError("PAF je prazan ili nijedan read nije mapiran.")

    results = list(best_hits.values())
    results = add_true_labels(results)
    results = add_correct_flag(results)

    accuracy = compute_accuracy(results)
    counts = count_predictions_by_label(results)

    save_results_csv(results, MINIMAP_ASSIGNMENTS_CSV)

    print(f"[INFO] Spremljeni minimap assignmenti: {MINIMAP_ASSIGNMENTS_CSV}")

    print("\n[RESULT] Broj readova po genomu:")

    total = 0

    for label, count in counts.items():
        print(f"  - {label}: {count}")
        total += count

    print(f"\n[RESULT] Ukupno klasificiranih readova: {total}")
    print(f"[RESULT] Minimap2 accuracy: {accuracy:.4f}")

    print("\n[INFO] Primjer nekoliko najboljih hitova:")

    for i, row in enumerate(results[:10]):
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


if __name__ == "__main__":
    main()