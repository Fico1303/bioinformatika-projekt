from pathlib import Path
import subprocess
from collections import defaultdict

from Bio import SeqIO

from io_utils import find_reference_files
from evaluation import add_true_labels, add_correct_flag, compute_accuracy, save_results_csv


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
READS_PATH = BASE_DIR / "data" / "processed" / "mixed_metagenome.fastq"

OUTPUT_DIR = BASE_DIR / "data" / "processed" / "minimap"
COMBINED_REFERENCE_PATH = OUTPUT_DIR / "all_references.fasta"
PAF_PATH = OUTPUT_DIR / "mappings.paf"

RESULTS_DIR = BASE_DIR / "data" / "results"
MINIMAP_ASSIGNMENTS_CSV = RESULTS_DIR / "minimap2_assignments.csv"

MINIMAP2_PRESET = "map-ont"


def build_combined_reference(reference_files, output_fasta):
    """
    Spoji sve referentne genome u jedan FASTA.
    Header svakog contiga označi prefiksom bakterije:
        >bacterium1|original_header
    """
    output_fasta.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with open(output_fasta, "w", encoding="utf-8") as out_handle:
        for label, fasta_path in reference_files.items():
            for record in SeqIO.parse(str(fasta_path), "fasta"):
                record.id = f"{label}|{record.id}"
                record.description = ""
                SeqIO.write(record, out_handle, "fasta")
                written += 1

    if written == 0:
        raise ValueError("Nijedna referentna sekvenca nije upisana u combined FASTA.")

    return output_fasta


def run_minimap2(reference_fasta, reads_path, paf_path, preset="map-ont"):
    """
    Pokreni minimap2 i spremi izlaz u PAF.
    """
    paf_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "minimap2",
        "-x", preset,
        str(reference_fasta),
        str(reads_path),
    ]

    print("[INFO] Pokrećem minimap2...")
    print("[CMD]", " ".join(cmd))

    with open(paf_path, "w", encoding="utf-8") as out_handle:
        result = subprocess.run(cmd, stdout=out_handle, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            "minimap2 nije uspješno završio.\n"
            f"Return code: {result.returncode}\n"
            f"Stderr:\n{result.stderr}"
        )

    print(f"[INFO] PAF spremljen u: {paf_path}")
    return paf_path


def parse_paf_best_hits(paf_path):
    """
    Pročitaj PAF i za svaki read zadrži najbolji hit.

    Kriterij:
    1) veći MAPQ
    2) više matching baza
    3) veći alignment block
    4) veći identity
    """
    best_hits = {}

    with open(paf_path, encoding="utf-8") as handle:
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

            identity = (n_match / aln_block) if aln_block > 0 else 0.0

            candidate = {
                "read_id": query_name,
                "target_name": target_name,
                "predicted_label": target_name.split("|")[0],
                "n_match": n_match,
                "aln_block": aln_block,
                "identity": round(identity, 6),
                "mapq": mapq,
            }

            rank = (mapq, n_match, aln_block, identity)

            if query_name not in best_hits or rank > best_hits[query_name][0]:
                best_hits[query_name] = (rank, candidate)

    return {read_id: data for read_id, (_, data) in best_hits.items()}


def summarize_assignments(rows):
    """
    Prebroji koliko readova je dodijeljeno kojem genomu.
    """
    counts = defaultdict(int)
    for row in rows:
        counts[str(row["predicted_label"])] += 1
    return dict(sorted(counts.items()))


def main():
    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError(f"Ne postoji direktorij s referencama: {RAW_DATA_DIR}")

    if not READS_PATH.exists():
        raise FileNotFoundError(f"Ne postoji reads datoteka: {READS_PATH}")

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
    run_minimap2(COMBINED_REFERENCE_PATH, READS_PATH, PAF_PATH, preset=MINIMAP2_PRESET)

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
    print(f"[INFO] Spremljeni minimap assignmenti: {MINIMAP_ASSIGNMENTS_CSV}")

    print("\n[RESULT] Broj readova po genomu:")
    total = 0
    for label, count in counts.items():
        print(f"  - {label}: {count}")
        total += count

    print(f"\n[RESULT] Ukupno klasificiranih readova: {total}")
    print(f"[RESULT] Minimap2 accuracy: {accuracy:.4f}")

    print("\n[INFO] Primjer nekoliko najboljih hitova:")
    for i, row in enumerate(results):
        if i >= 10:
            break
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