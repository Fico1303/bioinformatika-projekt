# 🧬 Bacterial Genome Analysis Pipeline

Ovaj projekt služi za analizu bakterijskih sekvenci koristeći:

* k-mer klasifikaciju
* minimap2 baseline pristup

Pipeline uspoređuje rezultate obje metode i generira:

* ukupne metrike
* rezultate po bakteriji
* classification reportove
* confusion matrix grafove

---

## 📥 Ulazni podaci

Pipeline koristi jedan FASTQ uzorak:

```text
data/processed/uzorak.fastq
```

Potrebno je uploadati ili kopirati FASTQ datoteku pod nazivom:

```text
uzorak.fastq
```

u direktorij:

```text
data/processed/
```

---

## 🧬 Reference

Referentni genomi moraju biti organizirani po bakterijama unutar:

```text
data/raw/{naziv_bakterije}/
```

Svaka bakterija mora sadržavati FASTA referencu.

### Primjer:

```text
data/raw/e_coli/reference.fasta
data/raw/salmonella/reference.fasta
```

---

## 🚀 Pokretanje

Glavna skripta projekta nalazi se u:

```text
src/main.py
```

Pokretanje:

```bash
python src/main.py
```

---

## 📦 Instalacija ovisnosti

Sve potrebne biblioteke nalaze se u:

```text
requirements.txt
```

Instalacija:

```bash
pip install -r requirements.txt
```

---

## 🧪 Napomene

* Preporučuje se korištenje virtualnog okruženja (`venv`)
* Rezultati se spremaju u `data/results/`
* Grafovi i reportovi spremaju se u `data/results/plots/`
