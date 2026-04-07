# 🧬 Bacterial Genome Analysis Pipeline

Ovaj projekt služi za analizu bakterijskih podataka koristeći FASTA referencu i FASTQ sekvence.

---

## 📁 Struktura podataka

Podaci se organiziraju unutar direktorija:

```
data/raw/{naziv_bakterije}/
```

Za svaku bakteriju potrebno je osigurati:

* **Referentni genom** u `.fasta` formatu
* **Sekvencijske podatke** u `.fastq.gz` formatu

### Primjer:

```
data/raw/e_coli/
├── reference.fasta
├── sample_1.fastq.gz
├── sample_2.fastq.gz
```

---

## 🚀 Pokretanje

Glavna skripta projekta nalazi se u:

```
src/main.py
```

Pokretanje:

```bash
python src/main.py
```

---

## 📦 Instalacija ovisnosti

Sve potrebne biblioteke nalaze se u `requirements.txt`.

Instalacija:

```bash
pip install -r requirements.txt
```

---

## 🧪 Napomena

* Svaka bakterija mora imati vlastiti direktorij unutar `data/raw/`
* Skripta automatski obrađuje dostupne podatke iz tog direktorija
* Preporučuje se korištenje virtualnog okruženja (`venv`)

---

## 📄 Struktura projekta (sažetak)

```
.
├── data/
│   └── raw/
├── src/
│   ├── main.py
│   └── ...
├── requirements.txt
└── README.md
```
