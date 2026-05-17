# Indic NLP Pipeline — Indian Names Dataset Processing

End-to-end Python pipeline for processing large-scale Indian names datasets (tested at 505M+ records). Handles OCR/Unicode cleaning, Devanagari + Gurmukhi transliteration, name parsing, religion classification, and frequency analysis.

## Pipeline Stages

| Stage | Script | What it does |
|-------|--------|--------------|
| 1 | `clean_unicode.py` | NFC normalization, zero-width char stripping, OCR artifact fixes, honorific removal |
| 2 | `transliterate.py` | Devanagari + Gurmukhi to Latin via IndicXlit/AI4Bharat; dedup cache cuts runtime ~70% |
| 3 | `parse_names.py` | Extracts given name + surname from full and parental name fields |
| 4 | `classify_religion.py` | Name-based religion classifier wrapper with confidence scores |
| 5 | `frequency_analysis.py` | Name counts by age cohort (5-yr bins), state/district, religion label |

## Quick Start

```bash
pip install -r requirements.txt

# Run full pipeline on your CSVs
python pipeline.py \
  --input "data/*.csv" \
  --output output/ \
  --chunk-size 1000000 \
  --name-col full_name \
  --parental-col parental_name
```

## Dataset Format

Expected CSV columns (names are configurable):

| Column | Description |
|--------|-------------|
| `full_name` | Full name — may be Devanagari, Gurmukhi, or Latin |
| `parental_name` | Parent/spouse name with S/O, D/O, W/O prefix (optional) |
| `age` | Numeric age for cohort binning |
| `state` | State name for geographic aggregation |
| `district` | District (optional) |

See `sample_data/sample.csv` for an example with mixed scripts.

## Memory Safety at Scale

- Polars lazy scan — never loads full dataset into RAM
- Configurable chunk size (default 1M rows)
- Checkpoint/resume: if interrupted, restart with same command — already-processed chunks are skipped
- Transliteration dedup cache persists across runs in `.cache/transliteration_cache.json`

## Outputs

```
output/
  classified_master.csv         # Full dataset with all added columns
  frequency/
    name_freq_overall.csv       # Given name frequency (all records)
    name_freq_by_cohort.csv     # Frequency by 5-year age cohort
    name_freq_by_state.csv      # Frequency by state
    name_freq_by_religion.csv   # Frequency by religion label
    surname_freq_overall.csv    # Surname frequency
    religion_distribution.csv   # Religion label counts + percentages
```

## Religion Classifier

`classify_religion.py` is a wrapper — plug in your specified ML algorithm by replacing `_predict_one()` in `_NameReligionClassifier`. The batch interface and confidence scoring remain unchanged.

Records with confidence below `CONFIDENCE_THRESHOLD` (default 0.6) are labelled `Uncertain` rather than silently misclassified.

## Configuration

All key parameters are CLI args to `pipeline.py`:

| Arg | Default | Description |
|-----|---------|-------------|
| `--chunk-size` | 1000000 | Rows per processing batch |
| `--name-col` | full_name | Primary name column name |
| `--parental-col` | None | Parental name column (optional) |
| `--model-path` | None | Path to classifier model weights |
| `--skip-frequency` | False | Skip Stage 5 if only cleaning needed |

## Author

Dr. Sandeep Grover — PhD Data Science, NLP + Large-Scale Data Engineering
