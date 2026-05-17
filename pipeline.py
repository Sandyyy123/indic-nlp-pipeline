"""
Main pipeline orchestrator — Indian Names Dataset Processing.
Runs all 5 stages: clean -> transliterate -> parse -> classify -> frequency.
Supports checkpoint/resume: processed chunks are tracked in .cache/progress.json.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import polars as pl
from tqdm import tqdm

from clean_unicode import clean_chunk, add_script_column
from transliterate import transliterate_chunk, _cache as translit_cache
from parse_names import parse_names_chunk
from classify_religion import classify_chunk, load_classifier
from frequency_analysis import run_frequency_analysis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

PROGRESS_FILE = Path(".cache/progress.json")
CHECKPOINT_EVERY = 10  # save progress every N chunks


def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"completed_chunks": []}


def save_progress(state: dict) -> None:
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        json.dump(state, f)


def process_csv_files(
    input_glob: str,
    output_dir: Path,
    chunk_size: int,
    name_cols: list[str],
    parental_col: str | None,
    model_path: Path | None,
    use_thinking: bool = False,  # reserved for future use
) -> Path:
    """
    Process all input CSVs through the 5-stage pipeline.
    Returns path to the classified master CSV.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    master_csv = output_dir / "classified_master.csv"
    progress = load_progress()
    completed = set(progress["completed_chunks"])

    load_classifier(model_path)

    input_files = sorted(Path(".").glob(input_glob))
    if not input_files:
        logger.error("No files matched: %s", input_glob)
        sys.exit(1)

    logger.info("Found %d input file(s)", len(input_files))

    first_write = not master_csv.exists()
    chunk_count = 0

    for csv_path in input_files:
        logger.info("Processing: %s", csv_path)

        # Estimate total chunks for tqdm
        file_lines = sum(1 for _ in open(csv_path)) - 1  # subtract header
        total_chunks = (file_lines // chunk_size) + 1

        reader = pl.read_csv_batched(csv_path, batch_size=chunk_size, infer_schema_length=10_000)

        pbar = tqdm(total=total_chunks, desc=csv_path.name, unit="chunk")
        while True:
            batch = reader.next_batches(1)
            if not batch:
                break
            df = batch[0]

            chunk_id = f"{csv_path.name}:{chunk_count}"
            if chunk_id in completed:
                chunk_count += 1
                pbar.update(1)
                continue

            # Stage 1: Unicode cleaning
            df = clean_chunk(df, name_cols)
            df = add_script_column(df, name_cols[0])

            # Stage 2: Transliteration
            df = transliterate_chunk(df, name_cols[0])
            if parental_col and parental_col in df.columns:
                df = transliterate_chunk(df, parental_col)

            # Stage 3: Name parsing
            df = parse_names_chunk(
                df,
                full_name_col="name_latin",
                parental_col=f"{parental_col}_latin" if parental_col else None,
            )

            # Stage 4: Religion classification
            df = classify_chunk(df)

            # Write output
            if first_write:
                df.write_csv(master_csv)
                first_write = False
            else:
                with open(master_csv, "ab") as f:
                    f.write(df.write_csv(include_header=False).encode())

            # Checkpoint
            completed.add(chunk_id)
            chunk_count += 1
            pbar.update(1)

            if chunk_count % CHECKPOINT_EVERY == 0:
                save_progress({"completed_chunks": list(completed)})
                translit_cache.save()
                logger.info("Checkpoint saved at chunk %d", chunk_count)

        pbar.close()

    # Final save
    save_progress({"completed_chunks": list(completed)})
    translit_cache.save()
    logger.info("All chunks processed. Master CSV: %s", master_csv)
    return master_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Indian Names Dataset Processing Pipeline")
    parser.add_argument("--input", required=True, help="Glob pattern for input CSVs, e.g. 'data/*.csv'")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--chunk-size", type=int, default=1_000_000, help="Rows per chunk (default 1M)")
    parser.add_argument("--name-col", default="full_name", help="Primary name column")
    parser.add_argument("--parental-col", default=None, help="Parental name column (optional)")
    parser.add_argument("--model-path", default=None, help="Path to religion classifier model weights")
    parser.add_argument("--skip-frequency", action="store_true", help="Skip Stage 5 frequency analysis")
    args = parser.parse_args()

    output_dir = Path(args.output)
    name_cols = [args.name_col]

    master_csv = process_csv_files(
        input_glob=args.input,
        output_dir=output_dir,
        chunk_size=args.chunk_size,
        name_cols=name_cols,
        parental_col=args.parental_col,
        model_path=Path(args.model_path) if args.model_path else None,
    )

    if not args.skip_frequency:
        logger.info("Stage 5: Running frequency analysis...")
        run_frequency_analysis(
            classified_csv=master_csv,
            output_dir=output_dir / "frequency",
        )

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
