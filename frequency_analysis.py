"""
Stage 5 — Frequency analysis: name counts by age cohort, geography, religion.
Reads the classified master CSV (Polars lazy scan) and exports summary tables.
"""

import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

AGE_BIN_YEARS = 5  # cohort width in years


def _add_age_cohort(df: pl.LazyFrame, age_col: str = "age") -> pl.LazyFrame:
    """Bin continuous age into 5-year cohorts: '0-4', '5-9', ..., '85+'."""
    return df.with_columns(
        pl.when(pl.col(age_col) >= 85)
        .then(pl.lit("85+"))
        .otherwise(
            (pl.col(age_col) // AGE_BIN_YEARS * AGE_BIN_YEARS).cast(pl.Utf8)
            + pl.lit("-")
            + ((pl.col(age_col) // AGE_BIN_YEARS * AGE_BIN_YEARS) + (AGE_BIN_YEARS - 1)).cast(pl.Utf8)
        )
        .alias("age_cohort")
    )


def name_frequency_table(
    lf: pl.LazyFrame,
    name_col: str = "given_name",
    group_cols: list[str] | None = None,
) -> pl.DataFrame:
    """
    Count name frequency, optionally grouped by age_cohort / state / religion_label.
    Returns a sorted DataFrame: name | group... | count | pct_of_group.
    """
    group_cols = group_cols or []
    agg_cols = [name_col] + group_cols

    freq = (
        lf.group_by(agg_cols)
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )

    if group_cols:
        total_per_group = (
            lf.group_by(group_cols)
            .agg(pl.len().alias("group_total"))
        )
        freq = freq.join(total_per_group, on=group_cols, how="left").with_columns(
            (pl.col("count") / pl.col("group_total") * 100).round(4).alias("pct_of_group")
        ).drop("group_total")

    return freq.collect()


def run_frequency_analysis(
    classified_csv: Path,
    output_dir: Path,
    age_col: str = "age",
    state_col: str = "state",
    religion_col: str = "religion_label",
    given_col: str = "given_name",
    surname_col: str = "surname",
) -> None:
    """
    Run all frequency aggregations on the classified master CSV and export.
    Outputs (in output_dir/):
      - name_freq_overall.csv
      - name_freq_by_cohort.csv
      - name_freq_by_state.csv
      - name_freq_by_religion.csv
      - surname_freq_overall.csv
      - religion_distribution.csv
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    lf = pl.scan_csv(classified_csv, infer_schema_length=10_000)

    if age_col in lf.schema:
        lf = _add_age_cohort(lf, age_col)

    # 1. Overall given name frequency
    logger.info("Computing overall name frequency...")
    name_freq_overall = name_frequency_table(lf, given_col)
    name_freq_overall.write_csv(output_dir / "name_freq_overall.csv")

    # 2. By age cohort
    if "age_cohort" in lf.schema:
        logger.info("Computing frequency by age cohort...")
        freq_cohort = name_frequency_table(lf, given_col, ["age_cohort"])
        freq_cohort.write_csv(output_dir / "name_freq_by_cohort.csv")

    # 3. By state
    if state_col in lf.schema:
        logger.info("Computing frequency by state...")
        freq_state = name_frequency_table(lf, given_col, [state_col])
        freq_state.write_csv(output_dir / "name_freq_by_state.csv")

    # 4. By religion
    if religion_col in lf.schema:
        logger.info("Computing frequency by religion...")
        freq_religion = name_frequency_table(lf, given_col, [religion_col])
        freq_religion.write_csv(output_dir / "name_freq_by_religion.csv")

    # 5. Surname frequency
    logger.info("Computing surname frequency...")
    surname_freq = name_frequency_table(lf, surname_col)
    surname_freq.write_csv(output_dir / "surname_freq_overall.csv")

    # 6. Religion distribution summary
    if religion_col in lf.schema:
        logger.info("Computing religion distribution...")
        rel_dist = (
            lf.group_by(religion_col)
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
            .collect()
        )
        total = rel_dist["count"].sum()
        rel_dist = rel_dist.with_columns(
            (pl.col("count") / total * 100).round(2).alias("pct")
        )
        rel_dist.write_csv(output_dir / "religion_distribution.csv")

    logger.info("Frequency analysis complete. Output: %s", output_dir)
