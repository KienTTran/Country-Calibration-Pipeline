"""Validation utilities for MaSim outputs and post-processing."""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
from pandas import DataFrame

from masim_analysis import analysis, commands, configure, utils
from masim_analysis.calibrate import get_last_year_statistics
from masim_analysis.configure import CountryParams

from ruamel.yaml import YAML

yaml = YAML()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _averaging_pass(
    country: CountryParams,
    output_dir: Path | str = Path("output"),
) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame, DataFrame, DataFrame, DataFrame]:
    """Run averaging over validation .db outputs and write CSV summaries."""
    base_dir = Path(output_dir) / country.country_code / "validation"
    base_dir.mkdir(parents=True, exist_ok=True)

    analysis_dir = Path("data") / country.country_code / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    (
        ave_population,
        ave_treatment,
        ave_cases,
        ave_prevalence_2_to_10,
        ave_cases_2_to_10,
        ave_prevalence_under_5,
        ave_cases_under_5,
    ) = analysis.get_average_summary_statistics(base_dir)

    ave_population.to_csv(analysis_dir / "ave_population.csv")
    ave_treatment.to_csv(analysis_dir / "ave_treatment.csv")
    ave_cases.to_csv(analysis_dir / "ave_cases.csv")
    ave_prevalence_2_to_10.to_csv(analysis_dir / "ave_prevalence_2_to_10.csv")
    ave_cases_2_to_10.to_csv(analysis_dir / "ave_cases_2_to_10.csv")
    ave_prevalence_under_5.to_csv(analysis_dir / "ave_prevalence_under_5.csv")
    ave_cases_under_5.to_csv(analysis_dir / "ave_cases_under_5.csv")

    return (
        ave_population, ave_treatment, ave_cases,
        ave_prevalence_2_to_10, ave_cases_2_to_10,
        ave_prevalence_under_5, ave_cases_under_5,
    )


def _prevalence_comparison(
    country: CountryParams,
    ave_cases: DataFrame,
    mean_prevalence_2_to_10: DataFrame,
    mean_prevalence_under_5: DataFrame,
    mean_population: DataFrame,
) -> DataFrame:
    """Prepare observed vs. predicted prevalence comparison table."""
    prevalence_obs, _ = utils.read_raster(
        Path("data") / country.country_code / f"{country.country_code}_pfpr2to10.asc"
    )
    district_raster, _ = utils.read_raster(
        Path("data") / country.country_code / f"{country.country_code}_districts.asc"
    )

    prevalence_comp = (
        mean_prevalence_2_to_10[["mean"]].copy().div(100).rename(columns={"mean": "mean_2_to_10"})
    )
    prevalence_comp["mean_under_5"] = (
        mean_prevalence_under_5[["mean"]].copy().div(100).rename(columns={"mean": "mean_under_5"})
    )

    prevalence_district = (
        DataFrame({"district": district_raster.flatten(), "obs": prevalence_obs.flatten()})
        .groupby("district")["obs"].mean().rename("obs").to_frame()
    )

    prevalence = prevalence_comp.merge(prevalence_district, left_index=True, right_index=True, how="outer")
    prevalence = prevalence.merge(
        mean_population["mean"].rename("population"), left_index=True, right_index=True, how="outer"
    )
    return prevalence


def post_process(
    country: CountryParams,
    params: dict,
    logger: logging.Logger | None = None,
    output_dir: Path | str = Path("output"),
):
    """Run validation post-processing: averaging, comparisons and plots."""
    if logger is None:
        logger = utils.get_country_logger(country.country_code, "validation")

    output_dir = Path(output_dir)
    analysis_dir = Path("data") / country.country_code / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    image_dir = output_dir.parent / "images" / country.country_code
    image_dir.mkdir(parents=True, exist_ok=True)

    # Averaging pass
    (
        ave_population, ave_treatment, ave_cases,
        ave_prevalence_2_to_10, ave_cases_2_to_10,
        ave_prevalence_under_5, ave_cases_under_5,
    ) = _averaging_pass(country, output_dir=output_dir)

    # Last-year summary statistics
    (
        mean_cases, mean_treatment,
        mean_prevalence_2_to_10, mean_prevalence_under_5,
        mean_population,
    ) = get_last_year_statistics(
        ave_cases, ave_treatment,
        ave_prevalence_2_to_10, ave_prevalence_under_5,
        ave_population,
    )

    mean_cases.to_csv(analysis_dir / "mean_cases.csv")
    mean_treatment.to_csv(analysis_dir / "mean_treatment.csv")
    mean_prevalence_2_to_10.to_csv(analysis_dir / "mean_prevalence_2_to_10.csv")
    mean_prevalence_under_5.to_csv(analysis_dir / "mean_prevalence_under_5.csv")
    mean_population.to_csv(analysis_dir / "mean_population.csv")

    scale = params["artificial_rescaling_of_population_size"]
    logger.info(f"Last-year mean treatment: {mean_treatment['mean'].sum():,.0f} | SCALED: {mean_treatment['mean'].sum() / scale:,.0f}")
    logger.info(f"Clinical episodes: {mean_cases['mean'].sum():,.0f} | SCALED: {mean_cases['mean'].sum() / scale:,.0f}")
    logger.info(f"Population: {mean_population['mean'].sum():,.0f} | SCALED: {mean_population['mean'].sum() / scale:,.0f}")

    # Prevalence comparison
    prevalence = _prevalence_comparison(
        country, ave_cases, mean_prevalence_2_to_10, mean_prevalence_under_5, mean_population,
    )
    prevalence.to_csv(analysis_dir / "prevalence_comparison.csv")

    fig = analysis.plot_prevalence_trend(
        prevalence["obs"].to_numpy(),
        prevalence["mean_2_to_10"].to_numpy(),
        prevalence["population"].to_numpy(),
        "2 to 10",
    )
    fig.savefig(image_dir / "prevalence_fit_2_to_10.png", dpi=300, bbox_inches="tight")
    logger.info(f"Saved prevalence fit plot to {image_dir}")
    logger.info("Validation post-processing completed.")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _build_validation_config(country: CountryParams, country_code_lower: str, scaling: float) -> dict:
    """Write the validation YAML config and return the params dict."""
    strategy_db = yaml.load((Path("conf") / country_code_lower / "test" / "strategy_db.yaml").read_text())
    events = yaml.load((Path("conf") / country_code_lower / "test" / "events.yaml").read_text())

    params = configure.configure(
        country_code=country.country_code,
        birth_rate=country.birth_rate,
        initial_age_structure=country.initial_age_structure,
        age_distribution=country.age_distribution,
        death_rates=country.death_rate,
        starting_date=country.starting_date,
        start_of_comparison_period=country.start_of_comparison_period,
        ending_date=country.ending_date,
        strategy_db=strategy_db,
        calibration_str="",
        calibration=False,
    )
    params["artificial_rescaling_of_population_size"] = scaling
    params["events"].extend(events)

    cfg_path = Path("conf") / country_code_lower / "test" / "validation_config.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(params, f)

    return params


def validate(
    country_code: str,
    repetitions: int = 50,
    output_dir: Path | str = Path("output"),
    job_dir: Path | str = Path("jobs"),
    scaling: float = 0.25,
    host_name: str = "nd04",
) -> None:
    """Run the full validation pipeline for a country.

    Steps:
        1. Write validation configuration YAML.
        2. Generate and submit MaSim commands via PBS.
        3. Check for errored runs and retry (unconditionally).
        4. Check for missing .db files and resubmit if needed.
        5. Run post-processing (averaging, prevalence comparison, plots).

    Parameters
    ----------
    country_code : str
        Country code (case-insensitive), e.g. ``'UGA'``.
    repetitions : int
        Number of repetitions per parameter set.
    output_dir : Path | str
        Base output directory for MaSim ``.db`` files.
    job_dir : Path | str
        Base directory for PBS scripts and logs.
    scaling : float
        Artificial rescaling of population size.
    host_name : str
        PBS host constraint (empty string for any host).
    """
    # ── Guard against positional-arg mistakes ──
    if isinstance(output_dir, (float, int)) or isinstance(job_dir, (float, int)):
        raise TypeError(
            f"output_dir/job_dir must be path-like. Got output_dir={output_dir!r}, "
            f"job_dir={job_dir!r}. Did you pass scaling as a positional argument?"
        )

    output_dir = Path(output_dir)
    job_dir = Path(job_dir)
    country_code_lower = country_code.lower()

    country = CountryParams.load(name=country_code)
    logger = utils.get_country_logger(country_code, "validation")
    logger.info(f"Starting validation for country: {country_code}")

    # ── 1. Build config ──
    params = _build_validation_config(country, country_code_lower, scaling)
    logger.info("Validation configuration file created.")

    # ── 2. Generate commands ──
    # Use country.country_code consistently for all paths so that
    # submit_and_wait_pbs (which builds jobs/<country_code>/...) matches.
    cc = country.country_code
    validation_output = output_dir / cc / "validation"
    validation_log = job_dir / cc / "validation" / "log"

    validation_output.mkdir(parents=True, exist_ok=True)
    validation_log.mkdir(parents=True, exist_ok=True)

    _, cmds = commands.generate_commands(
        Path("conf") / country_code_lower / "test" / "validation_config.yaml",
        validation_output,
        validation_log,
        repetitions,
        False,
    )
    logger.info(f"Generated {len(cmds)} validation commands.")

    # ── 3. Initial PBS submission ──
    utils.submit_and_wait_pbs(
        cmds=cmds,
        country_code=cc,
        run_type="validation",
        host_name=host_name,
        logger=logger,
        rotate_logs=False,
        max_active_jobs=50,  # Keep some buffer below the cluster limit to avoid overloading
    )
    logger.info("Initial validation run completed.")

    # ── 4. Check for errors and retry (UNCONDITIONAL) ──
    cmds_path = str(job_dir / cc / "validation" / "cmds.txt")
    log_dir = str(validation_log)

    utils.retry_failed_runs(
        log_dir=log_dir,
        cmds_path=cmds_path,
        country_code=cc,
        run_type="validation",
        host_name=host_name,
        logger=logger,
    )

    # ── 5. Check for missing .db files and resubmit ──
    missing_cmds = utils.check_missing_db_files(
        output_dir=validation_output,
        cmds_path=cmds_path,
        repetitions=repetitions,
        logger=logger,
    )
    if missing_cmds:
        logger.info(f"Found {len(missing_cmds)} missing .db files. Resubmitting...")
        utils.submit_and_wait_pbs(
            cmds=missing_cmds,
            country_code=cc,
            run_type="validation",
            host_name=host_name,
            logger=logger,
            rotate_logs=True,
            max_active_jobs=50,  # Keep some buffer below the cluster limit to avoid overloading
        )
        # Retry errors from the missing-file resubmission too
        utils.retry_failed_runs(
            log_dir=log_dir,
            cmds_path=cmds_path,
            country_code=cc,
            run_type="validation",
            host_name=host_name,
            logger=logger,
        )
    else:
        logger.info("No missing .db files.")

    # ── 6. Post-processing ──
    logger.info("Starting post-processing...")
    post_process(country, params, logger, output_dir=output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Validate MaSim model for a given country.")
    parser.add_argument("country_code", type=str, help="Country code (e.g. 'UGA').")
    parser.add_argument("-r", "--repetitions", type=int, default=50,
                        help="Repetitions per parameter set (default: 50).")
    parser.add_argument("-o", "--output_dir", type=str, default="output",
                        help="Output directory (default: 'output').")
    parser.add_argument("-s", "--scaling", type=float, default=0.25,
                        help="Population rescaling factor (default: 0.25).")
    parser.add_argument("-n", "--node_name", type=str, default="nd04",
                        help="PBS host name (default: 'nd04').")
    args = parser.parse_args()
    validate(
        args.country_code,
        repetitions=args.repetitions,
        output_dir=args.output_dir,
        scaling=args.scaling,
        host_name=args.node_name,
    )


if __name__ == "__main__":
    main()