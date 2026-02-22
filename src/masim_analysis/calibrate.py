"""masim_analysis.calibrate
=================================

Utilities to prepare, run and analyze MaSim calibration experiments.

This module contains routines to:
- generate MaSim configuration files and per-pixel inputs,
- assemble and run calibration command batches (via `bin/MaSim`),
- summarize `.db` outputs into calibration means, and
- fit/infer `beta` (transmission) using log-sigmoid and linear models.

Relevant locations in the repo:
- `scripts/generate_input_files.sh` — how templates are expanded into inputs.
- `bin/MaSim` — external model binary invoked by generated commands.
- `data/<country>/` and `conf/<country>/` — runtime inputs and configuration.
- `src/masim_analysis/calibrate.py` — this file: calibration orchestration and fitting.

Docstring conventions:
- All prevalence (pfpr) values coming from MaSim summaries are represented as
    percentages in raw output and are converted to fractions (divide by 100)
    before being passed to numerical fits elsewhere in the code.
"""

import argparse
import json
import logging
import os
import sqlite3
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray
from pandas import DataFrame
from ruamel.yaml import YAML
from ruamel.yaml.emitter import EmitterError
from scipy.optimize import curve_fit

from masim_analysis import analysis, configure, utils
from masim_analysis.commands import batch_generate_commands
from masim_analysis.configure import CountryParams

yaml = YAML()


# ---------------------------------------------------------------------------
# Configuration generation
# ---------------------------------------------------------------------------

def generate_configuration_files(
    country_code: str,
    calibration_year: int,
    access_rates: list[float],
    birth_rate: float,
    death_rate: list[float],
    initial_age_structure: list[int],
    age_distribution: list[float],
    population_scalar: float,
    strategy_db: dict[int, dict[str, str | list[int]]] = configure.STRATEGY_DB,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Create per-run MaSim YAML configuration files for the calibration batch.

    Produces files under ``conf/<country_code>/calibration/`` for every
    combination of population bin × access rate × beta.  Does **not** run
    MaSim; call ``batch_generate_commands`` afterwards.
    """
    comparison = date(calibration_year, 1, 1)
    start = date(calibration_year - 11, 1, 1)
    end = date(calibration_year + 1, 12, 31)

    country = CountryParams.load(name=country_code)
    POPULATION_BINS = country.calibration_population_bins
    BETAS = country.calibration_betas

    for pop in POPULATION_BINS:
        for access in access_rates:
            for beta in BETAS:
                execution_control = configure.configure(
                    country_code, birth_rate, initial_age_structure,
                    age_distribution, death_rate, start, comparison, end,
                    strategy_db, f"{pop}_{access}_{beta}", beta,
                    population_scalar, access, True, True,
                )

                write_pixel_data_files(execution_control["raster_db"], pop, access)
                output_path = os.path.join(
                    "conf", country_code, "calibration", f"cal_{pop}_{access}_{beta}.yml",
                )
                try:
                    yaml.dump(execution_control, open(output_path, "w"))
                except EmitterError as e:
                    if logger:
                        logger.error(f"Error writing YAML file {output_path}: {e}")


def write_pixel_data_files(raster_db: dict, population: int, access: float):
    """Write per-pixel ASCII files required by MaSim for a single population."""
    header = (
        "ncols 1\nnrows 1\nxllcorner 0\nyllcorner 0\n"
        f"cellsize 5\nNODATA_value {configure.NODATA_VALUE}\n"
    )
    with open(raster_db["population_raster"], "w") as f:
        f.write(f"{header}{population}")
    with open(raster_db["administrative_boundaries"][0]["raster"], "w") as f:
        f.write(f"{header}1")
    with open(raster_db["pr_treatment_under5"], "w") as f:
        f.write(f"{header}{access}")
    with open(raster_db["pr_treatment_over5"], "w") as f:
        f.write(f"{header}{access}")


def generate_calibration_commands(
    country: CountryParams,
    access_rates: list[float],
    repetitions: int = 20,
    population_scalar: float = 1.0,
    output_directory: Path = Path("output"),
    job_directory: Path = Path("jobs"),
) -> list[str]:
    """Generate shell command strings to run calibration simulations."""
    strategy_db = yaml.load(
        open(os.path.join("conf", country.country_code, "test", "strategy_db.yaml"), "r")
    )

    generate_configuration_files(
        country.country_code,
        country.start_of_comparison_period.year,
        access_rates,
        country.birth_rate,
        country.death_rate,
        country.initial_age_structure,
        country.age_distribution,
        population_scalar,
        strategy_db=strategy_db,
    )

    return batch_generate_commands(
        Path("conf") / country.country_code / "calibration",
        output_directory / country.country_code / "calibration",
        job_directory / country.country_code / "calibration" / "log",
        repetitions,
    )


# ---------------------------------------------------------------------------
# Missing-run checks
# ---------------------------------------------------------------------------

def is_valid_sqlite(file: Path) -> bool:
    """Check SQLite integrity and required tables."""
    if not file.exists() or file.stat().st_size < 4096:
        return False
    try:
        with sqlite3.connect(file) as conn:
            cur = conn.cursor()
            cur.execute("PRAGMA integrity_check;")
            if cur.fetchone()[0] != "ok":
                return False
            for table in ("monthly_data", "monthly_site_data_district"):
                cur.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                    (table,),
                )
                if cur.fetchone() is None:
                    return False
            cur.execute("SELECT COUNT(*) FROM monthly_data")
            if cur.fetchone()[0] == 0:
                return False
        return True
    except (sqlite3.DatabaseError, Exception):
        return False


# ---------------------------------------------------------------------------
# Fitting functions
# ---------------------------------------------------------------------------

def sinusoidal(x, amplitude, period, phase, offset):
    """Sinusoidal seasonal signal."""
    return amplitude * np.sin((2 * np.pi / period) * (x - phase)) + offset


def positive_sinusoidal(x, amplitude, period, phase, offset):
    """Sinusoidal signal clamped to be >= offset."""
    s = sinusoidal(x, amplitude, period, phase, offset)
    s[s <= offset] = offset
    return s


def linear(x, m, b):
    """Linear function: y = mx + b."""
    return m * x + b


def sigmoid(x, a, b, c):
    """Sigmoid function: y = a / (1 + exp(-b * (x - c)))."""
    return a / (1 + np.exp(-b * (x - c)))


def inverse_sigmoid(y, a, b, c):
    """Inverse sigmoid: x = c - (1/b) * ln(a/y - 1)."""
    return c - (1 / b) * np.log(a / y - 1)


def fit_log_sigmoid_model(
    betas: ArrayLike,
    pfpr: ArrayLike,
    pfpr_cutoff: float = 0.0,
    logger: Optional[logging.Logger] = None,
) -> NDArray[np.float64]:
    """Fit a log-sigmoid mapping from Beta → PfPR and return params [a, b, c].

    Fits ``y = a / (1 + exp(-b*(x-c)))`` where ``x = log10(beta)``
    and ``y = pfpr`` (fraction, not percent).  Returns an empty array
    on failure.
    """
    betas = np.array(betas)
    pfpr = np.array(pfpr)

    if np.any(pfpr < pfpr_cutoff):
        cutoff_beta_val = np.max(betas[pfpr < pfpr_cutoff])
        X_filtered = np.log10(betas[betas < cutoff_beta_val])
        y_filtered = pfpr[betas < cutoff_beta_val]
    else:
        X_filtered = np.log10(betas)
        y_filtered = pfpr

    if len(X_filtered) < 3:
        logging.warning(f"Not enough data points for regression: {len(X_filtered)} found.")
        return np.empty(0, dtype=np.float64)

    try:
        popt, _ = curve_fit(sigmoid, X_filtered, y_filtered, maxfev=5000)
        return np.array(popt)
    except (RuntimeError, TypeError) as exc:
        if logger:
            logger.warning(f"Curve fitting failed: {exc}")
        return np.empty(0)


# ---------------------------------------------------------------------------
# Beta map functions
# ---------------------------------------------------------------------------

def get_beta_models(
    populations: list[int],
    access_rates: list[float],
    means: DataFrame,
    pfpr_cutoff: float = 0.0,
) -> dict[float, dict[int, list[float]]]:
    """Fit log-sigmoid models for every (population, access_rate) combination."""
    models_map: dict[float, dict[int, list[float]]] = {
        ar: {pop: [] for pop in populations} for ar in access_rates
    }

    for population in populations:
        for treatment_access in access_rates:
            group = means[
                (means["population"] == population)
                & (means["access_rate"] == treatment_access)
            ]
            if group.empty:
                continue

            pfpr = group["pfpr_2to10"].to_numpy()
            beta = group["beta"].to_numpy()
            coefs = fit_log_sigmoid_model(beta, pfpr, pfpr_cutoff)
            if coefs.size == 0:
                logging.info(
                    f"Skipping fit for pop={population}, access={treatment_access}: insufficient data."
                )
                continue
            models_map[treatment_access][population] = coefs.tolist()

    return models_map


def load_beta_model(filename: str | Path) -> dict:
    """Load a beta model from a JSON file, converting keys to numeric types."""
    with open(filename, "r") as f:
        models = json.load(f)
    return {
        float(k): {int(float(k2)): v2 for k2, v2 in v.items()}
        for k, v in models.items()
    }


def _snap_access_key(
    models_map: dict[float, dict[int, list[float]]], access_rate: float,
) -> float:
    """Return the nearest access-rate key in models_map."""
    keys = np.array(sorted(models_map.keys()), dtype=float)
    return float(keys[np.argmin(np.abs(keys - access_rate))])


def _snap_population_bin(pop_to_coefs: dict[int, list[float]], population: int) -> int:
    """Return the largest population bin ≤ population, else the smallest."""
    bins = np.array(sorted(pop_to_coefs.keys()), dtype=int)
    if bins.size == 0:
        return -1
    le = bins[bins <= population]
    return int(le[-1]) if le.size > 0 else int(bins[0])


def get_beta(
    models_map: dict[float, dict[int, list[float]]],
    access_rate: float,
    population: int,
    pfpr: float,
    *,
    eps: float = 1e-6,
    min_population: int = 10,
) -> float:
    """Invert the fitted sigmoid to get beta from (access_rate, population, pfpr)."""
    if np.isnan(access_rate) or np.isnan(pfpr) or not models_map:
        return np.nan
    if population <= min_population:
        return 0.0

    acc_key = _snap_access_key(models_map, access_rate)
    pop_to_coefs = models_map.get(acc_key)
    if not pop_to_coefs:
        return np.nan

    pop_key = _snap_population_bin(pop_to_coefs, population)
    coefs = pop_to_coefs.get(pop_key)
    if not coefs or len(coefs) < 3:
        return np.nan

    a, b, c = float(coefs[0]), float(coefs[1]), float(coefs[2])
    if not np.isfinite(a) or not np.isfinite(b) or not np.isfinite(c):
        return np.nan
    if a <= 0.0 or abs(b) < 1e-12:
        return np.nan

    y = float(np.clip(pfpr, eps, a - eps))
    inside = a / y - 1.0
    if inside <= 0.0 or not np.isfinite(inside):
        return np.nan

    x_log10 = c - (1.0 / b) * np.log(inside)
    if not np.isfinite(x_log10):
        return np.nan

    beta = 10.0 ** x_log10
    return float(beta) if np.isfinite(beta) and beta > 0.0 else np.nan


def create_beta_map(
    models_map: dict[float, dict[int, list[float]]],
    population_raster: np.ndarray,
    access_rate_raster: np.ndarray,
    prevalence_raster: np.ndarray,
    *,
    eps: float = 1e-6,
    dtype=np.float32,
) -> np.ndarray:
    """Generate a raster of beta values from fitted models and input rasters."""
    pop = population_raster.astype(dtype, copy=False)
    acc = access_rate_raster.astype(dtype, copy=False)
    pfpr = prevalence_raster.astype(dtype, copy=False)
    beta_map = np.full(pop.shape, np.nan, dtype=dtype)

    rows, cols = beta_map.shape
    for r in range(rows):
        for c in range(cols):
            p, a, y = pop[r, c], acc[r, c], pfpr[r, c]
            if np.isnan(p) or np.isnan(a) or np.isnan(y):
                continue
            beta_map[r, c] = get_beta(
                models_map, access_rate=float(a),
                population=int(round(float(p))), pfpr=float(y), eps=eps,
            )
    return beta_map


def predicted_prevalence(
    models_map: dict[float, dict[int, list[float]]],
    population_raster: np.ndarray,
    treatment: np.ndarray,
    beta_map: np.ndarray,
    *,
    eps: float = 1e-6,
    dtype=np.float32,
) -> np.ndarray:
    """Compute predicted PfPR map from a beta map and fitted models."""
    pop = population_raster.astype(dtype, copy=False)
    acc = treatment.astype(dtype, copy=False)
    beta = beta_map.astype(dtype, copy=False)
    pfpr_map = np.full(pop.shape, np.nan, dtype=dtype)

    if not models_map:
        return pfpr_map

    access_keys = np.array(sorted(models_map.keys()), dtype=float)
    rows, cols = pfpr_map.shape

    for r in range(rows):
        for c in range(cols):
            p, a, bval = pop[r, c], acc[r, c], beta[r, c]
            if np.isnan(p) or np.isnan(a) or np.isnan(bval) or bval <= 0:
                continue

            acc_key = float(access_keys[np.argmin(np.abs(access_keys - float(a)))])
            pop_to_coefs = models_map.get(acc_key)
            if not pop_to_coefs:
                continue

            pop_int = int(round(float(p)))
            if pop_int <= 10:
                pfpr_map[r, c] = 0.0
                continue

            pop_key = _snap_population_bin(pop_to_coefs, pop_int)
            coefs = pop_to_coefs.get(pop_key)
            if not coefs or len(coefs) < 3:
                continue

            A, B, C = float(coefs[0]), float(coefs[1]), float(coefs[2])
            if A <= 0.0 or abs(B) < 1e-12:
                continue

            x = np.log10(float(bval) + eps)
            y = A / (1.0 + np.exp(-B * (x - C)))
            pfpr_map[r, c] = float(np.clip(y, 0.0, 1.0))

    return pfpr_map


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def get_last_year_statistics(
    ave_cases: DataFrame,
    ave_treatment: DataFrame,
    ave_prevalence_2_to_10: DataFrame,
    ave_prevalence_under_5: DataFrame,
    ave_population: DataFrame,
) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame, DataFrame]:
    """Calculate final-year summary statistics for cases, prevalence, and population."""
    months = ave_cases["monthly_data_id"].unique()
    end_month = months[-1] + 1
    start_month = end_month - 12

    def _last_year(df: DataFrame, agg: str, include_std: bool = True) -> DataFrame:
        subset = (
            df.loc[df["monthly_data_id"].between(start_month, end_month, inclusive="left")]
            .copy()
            .groupby("unit_id")
        )
        result = getattr(subset, agg)()
        result = result.drop(columns=["monthly_data_id"])
        result["mean"] = result.mean(axis=1)
        if include_std:
            result["std"] = result.std(axis=1)
        return result

    mean_treatment = _last_year(ave_treatment, "sum", include_std=False)
    mean_cases = _last_year(ave_cases, "sum")
    mean_population = _last_year(ave_population, "mean")
    mean_prevalence_2_to_10 = _last_year(ave_prevalence_2_to_10, "mean")
    mean_prevalence_under_5 = _last_year(ave_prevalence_under_5, "mean")

    return mean_cases, mean_treatment, mean_prevalence_2_to_10, mean_prevalence_under_5, mean_population


def summarize_calibration_results(
    country: CountryParams, data_path: Path | str = Path("output"),
) -> DataFrame:
    """Summarize calibration .db outputs into a means DataFrame."""
    data_path = Path(data_path)
    files = list(data_path.glob("*.db"))
    if not files:
        raise FileNotFoundError(f"No .db files found in {data_path}")

    # Use the first file to determine the end month
    data = analysis.get_table(files[0], "monthly_site_data_district")
    end_month = data["monthly_data_id"].unique()[-13]

    summary = DataFrame(
        columns=["population", "access_rate", "beta", "iteration",
                 "pfpr_under5", "pfpr_2to10", "pfpr_all"],
    )

    for count, file in enumerate(files, 1):
        data = analysis.get_table(file, "monthly_site_data_district")
        end_month = data["monthly_data_id"].unique()[-13]

        parts = file.stem.split("_")
        pop, access, beta, iteration = int(parts[1]), float(parts[2]), float(parts[3]), int(parts[-1])

        data = data.loc[data["monthly_data_id"].between(end_month - 12, end_month, inclusive="left")]
        summary.loc[file.stem] = data[["pfpr_under5", "pfpr_2to10", "pfpr_all"]].mean()
        summary.loc[file.stem, "population"] = pop
        summary.loc[file.stem, "access_rate"] = access
        summary.loc[file.stem, "beta"] = beta
        summary.loc[file.stem, "iteration"] = iteration

        if count % 10000 == 0:
            logging.info(f"Processed {count} files ({file})")

    # Convert percentages → fractions, then average over iterations
    for col in ("pfpr_under5", "pfpr_2to10", "pfpr_all"):
        summary[col] = summary[col].div(100)

    summary = (
        summary.drop(columns=["iteration"])
        .groupby(["population", "access_rate", "beta"])
        .mean()
        .reset_index()
    )
    return summary


# ---------------------------------------------------------------------------
# Simulation orchestration
# ---------------------------------------------------------------------------

def run_calibration_simulations(
    country: CountryParams,
    access_rates: list[float],
    repetitions: int,
    population_scalar: float,
    logger: Optional[logging.Logger] = None,
    host_name: Optional[str] = "",
    output_dir: Path | str = Path("output"),
    job_dir: Path | str = Path("jobs"),
) -> None:
    """Generate configs, submit calibration runs via PBS, and retry failures.

    Steps:
        1. Generate configs and MaSim commands.
        2. Submit initial PBS batch and wait.
        3. Unconditionally check for errored runs and retry.
        4. Check for missing .db files and resubmit if needed.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    output_dir = Path(output_dir)
    job_dir = Path(job_dir)

    cal_output = output_dir / country.country_code / "calibration"
    cal_log = job_dir / country.country_code / "calibration" / "log"
    cal_output.mkdir(parents=True, exist_ok=True)
    cal_log.mkdir(parents=True, exist_ok=True)

    # ── 1. Generate commands ──
    logger.info("Generating calibration commands...")
    cmds = generate_calibration_commands(
        country, access_rates, repetitions, population_scalar,
        output_directory=output_dir, job_directory=job_dir,
    )
    logger.info(f"Generated {len(cmds)} simulation commands.")

    # ── 2. Initial PBS submission ──
    utils.submit_and_wait_pbs(
        cmds=cmds,
        country_code=country.country_code,
        run_type="calibration",
        host_name=host_name,
        logger=logger,
        rotate_logs=False,
        max_active_jobs=400,
    )
    logger.info("Initial calibration run completed.")

    # ── 3. Check for errors and retry (UNCONDITIONAL) ──
    cmds_path = str(job_dir / country.country_code / "calibration" / "cmds.txt")
    log_dir = str(cal_log)

    utils.retry_failed_runs(
        log_dir=log_dir,
        cmds_path=cmds_path,
        country_code=country.country_code,
        run_type="calibration",
        host_name=host_name,
        logger=logger,
    )

    # ── 4. Check for missing .db files and resubmit ──
    missing_cmds = utils.check_missing_db_files(
        output_dir=cal_output,
        cmds_path=cmds_path,
        repetitions=repetitions,
        logger=logger,
    )
    if missing_cmds:
        logger.info(f"Found {len(missing_cmds)} missing .db files. Resubmitting...")
        utils.submit_and_wait_pbs(
            cmds=missing_cmds,
            country_code=country.country_code,
            run_type="calibration",
            host_name=host_name,
            logger=logger,
            rotate_logs=True,
            max_active_jobs=400,
        )
        utils.retry_failed_runs(
            log_dir=log_dir,
            cmds_path=cmds_path,
            country_code=country.country_code,
            run_type="calibration",
            host_name=host_name,
            logger=logger,
        )
    else:
        logger.info("No missing .db files.")


def calibrate(
    country_code: str,
    repetitions: int,
    population_scalar: float = 1.0,
    output_dir: Path | str = Path("output"),
    host_name: str = "",
) -> None:
    """Calibrate the MaSim model for a given country.

    Steps:
        1. Run calibration simulations (with retry).
        2. Summarize results into calibration means.
        3. Fit log-sigmoid models.
        4. Generate and save the beta map.
    """
    country_code = country_code.lower()
    output_dir = Path(output_dir)

    logger = utils.get_country_logger(country_code, "calibration")
    logger.info(f"Starting calibration for {country_code} ({repetitions} reps).")

    country = CountryParams.load(name=country_code)

    # Extract access rates from treatment-seeking raster
    treatment, _ = utils.read_raster(
        Path("data") / country.country_code / f"{country.country_code}_treatmentseeking.asc"
    )
    access_rates = sorted(float(t) for t in np.unique(treatment[~np.isnan(treatment)]))
    logger.info(f"Access rates: {access_rates}")

    # ── 1. Run simulations ──
    run_calibration_simulations(
        country, access_rates, repetitions, population_scalar,
        logger=logger, host_name=host_name, output_dir=output_dir,
    )

    # ── 2. Summarize ──
    logger.info("Summarizing calibration results...")
    means = summarize_calibration_results(
        country, output_dir / country.country_code / "calibration",
    )
    analysis_dir = Path("data") / country.country_code / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    means.to_csv(analysis_dir / "calibration_means.csv", index=False)

    # ── 3. Fit models ──
    logger.info("Fitting log-sigmoid models...")
    models_map = get_beta_models(
        populations=country.calibration_population_bins,
        access_rates=access_rates,
        means=means,
    )

    models_path = Path("data") / country.country_code / "models_map.json"
    with open(models_path, "w") as f:
        json.dump(models_map, f, indent=4)
    logger.info(f"Saved models map to {models_path}")
    
    plot_log_sigmoid_fits(models_map, access_rates, means, country, logger=logger)
    
    # ── 4. Create beta map ──
    logger.info("Creating beta map...")
    population_raster, meta = utils.read_raster(
        Path("data") / country.country_code / f"{country.country_code}_population.asc"
    )
    access_rate_raster, _ = utils.read_raster(
        Path("data") / country.country_code / f"{country.country_code}_treatmentseeking.asc"
    )
    prevalence_raster, _ = utils.read_raster(
        Path("data") / country.country_code / f"{country.country_code}_pfpr2to10.asc"
    )

    try:
        loaded_models = load_beta_model(models_path)
        beta_map = create_beta_map(loaded_models, population_raster, access_rate_raster, prevalence_raster)
        beta_path = Path("data") / country.country_code / f"{country.country_code}_beta.asc"

        write_kwargs = dict(
            xllcorner=meta["xllcorner"], yllcorner=meta["yllcorner"], cellsize=meta["cellsize"],
        )
        if country.use_masked_raster:
            write_kwargs["mask_raster"] = population_raster
        utils.write_raster(beta_map, beta_path, **write_kwargs)
        logger.info(f"Saved beta map to {beta_path}")
    except Exception as e:
        logger.error(f"Error creating beta map: {e}")
        raise


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_log_sigmoid_fits(
    models_map: dict[float, dict[int, list[float]]],
    access_rates: list[float],
    means: DataFrame,
    country: CountryParams,
    logger: logging.Logger | None = None,
) -> Figure:
    """Plot data points and fitted sigmoid curves for every (pop, access) pair."""
    if logger is None:
        logger = utils.get_country_logger(country.country_code, "calibration")

    num_rows = len(country.calibration_population_bins)
    num_cols = len(access_rates)
    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows),
        sharex=True, sharey=True, squeeze=False,
    )

    for i, population in enumerate(country.calibration_population_bins):
        for j, treatment_access in enumerate(access_rates):
            ax = axes[i, j]
            coefs = models_map[treatment_access][population]
            group = means[(means["population"] == population) & (means["access_rate"] == treatment_access)]

            ax.plot(group["beta"], group["pfpr_2to10"], ".", label="Data", color="black")
            X = np.linspace(1e-4, 10, 10000)
            try:
                Y = sigmoid(np.log10(X), *coefs)
                ax.plot(X, Y, color="red", label="Fitted Curve")
            except Exception as e:
                logger.warning(f"Sigmoid plot error (pop={population}, access={treatment_access}): {e}")

            ax.set_xscale("log")
            ax.set(xlabel="Beta", ylabel="pfpr_2to10", xlim=(1e-3, 10), ylim=(0, 1))
            ax.set_title(f"Pop: {population}, Access: {treatment_access}")
            ax.legend(fontsize=7)

    fig.suptitle("PfPR vs. Beta — Curve Fits by Population & Treatment Access", fontsize=24)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    out_path = Path("images") / country.country_code / f"{country.country_code}_log_sigmoid_fit.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    logger.info(f"Saved plot to {out_path}")
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Calibrate MaSim model for a given country.")
    parser.add_argument("country_code", type=str, help="Country code (e.g. 'UGA').")
    parser.add_argument("-r", "--repetitions", type=int, default=20,
                        help="Repetitions per parameter combination (default: 20).")
    parser.add_argument("-o", "--output_dir", type=str, default="output",
                        help="Output directory (default: 'output').")
    parser.add_argument("-s", "--population_scalar", type=float, default=1.0,
                        help="Population scale 0.1–1.0 (default: 1.0).")
    parser.add_argument("-n", "--node_name", type=str, default="",
                        help="PBS host name (default: '' for any host).")
    args = parser.parse_args()

    calibrate(args.country_code, args.repetitions, args.population_scalar, args.output_dir, args.node_name)


if __name__ == "__main__":
    main()