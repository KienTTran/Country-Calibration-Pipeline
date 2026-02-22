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

# Country calibration script
import argparse
import json
import os

from datetime import date
from pathlib import Path
from typing import Optional
import logging

import numpy as np
from pandas import DataFrame

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from numpy.typing import NDArray, ArrayLike
import numpy.typing as npt
from ruamel.yaml import YAML
from ruamel.yaml.emitter import EmitterError

from scipy.optimize import curve_fit

from masim_analysis import analysis, configure, utils
from masim_analysis.commands import batch_generate_commands, generate_commands
from masim_analysis.configure import CountryParams
# from masim_analysis.commands import setup_directories


yaml = YAML()


# ==== Configuration generation ====
def generate_configuration_files(
    country_code: str,
    calibration_year: int,
    access_rates: list[float],
    birth_rate: float,
    death_rate: list[float],
    initial_age_structure: list[int],
    age_distribution: list[float],
    # seasonality_file_name: str = "seasonality",
    population_scalar: float,
    strategy_db: dict[int, dict[str, str | list[int]]] = configure.STRATEGY_DB,
    # events: Optional[list[dict]] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Create per-run MaSim YAML configuration files used by the calibration batch.

    This function produces the set of MaSim configuration files placed under
    `conf/<country_code>/calibration/` (pixel-level inputs, execution control,
    and a `strategy_db` if provided). The generated files are consumed by
    `batch_generate_commands` to assemble command-line invocations of `MaSim`.

    Parameters
    ----------
    country_code
        ISO or project country code (e.g. ``'UGA'``) used to locate data and
        configuration directories under `data/` and `conf/`.
    calibration_year
        Year used to set the comparison window for calibration (the code uses
        an 11‑year lookback + comparison year windows when generating dates).
    access_rates
        Sorted list of unique treatment access rates (floats) derived from the
        treatment-seeking raster. Each access rate will have configurations
        generated for all population bins defined in ``POPULATION_BINS``.
    birth_rate
        Annual birth rate used to seed MaSim demographic inputs.
    death_rate
        Age-specific death rate vector used in configuration generation.
    initial_age_structure
        Initial age-structure counts used to create per-pixel population files.
    age_distribution
        Age distribution (fractions) used for certain demographic outputs.
    strategy_db
        Optional interventions/strategy definitions (loaded from
        `conf/<country>/test/strategy_db.yaml` by callers).
    logger
        Optional `logging.Logger` for informational messages.

    Notes
    -----
    - This function does not run MaSim; it only writes configuration files.
    - Caller is expected to call `batch_generate_commands` after config files
      are present to produce `MaSim` invocation strings.
    """
    # configure calibration dates
    comparison = date(calibration_year, 1, 1)
    start = date(calibration_year - 11, 1, 1)
    end = date(calibration_year + 1, 12, 31)
    # Create default execution control dictionary
    
    POPULATION_BINS = CountryParams.load(name=country_code).calibration_population_bins 
    BETAS = CountryParams.load(name=country_code).calibration_betas

    # Generate the configuration files
    for pop in POPULATION_BINS:
        for access in access_rates:
            for beta in BETAS:
                execution_control = configure.configure(
                    country_code,
                    birth_rate,
                    initial_age_structure,
                    age_distribution,
                    death_rate,
                    start,
                    comparison,
                    end,
                    strategy_db,
                    f"{pop}_{access}_{beta}",
                    beta,
                    population_scalar,
                    access,
                    True,
                    True
                )

                write_pixel_data_files(execution_control["raster_db"], pop, access)
                output_path = os.path.join("conf", country_code, "calibration", f"cal_{pop}_{access}_{beta}.yml")
                try:
                    yaml.dump(execution_control, open(output_path, "w"))
                except EmitterError as e:
                    if logger:
                        logger.error(f"Error writing YAML file {output_path}: {e}")


def write_pixel_data_files(raster_db: dict, population: int, access: float):
    """Write per-pixel ASCII files required by MaSim for a single population.

    The function expects ``raster_db`` to contain keys such as
    ``population_raster`` and ``district_raster`` pointing to output paths or
    templates. It writes the per-pixel population and district ASCII files used
    by MaSim when running a pixelized calibration experiment.

    Parameters
    ----------
    raster_db
        Mapping that must include the output paths for the population and
        administrative boundaries raster files (keys used in callers).
    population
        Population value (one of ``POPULATION_BINS``) used to populate the
        generated ASCII pixel file contents.
    """
    with open(raster_db["population_raster"], "w") as file:
        file.write(
            f"ncols 1\nnrows 1\nxllcorner 0\nyllcorner 0\ncellsize 5\nNODATA_value {configure.NODATA_VALUE}\n{population}"
        )
    with open(raster_db["administrative_boundaries"][0]["raster"], "w") as file:
        file.write(f"ncols 1\nnrows 1\nxllcorner 0\nyllcorner 0\ncellsize 5\nNODATA_value {configure.NODATA_VALUE}\n1")

    with open(raster_db["pr_treatment_under5"], "w") as file:
        file.write(
            f"ncols 1\nnrows 1\nxllcorner 0\nyllcorner 0\ncellsize 5\nNODATA_value {configure.NODATA_VALUE}\n{access}"
        )
    with open(raster_db["pr_treatment_over5"], "w") as file:
        file.write(
            f"ncols 1\nnrows 1\nxllcorner 0\nyllcorner 0\ncellsize 5\nNODATA_value {configure.NODATA_VALUE}\n{access}"
        )


def generate_calibration_commands(
    country: CountryParams, access_rates: list[float], repetitions: int = 20, population_scalar: float = 1.0, 
    output_directory: Path = Path("output"),
    job_directory: Path = Path("jobs"),
) -> list[str]:
    """Generate shell command strings to run calibration simulations.

    The returned list contains the command lines that invoke the `MaSim`
    binary for all combinations of population bins, access rates and beta
    values. Commands are created by `batch_generate_commands` from the
    `conf/<country>/calibration` directory into the `output/<country>/calibration`
    directory and are ready to be executed (for example with
    ``utils.multiprocess``).

    Parameters
    ----------
    country
        Loaded `CountryParams` object providing country metadata (paths,
        birth/death rates, start_of_comparison_period, etc.).
    access_rates
        Sorted list of unique treatment access rates to generate commands for.
    repetitions
        Number of repetitions to request per parameter set (default 20).
    output_directory
        Base output directory where MaSim results will be written.

    Returns
    -------
    list[str]
        Command strings ready to be executed to produce `.db` outputs under
        `output/<country>/calibration/`.
    """
    strategy_db = yaml.load(open(os.path.join("conf", country.country_code, "test", "strategy_db.yaml"), "r"))

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

    # Generate commands list
    cmds = batch_generate_commands(
        Path("conf") / country.country_code / "calibration",
        output_directory / country.country_code / "calibration",
        job_directory / country.country_code / "calibration" / "log",
        repetitions,
    )
    return cmds


import sqlite3
from pathlib import Path

def format_float(x: float) -> str:
    """Consistent float formatting for filenames."""
    return f"{x:.6g}"


def is_valid_sqlite(file: Path) -> bool:
    """Check SQLite integrity and required tables."""
    
    if not file.exists():
        return False
    
    if file.stat().st_size < 4096:
        return False

    try:
        with sqlite3.connect(file) as conn:
            cur = conn.cursor()

            # integrity check
            cur.execute("PRAGMA integrity_check;")
            if cur.fetchone()[0] != "ok":
                return False

            # check required tables exist
            cur.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='monthly_data'
            """)
            if cur.fetchone() is None:
                return False

            cur.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='monthly_site_data_district'
            """)
            if cur.fetchone() is None:
                return False

            # check table not empty
            cur.execute("SELECT COUNT(*) FROM monthly_data")
            if cur.fetchone()[0] == 0:
                return False

        return True

    except sqlite3.DatabaseError:
        return False

    except Exception:
        return False


def check_missing_runs(
    country_code: str,
    access_rates: list[float],
    output_dir: Path | str,
    repetitions: int = 20,
) -> list[str]:

    output_dir = Path(output_dir)
    base_path = output_dir / country_code / "calibration"

    country = CountryParams.load(name=country_code)

    POPULATION_BINS = country.calibration_population_bins
    BETAS = country.calibration_betas

    missing_cmds = []

    for pop in POPULATION_BINS:
        for access in access_rates:
            access_str = format_float(access)

            for beta in BETAS:
                beta_str = format_float(beta)

                for i in range(repetitions):

                    filename = f"cal_{pop}_{access_str}_{beta_str}_monthly_data_{i}.db"
                    file = base_path / filename

                    if not is_valid_sqlite(file):

                        cmd = (
                            f"./bin/MaSim "
                            f"-i ./conf/{country_code}/calibration/cal_{pop}_{access_str}_{beta_str}.yml "
                            f"-o ./output/{country_code}/calibration/cal_{pop}_{access_str}_{beta_str}_ "
                            f"-r SQLiteMonthlyReporter "
                            f"-j {i}"
                        )

                        missing_cmds.append(cmd)

    return missing_cmds

import os
from pathlib import Path

def format_float(x: float) -> str:
    return f"{x:.6g}"

def check_missing_runs_exists_only_calibration(
    country_code: str,
    access_rates: list[float],
    output_dir: Path | str,
    repetitions: int = 20,
) -> list[str]:

    output_dir = Path(output_dir)
    base_path = output_dir / country_code / "calibration"

    country = CountryParams.load(name=country_code)
    POPULATION_BINS = country.calibration_population_bins
    BETAS = country.calibration_betas

    # Scan directory once
    existing = set()
    if base_path.exists():
        with os.scandir(base_path) as it:
            for e in it:
                if e.is_file() and e.name.endswith(".db"):
                    existing.add(e.name)

    missing_cmds: list[str] = []

    for pop in POPULATION_BINS:
        for access in access_rates:
            access_str = format_float(access)

            for beta in BETAS:
                beta_str = format_float(beta)

                # FIX: 0 → repetitions-1
                for i in range(repetitions):

                    db_name = f"cal_{pop}_{access_str}_{beta_str}_monthly_data_{i}.db"

                    if db_name not in existing:
                        missing_cmds.append(
                            f"./bin/MaSim "
                            f"-i ./conf/{country_code}/calibration/cal_{pop}_{access_str}_{beta_str}.yml "
                            f"-o ./output/{country_code}/calibration/cal_{pop}_{access_str}_{beta_str}_ "
                            f"-r SQLiteMonthlyReporter "
                            f"-j {i}"
                        )

    # De-dup
    seen = set()
    out = []
    for c in missing_cmds:
        if c not in seen:
            seen.add(c)
            out.append(c)

    return out



# ==== Fitting functions ====
def sinusoidal(x, amplitude, period, phase, offset):
    """
    Generate a seasonal signal according to a sinusoidal model.
    """
    return amplitude * np.sin((2 * np.pi / period) * (x - phase)) + offset


def positive_sinusoidal(x, amplitude, period, phase, offset):
    """
    Generate a seasonal signal according to a sinusoidal model.
    """
    s = sinusoidal(x, amplitude, period, phase, offset)
    s[s <= offset] = offset
    return s


def linear(x, m, b):
    """
    Linear function for curve fitting.

    Equation: y = mx + b

    Parameters
    ----------
    x : array_like
        The independent variable.
    m : float
        The slope of the line.
    b : float
        The y-intercept of the line.

    Returns
    -------
    array_like
        The calculated y-values of the linear function.
    """
    return m * x + b


def sigmoid(x, a, b, c):
    """
    Sigmoid function for curve fitting.

    Equation: y = 1 / (1 + exp(-c * (x - b))) * a

    Parameters
    ----------
    x : array_like
        The independent variable.
    a : float
        The maximum value (amplitude) of the sigmoid.
    b : float
        The x-value of the sigmoid's midpoint.
    c : float
        The steepness of the sigmoid curve.

    Returns
    -------
    array_like
        The calculated y-values of the sigmoid function.
    """
    return a / (1 + np.exp(-b * (x - c)))


def inverse_sigmoid(y, a, b, c):
    """
    Inverse sigmoid function for prediction.

    Equation: x = c - (1 / b) * log(a / y - 1)

    Parameters
    ----------
    x : array_like
        The dependent variable (y-values).
    a : float
        The maximum value (amplitude) of the sigmoid.
    b : float
        The steepness of the sigmoid curve.
    c : float
        The x-value of the sigmoid's midpoint.

    Returns
    -------
    array_like
        The calculated x-values of the inverse sigmoid function.
    """
    return c - (1 / b) * np.log(a / y - 1)


def fit_log_sigmoid_model(
    betas: ArrayLike, pfpr: ArrayLike, pfpr_cutoff: float = 0.0, logger: Optional[logging.Logger] = None
) -> NDArray[np.float64]:
    """Fit a log-sigmoid mapping from Beta -> PfPR and return fitted params.

    The function fits a sigmoid model of the form ``y = a / (1 + exp(-b*(x-c)))``
    where ``x`` is ``log10(beta)`` and ``y`` is ``pfpr`` (fraction, not
    percent). If the supplied data have only a few points below ``pfpr_cutoff``,
    the implementation may fallback to a linear fit (handled by callers).

    Parameters
    ----------
    betas
        1-D arraylike of Beta values (transmission parameter) used as the
        independent variable. Values are expected > 0 and will be log10-transformed.
    pfpr
        1-D arraylike of PfPR values (fractions 0..1) corresponding to ``betas``.
    pfpr_cutoff
        PfPR threshold used to select a subset of points for the fit. Default 0.0.
    logger
        Optional logger used to record warnings or fit diagnostics.

    Returns
    -------
    numpy.ndarray
        Fitted parameter vector ``[a, b, c]`` for the sigmoid function. If the
        fit fails, an attempt to return a fitted linear fallback or an array of
        NaNs may be used depending on internal error handling.
    """
    # X = beta"].values
    # y = group["pfpr_2to10"].values

    # Convert betas and pfpr to np arrays for element-wise operations
    betas = np.array(betas)
    pfpr = np.array(pfpr)

    # Determine cutoff Beta based on pfpr2to10_mean
    if np.any(pfpr < pfpr_cutoff):
        cutoff_beta_val = np.max(betas[pfpr < pfpr_cutoff])  # Largest Beta where pfpr2to10_mean <= cutoff
        # X_filtered = np.log(betas[betas < cutoff_beta_val])
        X_filtered = np.log10(betas[betas < cutoff_beta_val])
        y_filtered = pfpr[betas < cutoff_beta_val]
    else:
        # X_filtered = np.log(betas)
        X_filtered = np.log10(betas)
        y_filtered = pfpr

    if len(X_filtered) < 3:  # Check if enough data points for regression
        logging.warning(f"Not enough data points for regression: {len(X_filtered)} points found.")
        return np.empty(0, dtype=np.float64())
    try:
        # Perform sigmoid regression
        popt, _ = curve_fit(
            sigmoid,
            X_filtered,
            y_filtered,
            maxfev=5000,
        )
        return np.array(popt)  # Store parameters

    except RuntimeError:
        if logger:
            logger.warning("Curve fitting failed to converge. Not enough data points or poor initial guess.")
        return np.empty(0)  # Or handle error as needed
    except TypeError:  # Handle cases where curve_fit might receive empty arrays from p0 logic
        if logger:
            logger.warning(
                "TypeError: Invalid input types for curve fitting. Ensure betas and pfpr are numeric arrays."
            )
        return np.empty(0)


# ==== Beta map functions ====
def get_beta_models(
    populations: list[int],
    access_rates: list[float],
    means: DataFrame,
    pfpr_cutoff: float = 0.0,
) -> dict[float, dict[int, list[float]]]:
    """
    Perform log-sigmoid regression on calibration data.

    This function fits a sigmoid model to the relationship between
    log-transformed Beta values and PfPR (Plasmodium falciparum parasite rate)
    for different combinations of population and treatment access rates.

    Parameters
    ----------
    populations : list[int] | np.typing.NDArray
        List of unique population values for which to fit the model.
    access_rates : list[float] | np.typing.NDArray
        List of unique treatment access rates for which to fit the model.
    means : pandas.DataFrame
        A DataFrame containing the mean PfPR and Beta values from calibration runs.
        It must include columns: 'population', 'access_rate', 'pfpr2to10', and 'beta'.
        'pfpr2to10' should be the mean PfPR in 2-10 year olds.
        'beta' is the transmission parameter.
    pfpr_cutoff : float, optional
        The cutoff value for PfPR below which data points will be excluded from the fitting process
        and an alternative linear model used.

    Returns
    -------
    dict[float, dict[int, typing.Any]]
        A nested dictionary where the outer keys are access rates, inner keys are
        populations, and values are the parameters (e.g., from `scipy.optimize.curve_fit`)
        of the fitted log-sigmoid model for that combination.

    Raises
    ------
    RuntimeError
        If `curve_fit` fails to converge for a particular data subset.
    """
    # Define cutoff based on pfpr2to10_mean
    pfpr_cutoff = 0.0  # Set the desired cutoff for pfpr2to10_mean
    models_map = {
        access_rate: {population: [] for population in populations} for access_rate in access_rates
    }  # stores trained model for every parameter configuration

    # Perform regression for each (Population, TreatmentAccess) group
    for population in populations:
        for treatment_access in access_rates:
            # Filter the data for the current Population and TreatmentAccess
            group = means[(means["population"] == population) & (means["access_rate"] == treatment_access)]
            if group.empty:
                continue

            group = group.copy()  # Create a copy to avoid SettingWithCopyWarning
            pfpr = group["pfpr_2to10"].to_numpy()
            beta = group["beta"].to_numpy()
            coefs = fit_log_sigmoid_model(beta, pfpr, pfpr_cutoff)
            if coefs.size == 0:
                print(
                    f"Skipping fitting for population {population} and access rate {treatment_access} due to insufficient data."
                )
                continue
            # Store the fitted model parameters
            # coefs_as_list = [float(coef) for coef in coefs]  # Convert to list of floats
            models_map[treatment_access][population] = coefs.tolist()  # type: ignore # Convert to list
    return models_map


def load_beta_model(filename: str) -> dict:
    """
    Load the beta model from a json file.

    Parameters
    ----------
    filename : str
        The name of the json file to load.

    Returns
    -------
    dict
        A dictionary containing the beta model.
    """
    models = json.load(open(filename, "r"))
    numeric = {float(k): {int(float(k2)): v2 for k2, v2 in v.items()} for k, v in models.items()}
    return numeric


import numpy as np


def create_beta_map(
    models_map: dict[float, dict[int, list[float]]],
    population_raster: np.ndarray,
    access_rate_raster: np.ndarray,
    prevalence_raster: np.ndarray,
    *,
    eps: float = 1e-6,
    dtype=np.float32,
) -> np.ndarray:
    """
    Generate a raster of beta values from fitted models and inputs.

    Fixes vs old version:
    - beta_map is FLOAT dtype (not inherited from population dtype via zeros_like).
    - Preserves NaNs: if any input is NaN, output stays NaN.
    - Uses get_beta() which snaps float access_rate to nearest model key and uses sorted bins.
    """
    pop = population_raster.astype(dtype, copy=False)
    acc = access_rate_raster.astype(dtype, copy=False)
    pfpr = prevalence_raster.astype(dtype, copy=False)

    beta_map = np.full(pop.shape, np.nan, dtype=dtype)

    rows, cols = beta_map.shape
    for r in range(rows):
        for c in range(cols):
            p = pop[r, c]
            a = acc[r, c]
            y = pfpr[r, c]

            if np.isnan(p) or np.isnan(a) or np.isnan(y):
                continue

            beta_map[r, c] = get_beta(
                models_map,
                access_rate=float(a),
                population=int(round(float(p))),
                pfpr=float(y),
                eps=eps,
            )

    return beta_map


def get_beta(
    models_map: dict[float, dict[int, list[float]]],
    access_rate: float,
    population: int,
    pfpr: float,
    *,
    eps: float = 1e-6,
    min_population: int = 10,
) -> float:
    """
    Robust inversion: (access_rate, population, pfpr) -> beta.

    Model:
      sigmoid(x,a,b,c) = a / (1 + exp(-b*(x - c)))
    where x = log10(beta).

    Inverse:
      x = c - (1/b) * ln(a/pfpr - 1)
      beta = 10**x

    Fixes vs old version:
    - Nearest-key lookup for float access_rate (avoids KeyError due to float precision).
    - Sorted population bins; chooses largest bin <= population (else smallest).
    - Clamps pfpr to (eps, a-eps) to avoid log domain errors.
    - Returns NaN on invalid math (does not silently return 0), except population<=min_population returns 0.0.
    """
    if np.isnan(access_rate) or np.isnan(pfpr):
        return np.nan
    if not models_map:
        return np.nan

    # Keep your rule: tiny population returns 0 beta
    if population <= min_population:
        return 0.0

    # --- snap access_rate to nearest available key
    access_keys = np.array(sorted(models_map.keys()), dtype=float)
    acc_key = float(access_keys[np.argmin(np.abs(access_keys - access_rate))])

    pop_to_coefs = models_map.get(acc_key)
    if not pop_to_coefs:
        return np.nan

    # --- choose population bin (largest <= population else smallest)
    pop_bins = np.array(sorted(pop_to_coefs.keys()), dtype=int)
    if pop_bins.size == 0:
        return np.nan

    le = pop_bins[pop_bins <= population]
    pop_key = int(le[-1]) if le.size > 0 else int(pop_bins[0])

    coefs = pop_to_coefs.get(pop_key)
    if not coefs or len(coefs) < 3:
        return np.nan

    a, b, c = float(coefs[0]), float(coefs[1]), float(coefs[2])

    if not np.isfinite(a) or not np.isfinite(b) or not np.isfinite(c):
        return np.nan
    if a <= 0.0 or abs(b) < 1e-12:
        return np.nan

    # Clamp pfpr to (0, a)
    y = float(np.clip(pfpr, eps, a - eps))

    inside = a / y - 1.0
    if inside <= 0.0 or not np.isfinite(inside):
        return np.nan

    x_log10 = c - (1.0 / b) * np.log(inside)  # natural log here is correct for the inverse derivation
    if not np.isfinite(x_log10):
        return np.nan

    beta = 10.0 ** x_log10
    if not np.isfinite(beta) or beta <= 0.0:
        return np.nan

    return float(beta)


def predicted_prevalence(
    models_map: dict[float, dict[int, list[float]]],
    population_raster: np.ndarray,
    treatment: np.ndarray,
    beta_map: np.ndarray,
    *,
    eps: float = 1e-6,
    dtype=np.float32,
) -> np.ndarray:
    """
    Compute predicted PfPR map from a beta map and fitted models.

    Fix vs old version:
    - Uses log10(beta) (not ln(beta)) to match fitting (fit_log_sigmoid_model uses log10).
    - Preserves NaNs.
    """
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
            p = pop[r, c]
            a = acc[r, c]
            bval = beta[r, c]

            if np.isnan(p) or np.isnan(a) or np.isnan(bval):
                continue
            if bval <= 0:
                continue

            # Snap access rate to nearest key
            acc_key = float(access_keys[np.argmin(np.abs(access_keys - float(a)))])
            pop_to_coefs = models_map.get(acc_key)
            if not pop_to_coefs:
                continue

            pop_bins = np.array(sorted(pop_to_coefs.keys()), dtype=int)
            if pop_bins.size == 0:
                continue

            pop_int = int(round(float(p)))
            if pop_int <= 10:
                pfpr_map[r, c] = 0.0
                continue

            le = pop_bins[pop_bins <= pop_int]
            pop_key = int(le[-1]) if le.size > 0 else int(pop_bins[0])

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


def get_last_year_statistics(
    ave_cases: DataFrame,
    ave_treatment: DataFrame,
    ave_prevalence_2_to_10: DataFrame,
    ave_prevalence_under_5: DataFrame,
    ave_population: DataFrame,
) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    """
    Calculate the final year statistics for cases, prevalence, and population.

    Arguments
    ----------
        ave_cases : DataFrame
            The data frame containing average cases data
        ave_treatment : DataFrame
            The data frame containing average treatment data
        ave_prevalence_2_to_10 : DataFrame
            The data frame containing average prevalence data for ages 2 to 10
        ave_prevalence_under_5 : DataFrame
            The data frame containing average prevalence data for ages under 5
        ave_population : DataFrame
            The data frame containing average population data

    Returns
    -------
    A tuple containing three DataFrames: mean_cases, mean_prevalence, mean_population
    """
    months = ave_cases["monthly_data_id"].unique()
    end_month = months[-1] + 1
    start_month = end_month - 12
    
    mean_treatment = (
        ave_treatment.loc[ave_treatment["monthly_data_id"].between(start_month, end_month, inclusive="left")]
        .copy()
        .groupby("unit_id")
        .sum()
    )
    mean_treatment = mean_treatment.drop(columns=["monthly_data_id"])
    # mean_treatment = mean_treatment.drop(columns=["clinical_episodes"])
    mean_treatment["mean"] = mean_treatment.mean(axis=1)

    mean_cases = (
        ave_cases.loc[ave_cases["monthly_data_id"].between(start_month, end_month, inclusive="left")]
        .copy()
        .groupby("unit_id")
        .sum()
    )
    mean_cases = mean_cases.drop(columns=["monthly_data_id"])
    # mean_cases = mean_cases.drop(columns=["clinical_episodes"])
    mean_cases["mean"] = mean_cases.mean(axis=1)
    mean_cases["std"] = mean_cases.std(axis=1)

    mean_population = (
        ave_population.loc[ave_population["monthly_data_id"].between(start_month, end_month, inclusive="left")]
        .copy()
        .groupby("unit_id")
        .mean()
    )
    mean_population = mean_population.drop(columns=["monthly_data_id"])
    # mean_population = mean_population.drop(columns=["population"])
    mean_population["mean"] = mean_population.mean(axis=1)
    mean_population["std"] = mean_population.std(axis=1)

    mean_prevalence_2_to_10 = (
        ave_prevalence_2_to_10.loc[
            ave_prevalence_2_to_10["monthly_data_id"].between(start_month, end_month, inclusive="left")
        ]
        .copy()
        .groupby("unit_id")
        .mean()
    )
    mean_prevalence_2_to_10 = mean_prevalence_2_to_10.drop(columns=["monthly_data_id"])
    # mean_prevalence_2_to_10 = mean_prevalence_2_to_10.drop(columns=["pfpr_2to10"])
    mean_prevalence_2_to_10["mean"] = mean_prevalence_2_to_10.mean(axis=1)
    mean_prevalence_2_to_10["std"] = mean_prevalence_2_to_10.std(axis=1)

    mean_prevalence_under_5 = (
        ave_prevalence_under_5.loc[
            ave_prevalence_under_5["monthly_data_id"].between(start_month, end_month, inclusive="left")
        ]
        .copy()
        .groupby("unit_id")
        .mean()
    )
    mean_prevalence_under_5 = mean_prevalence_under_5.drop(columns=["monthly_data_id"])
    # mean_prevalence_under_5 = mean_prevalence_under_5.drop(columns=["pfpr_under5"])
    mean_prevalence_under_5["mean"] = mean_prevalence_under_5.mean(axis=1)
    mean_prevalence_under_5["std"] = mean_prevalence_under_5.std(axis=1)

    return mean_cases, mean_treatment, mean_prevalence_2_to_10, mean_prevalence_under_5, mean_population


# ==== Main functionality ====
def run_calibration_simulations(
    country: CountryParams,
    access_rates: list[float],
    repetitions: int,
    population_scalar: float,
    max_workers: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
    host_name: Optional[str] = '',
) -> None:
    """
    Runs the full country-wide model calibration process using multiprocessing.
    This method assumes the following pre-processing has been completed:
    - Basic country-model data (ex: initial age structure, age distribution, death rate) and raster files have been assembled and placed under `data/<country_code>/`
    - Birth rate has been verified with a basic configuration file saved to `conf/<country_code>/test/<country_code>_params.yaml
    - `drug_db`, `therapy_db`, and `strategy_db` have been created and saved to `conf/<country_code>/test/strategy_db.yaml`
    - The implementation events have been created and saved to `conf/<country_code>/test/events.yaml`
    - Any seasonality effects are calculated and saved to `data/<country_code>/<country_code>_seasonality.csv`

    Parameters
    ----------
    country_code : str
        The country code for calibration
    repetitions : int
        Number of repetitions per parameter combination
    max_workers : Optional[int], optional
        Maximum number of worker processes. If None, uses os.cpu_count()
    """

    if logger is None:
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    logger.info("Generating calibration commands...")
    cmds = generate_calibration_commands(country, access_rates, repetitions,population_scalar)
    logger.info(f"Generated {len(cmds)} simulation commands")

    # Create output directory if it doesn't exist
    output_dir = os.path.join("output", country.country_code, "calibration")
    os.makedirs(output_dir, exist_ok=True)
    
    job_dir = os.path.join("jobs", country.country_code, "calibration")
    os.makedirs(job_dir, exist_ok=True)
    
    logger.info("Running calibration simulations via PBS")

    utils.submit_and_wait_pbs(
        cmds=cmds,
        country_code=country.country_code,
        run_type="calibration",
        host_name=host_name,
        logger=logger,
        rotate_logs=True,
    )

    logger.info("\nRunning calibration completed")
    
    cmds_path = os.path.join("jobs", country.country_code, "calibration", "cmds.txt")
    try:
        # check and return failed runs for 2 times, 
        # then give up and exit if error still exists    
        run_error_cmds = []
        for attempt in range(2):    
            run_error_cmds = utils.check_error_cmds(
                os.path.join("jobs", country.country_code, "calibration", "log"),
                cmds_path,
                logger
            )
            if run_error_cmds:
                logger.info(f"Attempting to re-run {len(run_error_cmds)} failed runs (Attempt {attempt + 1}/2).")
                utils.submit_and_wait_pbs(
                    cmds=run_error_cmds,
                    country_code=country.country_code,
                    run_type="calibration",
                    host_name=host_name,
                    logger=logger,
                    rotate_logs=True,
                )
            else:
                break    
        
        # Final check for any remaining failed runs
        run_error_cmds = utils.check_error_cmds(
            os.path.join("jobs", country.country_code, "calibration", "log"),
            cmds_path,
            logger
        )
        
        if run_error_cmds:
            logger.error(f"There are still {len(run_error_cmds)} failed runs after retries. Please check the logs for details.")
            exit(1)
        else:
            logger.info("All calibration runs completed successfully.")
    except Exception as e:
        logger.exception(f"Post-calibration error checking crashed: {e}")
        raise


def _summarize_calibration_results(
    country_code: str,
    access_rates: list[float],
    comparison_start_month: int,
    comparison_end_month: int,
    output_dir: Path | str,
    repetitions: int = 20,
) -> DataFrame:
    """
    Summarize the results of MaSim calibration runs.

    This function reads output files from multiple simulation runs,
    aggregates relevant metrics (e.g., PfPR), and returns a summary DataFrame.

    Parameters
    ----------
    country_code : str
        The country code.
    population_bins : list[int]
        Population bins used in calibration.
    access_rates : list[float]
        Access rates used in calibration.
    beta_values : list[float]
        Beta values used in calibration.
    comparison_year : int
        The year used for comparison or validation of results.
    output_dir : str
        Directory containing the MaSim output files.
    repetitions : int, optional
        Number of repetitions run for each parameter set, by default 20.

    Returns
    -------
    pandas.DataFrame
        A DataFrame summarizing the calibration results.
    """
    base_file_path = os.path.join(output_dir, country_code, "calibration")
    summary = DataFrame(
        columns=["population", "access_rate", "beta", "iteration", "pfpr_under5", "pfpr_2to10", "pfpr_all"]
    )
    # comparison = date(comparison_year, 1, 1)
    # year_end = date(comparison_year + 1, 1, 1)
    # Process summary
    
    POPULATION_BINS = CountryParams.load(name=country_code).calibration_population_bins 
    BETAS = CountryParams.load(name=country_code).calibration_betas
    
    for pop in POPULATION_BINS:
        for access in access_rates:
            for beta in BETAS:
                for i in range(repetitions):
                    filename = f"cal_{pop}_{access}_{beta}_monthly_data_{i}"
                    file = os.path.join(base_file_path, f"{filename}.db")
                    try:
                        data = analysis.get_table(file, "monthly_site_data_district")
                    except FileNotFoundError as _:
                        filename = f"cal_{pop}_{access}_{int(beta)}_monthly_data_{i}"  # TODO: #15 fix the masim file output to ensure consistent int/float digits
                        file = os.path.join(base_file_path, f"{filename}.db")
                        try:
                            data = analysis.get_table(file, "monthly_site_data_district")
                        except FileNotFoundError as e:
                            logging.warning(f"File not found: {e}")
                            continue
                    data = data.loc[
                        data["monthly_data_id"].between(comparison_start_month, comparison_end_month, inclusive="left")
                    ]
                    summary.loc[filename] = data[["pfpr_under5", "pfpr_2to10", "pfpr_all"]].mean()
                    # mean_pop = data["population"].mean()
                    # clinincal_episodes = data["clinical_episodes"].sum()
                    # pfpr = clinincal_episodes / mean_pop
                    summary.loc[filename, "population"] = pop
                    summary.loc[filename, "access_rate"] = access
                    summary.loc[filename, "beta"] = beta
                    summary.loc[filename, "iteration"] = int(i)
                    # summary.loc[filename, "pfpr"] = pfpr

    summary.to_csv(Path("data") / country.country_code / "analysis" / "calibration_summary.csv", index=False)
    return summary


def summarize_calibration_results(country: CountryParams, data_path: Path | str = Path("output")) -> DataFrame:
    data_path = Path(data_path)
    files = data_path.glob("*.db")
       
    data = analysis.get_table(next(files), "monthly_site_data_district")
    end_month = data["monthly_data_id"].unique()[-13]
    summary = DataFrame(
        columns=["population", "access_rate", "beta", "iteration", "pfpr_under5", "pfpr_2to10", "pfpr_all"]
    )
    count = 0
    for file in files:
        data = analysis.get_table(file, "monthly_site_data_district")
        end_month = data["monthly_data_id"].unique()[-13]
        file_name = file.stem
        parts = file_name.split("_")
        pop = int(parts[1])
        access = float(parts[2])
        beta = float(parts[3])
        iteration = int(parts[-1])
        data = data.loc[data["monthly_data_id"].between(end_month - 12, end_month, inclusive="left")]
        summary.loc[file_name] = data[["pfpr_under5", "pfpr_2to10", "pfpr_all"]].mean()
        summary.loc[file_name, "population"] = pop
        summary.loc[file_name, "access_rate"] = access
        summary.loc[file_name, "beta"] = beta
        summary.loc[file_name, "iteration"] = int(iteration)
        count += 1
        if count % 10000 == 0:
            print(count,file)

    summary["pfpr_under5"] = summary["pfpr_under5"].div(100)
    summary["pfpr_2to10"] = summary["pfpr_2to10"].div(100)
    summary["pfpr_all"] = summary["pfpr_all"].div(100)
    summary = summary.drop(columns=["iteration"])
    summary = summary.groupby(["population", "access_rate", "beta"]).mean().reset_index()
    # summary.to_csv(Path("data") / country.country_code / "analysis" / "calibration_means.csv", index=False)
    return summary


def calibrate(country_code: str, repetitions: int, population_scalar: float = 1.0, output_dir: Path | str = Path("output"), host_name: str = "") -> None:
    """
    Calibrate the MaSim model for a given country.
    """
    # Back up run to ensure output and log directories exist
    # setup_directories(country_code)
    country_code = country_code.lower()
    # Set up logger
    logger = utils.get_country_logger(country_code, "calibration")
    logger.info(f"Starting calibration for country: {country_code} with {repetitions} repetitions per parameter set.")

    # Load country parameters
    country = CountryParams.load(name=country_code)
    treatment, _ = utils.read_raster(
        os.path.join("data", country.country_code, f"{country.country_code}_treatmentseeking.asc")
    )
    treatment = np.unique(treatment)
    treatment = treatment[~np.isnan(treatment)]
    treatment = np.sort(treatment)
    access_rates = [float(t) for t in treatment]  # Convert to float for consistency and to make pyright happy
    logger.info(f"Access rates found in raster: {access_rates}")

    # Run calibration simulations
    logger.info("Running calibration simulations...")
    run_calibration_simulations(country, access_rates, repetitions, population_scalar, logger=logger, host_name=host_name)

    # Check for missing runs
    logger.info("Checking for missing calibration runs...")
    missing_cmds = check_missing_runs_exists_only_calibration(country.country_code, access_rates, output_dir, repetitions)
    if missing_cmds:
        logger.info(f"Found {len(missing_cmds)} missing runs. Re-running these simulations...")
        
        utils.submit_and_wait_pbs(
            cmds=missing_cmds,
            country_code=country.country_code,
            logger=logger,
            run_type="calibration",
            host_name=host_name,
            rotate_logs=True,
        )

        logger.info("\nRunning missing calibration completed")
        
        run_error_cmds = []
        for attempt in range(2):    
            run_error_cmds = utils.check_error_cmds(
                os.path.join("jobs", country.country_code, "calibration", "log"),
                logger
            )
            if run_error_cmds:
                logger.info(f"Attempting to re-run {len(run_error_cmds)} failed runs (Attempt {attempt + 1}/2).")
                utils.submit_and_wait_pbs(
                    cmds=run_error_cmds,
                    country_code=country.country_code,
                    logger=logger,
                    run_type="calibration",
                    host_name=host_name,
                    rotate_logs=True,
                )
            else:
                break    
        
        # Final check for any remaining failed runs
        run_error_cmds = utils.check_error_cmds(
            os.path.join("jobs", country.country_code, "calibration", "log"),
            logger
        )
        
        if run_error_cmds:
            logger.error(f"There are still {len(run_error_cmds)} failed runs after retries. Please check the logs for details.")
            exit()
        else:
            logger.info("All missing calibration runs completed successfully.")
            
    # Summarize calibration results
    logger.info("Summarizing calibration results...")
    means = summarize_calibration_results(country, Path(output_dir) / country.country_code / "calibration")
    means.to_csv(Path("data") / country.country_code / "analysis" / "calibration_means.csv", index=False)
    logger.info("Fitting log-sigmoid models to calibration data...")
    models_map = get_beta_models(
        populations=country.calibration_population_bins,
        access_rates=access_rates,
        means=means,
        pfpr_cutoff=0.0,
    )

    # Save the models map to a json file
    models_map_filename = "models_map.json"
    with open(Path("data") / country.country_code / models_map_filename, "w") as f:  # noqa: F811, ruff disabled
        json.dump(models_map, f, indent=4)
    logger.info(f"Saved models map to {Path('data') / country.country_code / models_map_filename}")

    # Plot the log-sigmoid fits
    # fig = plot_log_sigmoid_fits(models_map, access_rates, means, country, logger)
    # Create beta map
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
        models_map = load_beta_model(Path("data") / country.country_code / models_map_filename)
        beta_map = create_beta_map(models_map, population_raster, access_rate_raster, prevalence_raster)
        beta_map_filename = Path("data") / country.country_code / f"{country.country_code}_beta.asc"
        if country.use_masked_raster:
            utils.write_raster(beta_map, beta_map_filename, meta["xllcorner"], meta["yllcorner"], meta["cellsize"],mask_raster=population_raster)
        else:
            utils.write_raster(beta_map, beta_map_filename, meta["xllcorner"], meta["yllcorner"], meta["cellsize"])
        logger.info(f"Saved beta map to {beta_map_filename}")
    except Exception as e:
        logger.error(f"Error creating beta map: {e}")

def plot_log_sigmoid_fits(
    models_map: dict[float, dict[int, list[float]]],
    access_rates: list[float],
    means: DataFrame,
    country: CountryParams,
    logger: logging.Logger | None = None,
) -> Figure:
    if logger is None:
        logger = utils.get_country_logger(country.country_code, "calibration")
    # Plot all the model data, fits, and inverse fits on the same figure
    num_rows = len(country.calibration_population_bins)
    num_cols = len(access_rates)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows), sharex=True, sharey=True)
    for i, population in enumerate(country.calibration_population_bins):
        for j, treatment_access in enumerate(access_rates):
            try:
                ax = axes[i, j]  # Select subplot location
            except IndexError:
                ax = axes[i]
            coefs = models_map[treatment_access][population]
            group = means[(means["population"] == population) & (means["access_rate"] == treatment_access)]
            betas = group["beta"].to_numpy()
            pfpr = group["pfpr_2to10"].to_numpy()

            ax.plot(betas, pfpr, ".", label="Data", color="black")
            X = np.linspace(1e-4, 10, 10000)
            try:
                Y = sigmoid(np.log10(X), *coefs)
                ax.plot(X, Y, color="red", label="Fitted Curve")
            except Exception as e:
                print(f"Error fitting sigmoid for Population: {population}, Access: {treatment_access} - {e}")
            ax.set_xscale("log")
            ax.set_xlabel("Beta")
            ax.set_ylabel("pfpr_2to10")
            ax.set_title(f"Population : {population}, Access : {treatment_access}")
            ax.legend(fontsize=7)
            ax.set_xlim(1e-3, 10)
            ax.set_ylim(0, 1)
    fig.suptitle("pfPr vs. Beta Data and Curve Fits by Population & Treatment Access", fontsize=24)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(Path("images") / country.country_code / f"{country.country_code}_log_sigmoid_fit.png")
    logger.info(
        f"Saved plot to {Path('images') / country.country_code / f'{country.country_code}_log_sigmoid_fit.png'}"
    )
    return fig
    # plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Calibrate MaSim model for a given country.")
    parser.add_argument("country_code", type=str, help="Country code for calibration (e.g., 'UGA').")
    parser.add_argument(
        "-r",
        "--repetitions",
        type=int,
        default=20,
        help="Number of repetitions per parameter combination (default: 20).",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="output",
        help="Directory to store output files (default: 'output').",
    )
    parser.add_argument(
        "-s",
        "--population_scalar",
        type=float,
        default=1.0,
        help="Population scale (0.1-1.0)",
    )
    parser.add_argument(
        "-n",
        "--node_name",
        type=str,
        default="",
        help="Node name for PBS job submission (default: '' for max queue to run on all hosts).",
    )
    args = parser.parse_args()

    calibrate(args.country_code, args.repetitions, args.population_scalar, args.output_dir, args.node_name)


if __name__ == "__main__":
    main()
