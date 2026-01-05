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
from masim_analysis.commands import batch_generate_commands
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
    country: CountryParams, access_rates: list[float], repetitions: int = 20, population_scalar: float = 1.0, output_directory: Path = Path("output"),
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
        repetitions,
    )
    return cmds


def check_missing_runs(
    country_code: str,
    access_rates: list[float],
    output_dir: Path | str,
    repetitions: int = 20,
) -> list[str]:
    """Check `output/<country>/calibration` for missing MaSim result files.

    The function inspects the expected file naming pattern
    ``cal_<pop>_<access>_<beta>_monthly_data_<iteration>.db`` for all
    ``POPULATION_BINS`` x ``access_rates`` x ``BETAS`` x ``repetitions`` and
    returns a list of command strings to re-run for any missing files.

    Parameters
    ----------
    country_code
        Country code used to construct the output folder path.
    access_rates
        Treatment access rates (floats) expected in the outputs.
    output_dir
        Base output directory (usually ``output``) containing per-country
        `calibration/` folders.
    repetitions
        Number of repetitions that should exist for each parameter set.

    Returns
    -------
    list[str]
        List of commands (as strings) that the caller can pass to
        ``utils.multiprocess`` to attempt reruns for missing outputs.
    """
    base_file_path = os.path.join(output_dir, country_code, "calibration")
    missing_cmds: list[str] = []
    for pop in POPULATION_BINS:
        for access in access_rates:
            for beta in BETAS:
                for i in range(repetitions):
                    filename = f"cal_{pop}_{access}_{beta}_monthly_data_{i + 1}"
                    file = os.path.join(base_file_path, f"{filename}.db")
                    try:
                        # Attempt to read the monthly data and monthly site data
                        # If the file exists, it will be read successfully
                        _months = analysis.get_table(file, "monthly_data")
                        _monthlysitedata = analysis.get_table(file, "monthly_site_data_district")
                    except FileNotFoundError:
                        # with open(f"missing_calibration_runs_{pop}.txt", "a") as f:
                        #     # f.write(f"{e}\n")
                        #     f.write(
                        #         f"./bin/MaSim -i ./conf/{country_code}/calibration/cal_{pop}_{access}_{beta}.yml -o ./output/{country_code}/calibration/cal_{pop}_{access}_{beta}_ -r SQLiteMonthlyReporter -j {i + 1}\n"
                        #     )
                        # if not os.path.exists(f"missing_calibration_runs_{pop}_job.sh"):
                        #     with open(f"missing_calibration_runs_{pop}_job.sh", "w") as f:
                        #         f.write("#!/bin/sh\n")
                        #         f.write("#PBS -l walltime=48:00:00\n")
                        #         f.write(f"#PBS -N MissingCalibrationRuns_{pop}\n")
                        #         f.write("#PBS -q normal\n")
                        #         f.write("#PBS -l nodes=4:ppn=28\n")
                        #         f.write("cd $PBS_O_WORKDIR\n")
                        #         f.write(f"torque-launch missing_calibration_runs_{pop}.txt\n")
                        missing_cmds.append(
                            f"./bin/MaSim -i ./conf/{country_code}/calibration/cal_{pop}_{access}_{beta}.yml -o ./output/{country_code}/calibration/cal_{pop}_{access}_{beta}_ -r SQLiteMonthlyReporter -j {i + 1}"
                        )
                        continue
    return missing_cmds


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


def create_beta_map(
    models_map: dict[float, dict[int, list[float]]],
    population_raster: npt.NDArray,
    access_rate_raster: npt.NDArray,
    prevalence_raster: npt.NDArray,
) -> npt.NDArray:
    """Generate a raster of `beta` values from fitted models and inputs.

    For each cell in ``population_raster`` the function selects the nearest
    population key from ``models_map`` and uses the access-rate cell and the
    observed pfpr (from ``prevalence_raster``) to invert the fitted sigmoid
    and compute an estimated ``beta``. Cells with NaN inputs are preserved as
    NaN in the output.

    Parameters
    ----------
    models_map
        Nested mapping: access_rate -> population_bin -> fitted coefficients
        (``[a, b, c]``) returned by the fitting routines.
    population_raster
        2-D array of per-cell population values.
    access_rate_raster
        2-D array of per-cell treatment access rates (floats).
    prevalence_raster
        2-D array of per-cell PfPR (fractions 0..1) used to infer beta.

    Returns
    -------
    numpy.ndarray
        2-D array shaped like ``population_raster`` containing estimated
        ``beta`` values on a per-cell basis.
    """
    # Create a beta map
    beta_map = np.zeros_like(population_raster)
    # Naive implementation of beta map
    rows, cols = beta_map.shape
    for r in range(rows):
        for c in range(cols):
            beta_map[r, c] = get_beta(
                models_map, access_rate_raster[r, c], population_raster[r, c], prevalence_raster[r, c]
            )
    return beta_map


def get_beta(
    models_map: dict[float, dict[int, list[float]]], access_rate: float, population: int, pfpr: float
) -> float:
    """Get the beta value for a given access rate, population, and pfpr target.

    Retrieve an estimated ``beta`` for a single cell using the fitted model
    coefficients in ``models_map``.

    Behavior notes
    --------------
    - If ``population <= 10`` the function returns ``0.0`` (too small to model).
    - If the access rate or population key is missing in ``models_map`` the
      function logs an error and returns ``np.nan``.
    - The inversion uses the closed-form inverse of the sigmoid on the log10
      scale and returns ``10**beta_log``.

    Parameters
    ----------
    models_map
        Mapping access_rate -> population_bin -> coefficients ``[a, b, c]``.
    access_rate
        Treatment access rate (float) used to select the correct model.
    population
        Population integer value used to select the nearest population bin.
    pfpr
        Observed PfPR (fraction 0..1) to invert into a beta value.

    Returns
    -------
    float
        Estimated beta (positive float) or ``np.nan``/``0.0`` depending on
        the error case or small-population rule.
    """
    if np.isnan(access_rate) or np.isnan(population):
        return np.nan
    # Find which population key to use by searching for the largest population less than or equal to the given population
    populations = np.asarray(list(models_map[access_rate].keys())).squeeze()
    if population <= 10:
        # population = 10 # Maybe this should simply return a beta of 0?
        return 0.0  ### <-- This is a change to return 0.0 for small populations
    else:
        population_key = np.argwhere(populations <= population).squeeze().tolist()
        if type(population_key) is list:
            if len(population_key) > 0:
                population = int(populations[population_key[-1]])
        else:
            population = int(populations[population_key])
    # Get the model
    a = 0.0
    b = 0.0
    c = 0.0
    try:
        a, b, c = models_map[access_rate][population]
    except TypeError:
        coefs = models_map[access_rate][population]
        a = coefs[0]
        b = coefs[1]
        c = coefs[2]
    except KeyError as e:
        logging.error(f"KeyError: {e} for access rate {access_rate} and population {population}")
        return np.nan
    except ValueError as e:
        logging.error(f"ValueError: {e} for access rate {access_rate} and population {population}")
        logging.error(f"Received the following coefficients: {models_map[access_rate][population]}")
        return 0.0
    # SMOOTH OUT THE BETA VALUE
    # b *= 1.25
    # Get the beta value
    try:
        beta_log = c - (1 / b) * np.log(a / pfpr - 1)
    except ZeroDivisionError:
        beta_log = np.nan
    beta = 10**beta_log
    if np.isnan(beta):
        return 0
    return beta


def predicted_prevalence(models_map, population_raster, treatment, beta_map):
    """Compute predicted PfPR map from a beta map and fitted models.

    This function applies the forward sigmoid (using `sigmoid`) to the log10
    of per-cell beta values to produce a predicted PfPR map. It mirrors the
    logic of `create_beta_map` but performs the forward prediction step.

    Parameters
    ----------
    models_map
        Nested mapping of fitted model coefficients per access rate and
        population bin.
    population_raster
        2-D population array.
    treatment
        2-D treatment access-rate raster array.
    beta_map
        2-D beta raster (same shape as `population_raster`).

    Returns
    -------
    numpy.ndarray
        2-D array of predicted PfPR (fractions 0..1).
    """
    # Create a PfPR map
    pfpr_map = np.zeros_like(population_raster)
    # Naive implementation of PfPR map
    rows, cols = pfpr_map.shape

    for r in range(rows):
        for c in range(cols):
            if np.isnan(treatment[r, c]) or np.isnan(population_raster[r, c]) or np.isnan(beta_map[r, c]):
                pfpr_map[r, c] = np.nan
                continue
            access_rate = treatment[r, c]
            population = population_raster[r, c]
            populations = np.asarray(list(models_map[access_rate].keys())).squeeze()
            if population <= 10:
                population = 10
            else:
                population_key = np.argwhere(populations <= population).squeeze().tolist()
                if type(population_key) is list:
                    if len(population_key) > 0:
                        population = int(populations[population_key[-1]])
                else:
                    population = int(populations[population_key])
            coefs = models_map[access_rate][population]
            try:
                pfpr_map[r, c] = sigmoid(np.log(beta_map[r, c]), *coefs)
            except Exception as e:
                logging.error(f"Error occurred while calibrating PfPR at ({r}, {c}): {e}")
                pfpr_map[r, c] = 0.0
    return pfpr_map


def get_last_year_statistics(
    ave_cases: DataFrame,
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
        ave_prevalence : DataFrame
            The data frame containing average prevalence data
        ave_population : DataFrame
            The data frame containing average population data

    Returns
    -------
    A tuple containing three DataFrames: mean_cases, mean_prevalence, mean_population
    """
    months = ave_cases["monthly_data_id"].unique()
    end_month = months[-13]
    start_month = end_month - 12

    mean_cases = (
        ave_cases.loc[ave_cases["monthly_data_id"].between(start_month, end_month, inclusive="left")]
        .copy()
        .groupby("unit_id")
        .sum()
    )
    mean_cases = mean_cases.drop(columns=["monthly_data_id"])
    mean_cases = mean_cases.drop(columns=["clinical_episodes"])
    mean_cases["mean"] = mean_cases.mean(axis=1)
    mean_cases["std"] = mean_cases.std(axis=1)

    mean_population = (
        ave_population.loc[ave_population["monthly_data_id"].between(start_month, end_month, inclusive="left")]
        .copy()
        .groupby("unit_id")
        .mean()
    )
    mean_population = mean_population.drop(columns=["monthly_data_id"])
    mean_population = mean_population.drop(columns=["population"])
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
    mean_prevalence_2_to_10 = mean_prevalence_2_to_10.drop(columns=["pfpr_2to10"])
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
    mean_prevalence_under_5 = mean_prevalence_under_5.drop(columns=["pfpr_under5"])
    mean_prevalence_under_5["mean"] = mean_prevalence_under_5.mean(axis=1)
    mean_prevalence_under_5["std"] = mean_prevalence_under_5.std(axis=1)

    return mean_cases, mean_prevalence_2_to_10, mean_prevalence_under_5, mean_population


# ==== Main functionality ====
def run_calibration_simulations(
    country: CountryParams,
    access_rates: list[float],
    repetitions: int,
    population_scalar: float,
    max_workers: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
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
    
    # # write cmds to a file for record-keeping, save in script/country-name
    # country_script_dir = os.path.join("scripts", country.country_code)
    # os.makedirs(country_script_dir, exist_ok=True)
    # country_cmds_path = os.path.join(country_script_dir, "cmds.txt")
    # with open(country_cmds_path, "w") as f:
    #     for cmd in cmds:
    #         f.write(f"{cmd}\n")
    # logger.info(f"Calibration commands written to: {country_cmds_path}")

    # # Execute commands using multiprocessing
    # if max_workers is None:
    #     max_workers = utils.get_optimal_worker_count()

    # logger.info(f"Starting calibration with {max_workers} worker processes...")

    # successful, failed_commands = utils.multiprocess(cmds, max_workers, logger)
    
    logger.info("Running calibration simulations via PBS")

    utils.submit_and_wait_pbs(
        cmds=cmds,
        country_code=country.country_code,
        logger=logger,
    )


    logger.info("\nCalibration completed:")

    
    
    exit()
    # if has_error:
    # logger.info(f"  Successful runs: {successful}")
    # logger.info(f"  Failed runs: {len(failed_commands)}")

    # if failed_commands:
    #     logger.info("Retrying failed commands.")
    #     # Extract the command text
    #     failed_commands = [cmd for (cmd, error) in failed_commands]
    #     successful, failed_commands = utils.multiprocess(failed_commands, max_workers, logger)

    # if failed_commands:
    #     # Save failed commands to a file for debugging
    #     failed_log_path = os.path.join("log", country.country_code, "calibration_failures.txt")
    #     logger.info(f"There are {len(failed_commands)} failed commands. Writing these to a: {failed_log_path}")
    #     os.makedirs(os.path.dirname(failed_log_path), exist_ok=True)

    #     with open(failed_log_path, "w") as f:
    #         f.write(f"Calibration failures for {country.country_code}\n")
    #         f.write(f"Date: {date.today()}\n\n")
    #         for cmd, error in failed_commands:
    #             f.write(f"Command: {cmd}\n")
    #             f.write(f"Error: {error}\n")
    #             f.write("-" * 80 + "\n")

    #     logger.info(f"Failed commands logged to: {failed_log_path}")


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
    for pop in POPULATION_BINS:
        for access in access_rates:
            for beta in BETAS:
                for i in range(1, repetitions + 1):
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

    # summary.to_csv(f"{base_file_path}/calibration_summary.csv")
    return summary


def summarize_calibration_results(country: CountryParams, data_path: Path | str = Path("output")) -> DataFrame:
    data_path = Path(data_path)
    files = data_path.glob("*.db")
       
    data = analysis.get_table(next(files), "monthly_site_data_district")
    end_month = data["monthly_data_id"].unique()[-13]
    summary = DataFrame(
        columns=["population", "access_rate", "beta", "iteration", "pfpr_under5", "pfpr_2to10", "pfpr_all"]
    )
    for file in files:
        print(file)
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

    summary["pfpr_under5"] = summary["pfpr_under5"].div(100)
    summary["pfpr_2to10"] = summary["pfpr_2to10"].div(100)
    summary["pfpr_all"] = summary["pfpr_all"].div(100)
    summary = summary.drop(columns=["iteration"])
    summary = summary.groupby(["population", "access_rate", "beta"]).mean().reset_index()
    return summary
    # summary.to_csv(f"{base_file_path}/calibration_means.csv", index=False)
    # summary.head(25)


def calibrate(country_code: str, repetitions: int, population_scalar: float = 1.0, output_dir: Path | str = Path("output")) -> None:
    """
    Calibrate the MaSim model for a given country.
    """
    # Back up run to ensure output and log directories exist
    # setup_directories(country_code)

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
    run_calibration_simulations(country, access_rates, repetitions, population_scalar, logger=logger)

    # Check for missing runs
    logger.info("Checking for missing calibration runs...")
    missing_cmds = check_missing_runs(country.country_code, access_rates, output_dir, repetitions)
    if missing_cmds:
        logger.info(f"Found {len(missing_cmds)} missing runs. Re-running these simulations...")
        successful, failed_commands = utils.multiprocess(missing_cmds, utils.get_optimal_worker_count(), logger)
        logger.info(f"Re-run completed: {successful} successful, {len(failed_commands)} failed.")
        if failed_commands:
            logger.warning("Some commands still failed after re-run. Check logs for details.")

    # Summarize calibration results
    logger.info("Summarizing calibration results...")
    means = summarize_calibration_results(country, Path("output") / country.country_code / "calibration")
    means.to_csv(Path(output_dir) / country.country_code / "calibration" / "calibration_means.csv", index=False)
    logger.info("Fitting log-sigmoid models to calibration data...")
    models_map = get_beta_models(
        populations=POPULATION_BINS,
        access_rates=access_rates,
        means=means,
        pfpr_cutoff=0.0,
    )

    # Save the models map to a json file
    models_map_filename = "models_map.json"
    with open(Path("data") / country.country_code / "calibration" / models_map_filename, "w") as f:  # noqa: F811, ruff disabled
        json.dump(models_map, f, indent=4)
    logger.info(f"Saved models map to {Path('data') / country.country_code / 'calibration' / models_map_filename}")

    # Plot the log-sigmoid fits

    # Create beta map
    logger.info("Creating beta map...")
    population_raster, meta = utils.read_raster(
        Path("data") / country.country_code / f"{country.country_code}_population_v2.asc"
    )
    access_rate_raster, _ = utils.read_raster(
        Path("data") / country.country_code / f"{country.country_code}_treatmentseeking.asc"
    )
    prevalence_raster, _ = utils.read_raster(
        Path("data") / country.country_code / f"{country.country_code}_pfpr2to10.asc"
    )
    beta_map = create_beta_map(models_map, population_raster, access_rate_raster, prevalence_raster)
    beta_map_filename = Path("data") / country.country_code / f"{country.country_code}_beta.asc"
    utils.write_raster(beta_map, beta_map_filename, meta["xllcorner"], meta["yllcorner"], meta["cellsize"])
    logger.info(f"Saved beta map to {beta_map_filename}")


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
    num_rows = len(POPULATION_BINS)
    num_cols = len(access_rates)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows), sharex=True, sharey=True)
    for i, population in enumerate(POPULATION_BINS):
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
    return fig
    # fig.savefig(Path("images") / country.country_code / f"{country.country_code}_log_sigmoid_fit.png")
    # logger.info(
    #     f"Saved plot to {Path('images') / country.country_code / f'{country.country_code}_log_sigmoid_fit.png'}"
    # )
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
    args = parser.parse_args()

    calibrate(args.country_code, args.repetitions, args.population_scalar, args.output_dir)


if __name__ == "__main__":
    main()
