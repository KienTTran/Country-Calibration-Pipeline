"""
Utilities for plotting and raster file manipulation.

This module provides functions for visualizing district, population, and prevalence
data from raster arrays, as well as reading and writing raster files in ASCII grid format.
"""

import logging
import subprocess
import numpy
import os

from concurrent.futures import ProcessPoolExecutor, as_completed
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from pathlib import Path

from masim_analysis import configure


def plot_districts(
    districts_raster: numpy.ndarray,
    labels: list[str],
    country_name: str,
    fig_size: tuple[int, int] = (10, 10),
    loc=None,
) -> Figure:
    """
    Plot the district mapping of the country according to the raster array.

    Parameters
    ----------
    districts_raster : numpy.typing.NDArray
        Raster array representing the districts.
    labels : list[str]
        List of labels for the districts.
    country_name : str
        Name of the country for the plot title.
    fig_size : tuple[int, int], optional
        Size of the figure, by default (10, 10).
    loc : str, optional
        Location of the legend, by default None.

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib Figure object containing the plot.
    """
    cmap = plt.get_cmap("tab20", len(labels))
    fig, ax = plt.subplots(figsize=fig_size)
    ax.imshow(districts_raster, cmap=cmap)
    ax.set_title(f"{country_name} Districts")
    # create legend handles
    handles = [Patch(color=cmap(i), label=labels[i + 1].replace("_", " ")) for i in range(11)]
    ax.legend(
        # bbox_to_anchor=bbox_to_anchor,
        handles=handles,
        title="Districts",
        loc=loc,
    )
    return fig


def plot_population(
    population_raster: numpy.ndarray,
    country_name: str,
    fig_size: tuple[int, int] = (10, 10),
    population_upper_limit: float = 1000,
) -> Figure:
    """
    Plot the population density of the country according to the raster array.

    Parameters
    ----------
    population_raster : numpy.typing.NDArray
        Raster array representing population density.
    country_name : str
        Name of the country for the plot title.
    fig_size : tuple[int, int], optional
        Size of the figure, by default (10, 10).

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib Figure object containing the plot.
    """
    fig, ax = plt.subplots(figsize=fig_size)
    img = ax.imshow(population_raster, cmap="turbo")
    img.set_clim(0, population_upper_limit)
    ax.set_title(f"{country_name} Population")
    fig.colorbar(img, ax=ax, label="Population")

    return fig


def plot_prevalence(
    prevalence_raster: numpy.ndarray, country_name: str, fig_size: tuple[int, int] = (10, 10)
) -> Figure:
    """
    Plot the prevalence of malaria according to the raster array.

    Parameters
    ----------
    prevalence_raster : numpy.typing.NDArray
        Raster array representing malaria prevalence.
    country_name : str
        Name of the country for the plot title.
    fig_size : tuple[int, int], optional
        Size of the figure, by default (10, 10).

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib Figure object containing the plot.
    """
    fig, ax = plt.subplots(figsize=fig_size)
    img = ax.imshow(prevalence_raster, cmap="coolwarm")
    ax.set_title(f"{country_name} Prevalence")
    fig.colorbar(img, ax=ax, label="Prevalence")
    return fig


def read_raster(file: Path | str) -> tuple[numpy.ndarray, dict]:
    """
    Read in a raster file and return the raster array and metadata.

    Parameters
    ----------
    file : str
        Path to the raster file.

    Returns
    -------
    tuple
        A tuple containing the raster array (numpy.typing.NDArray) and metadata dictionary (dict).
    """
    file_path = Path(file)
    if not file_path.is_file():
        raise FileNotFoundError(f"Raster file not found: {file}")
    with open(file_path, "r") as f:
        data = f.read().splitlines()
    metadata = data[:6]
    data = data[6:]
    metadata = {line.split()[0]: float(line.split()[1]) for line in metadata}
    raster = numpy.zeros((int(metadata["nrows"]), int(metadata["ncols"])))
    for i, line in enumerate(data):
        line = line.split()
        line = numpy.asarray(line, dtype=float)
        raster[i, :] = line
    raster[raster == metadata["NODATA_value"]] = numpy.nan
    return raster, metadata


def write_raster(
    raster: numpy.ndarray, file: Path | str, xllcorner: float, yllcorner: float, cellsize: int = 5000
) -> None:
    """
    Write a raster array to a file.

    Parameters
    ----------
    raster : numpy.typing.NDArray
        The raster array to write.
    file : str
        The path to the output file.
    xllcorner : float
        The x-coordinate of the lower left corner of the raster.
    yllcorner : float
        The y-coordinate of the lower left corner of the raster.
    cellsize : int, optional
        The size of each cell in the raster, by default 5000.
    """
    file_path = Path(file)
    if not file_path.parent.exists():
        raise FileNotFoundError(f"Directory does not exist: {file_path.parent}")
    nrows, ncols = raster.shape
    raster = numpy.where(numpy.isnan(raster), configure.NODATA_VALUE, raster)
    with open(file, "w") as f:
        f.write(f"ncols\t{ncols}\n")
        f.write(f"nrows\t{nrows}\n")
        f.write(f"xllcorner\t{xllcorner}\n")
        f.write(f"yllcorner\t{yllcorner}\n")
        f.write(f"cellsize\t{cellsize}\n")
        f.write(f"NODATA_value\t{configure.NODATA_VALUE}\n")
        for row in raster:
            f.write(" ".join([str(value) for value in row]) + "\n")


# ==== Logger setup ====
def get_country_logger(country_code: str, logger_name: str) -> logging.Logger:
    """
    Returns a logger that writes to log/<country_code>/<logger_name>.log.
    """
    log_dir = Path("log") / country_code
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{logger_name}.log"

    logger = logging.getLogger(f"{country_code}.{logger_name}")
    logger.setLevel(logging.INFO)

    # Avoid adding multiple handlers if logger is reused
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(file_handler)
        # Optional: also log to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(stream_handler)

    return logger


# ==== Generic multiprocessing utilities ====
def get_optimal_worker_count(utilization: float = 1.0) -> int:
    """
    Determine optimal number of worker processes based on system resources.

    Returns
    -------
    int
        Recommended number of worker processes
    """
    cpu_count = os.cpu_count() or 1
    # Use 75% of available CPUs, with a minimum of 1 and maximum of 16
    # This leaves some headroom for the system and prevents oversubscription
    optimal_workers = max(1, int(cpu_count * utilization))
    return optimal_workers


def run_simulation_command(cmd: str) -> tuple[str, bool, str]:
    """
    Execute a single MaSim simulation command.

    Parameters
    ----------
    cmd : str
        The command string to execute

    Returns
    -------
    tuple[str, bool, str]
        A tuple containing (command, success_flag, error_message)
    """
    try:
        # Remove trailing newline if present
        cmd = cmd.strip()
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            # timeout=3600,  # 1 hour timeout per simulation
        )

        if result.returncode == 0:
            return (cmd, True, "")
        else:
            return (cmd, False, result.stderr)

    except subprocess.TimeoutExpired:
        return (cmd, False, "Command timed out after 1 hour")
    except Exception as e:
        return (cmd, False, f"Unexpected error: {str(e)}")


def multiprocess(cmds: list[str], max_workers: int, logger: logging.Logger) -> tuple[int, list[tuple[str, str]]]:
    """
    Generic multiprocessing wrapper for a list of shell commands.
    """
    successful_runs = 0
    failed_runs = 0
    failed_commands = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_cmd = {executor.submit(run_simulation_command, cmd): cmd for cmd in cmds}

        # Process completed jobs with progress bar
        for future in as_completed(future_to_cmd):
            try:
                cmd, success, error_msg = future.result()
                if success:
                    logger.info(f"Successfully executed command: {cmd}")
                    successful_runs += 1
                else:
                    failed_runs += 1
                    failed_commands.append((cmd, error_msg))
                    # Only log first few failures to avoid spam
                    if failed_runs <= 5:
                        logger.error(f"Failed command: {cmd}")
                        logger.error(f"Error: {error_msg}")
                    elif failed_runs == 6:
                        logger.error("Additional failures will be logged to file...")

            except Exception as e:
                failed_runs += 1
                cmd = future_to_cmd[future]
                error_msg = f"Future execution error: {str(e)}"
                failed_commands.append((cmd, error_msg))
                if failed_runs <= 5:
                    logger.error(f"Failed command: {cmd}")
                    logger.error(f"Error: {error_msg}")

    return successful_runs, failed_commands

<<<<<<< HEAD
def prepare_pbs_files(country_code: str, base_dir: str, logger):
    import shutil
=======
def prepare_pbs_files(country_code: str, base_dir: str, logger, rotate_logs: bool = True):
    import os, shutil
    from datetime import datetime
>>>>>>> e3a5426 (Refactor code for v4.2)

    script_dir = os.path.join(base_dir, "script")
    log_dir = os.path.join(base_dir, "log")

    os.makedirs(script_dir, exist_ok=True)
<<<<<<< HEAD
    os.makedirs(log_dir, exist_ok=True)
    
    # If log dir exist, rename it to log_before_<current day>, then create a new log dir
    if os.path.exists(log_dir):
        from datetime import datetime
=======

    if rotate_logs and os.path.exists(log_dir):
>>>>>>> e3a5426 (Refactor code for v4.2)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_log_dir = f"{log_dir}_before_{timestamp}"
        shutil.move(log_dir, new_log_dir)
        logger.info(f"Existing log directory renamed to {new_log_dir}")
<<<<<<< HEAD
        os.makedirs(log_dir, exist_ok=True)
        
=======

    os.makedirs(log_dir, exist_ok=True)
>>>>>>> e3a5426 (Refactor code for v4.2)

    for fname in ("job_template.template", "submit_jobs.template"):
        src = os.path.join("scripts", fname)
        dst = os.path.join(script_dir, fname.replace(".template", ".pbs"))

        with open(src, "r") as f:
            text = f.read().replace("#JOB_NAME#", country_code)

        with open(dst, "w") as f:
            f.write(text)

        os.chmod(dst, 0o755)
<<<<<<< HEAD

=======
>>>>>>> e3a5426 (Refactor code for v4.2)
        logger.info(f"Prepared PBS file: {dst}")

    return script_dir, log_dir

<<<<<<< HEAD
def pbs_counts_qselect(user: str, country_code: str, jobname_target: str) -> tuple[int, int]:
=======

def pbs_counts_qselect(user: str, country_code: str, jobname_target: str) -> tuple[int, int, int]:
    import subprocess
>>>>>>> e3a5426 (Refactor code for v4.2)
    
    total_out_submit = subprocess.run(
        ["qselect", "-u", user, "-N", f"submit_{country_code}"],
        capture_output=True, text=True
    )
    total_out_submit = sum(1 for x in total_out_submit.stdout.splitlines() if x.strip())
    
    total_out = subprocess.run(
        ["qselect", "-u", user, "-N", jobname_target],
        capture_output=True, text=True
    )
    total = sum(1 for x in total_out.stdout.splitlines() if x.strip()) + total_out_submit

    run_out = subprocess.run(
        ["qselect", "-u", user, "-N", jobname_target, "-s", "R"],
        capture_output=True, text=True
    )
    running = sum(1 for x in run_out.stdout.splitlines() if x.strip())

    ended_out = subprocess.run(
        ["qselect", "-u", user, "-N", jobname_target, "-s", "E"],
        capture_output=True, text=True
    )
    ended = sum(1 for x in ended_out.stdout.splitlines() if x.strip())

    return running, ended, total


<<<<<<< HEAD
def submit_and_wait_pbs(cmds, country_code, logger, type,
                        max_active_jobs=500, sleep_sec=15):

    import subprocess, time, getpass
    
    user = getpass.getuser()
    os.makedirs("./jobs", exist_ok=True)
    base_dir = os.path.join("jobs", country_code, type)
    os.makedirs(base_dir, exist_ok=True)    
    cmds_path = os.path.join(base_dir, "cmds.txt")

    # write cmds.txt
    with open(cmds_path, "w") as f:
        for c in cmds:
            f.write(f"{c}")

    logger.info(f"Commands written to {cmds_path}")

    script_dir, log_dir = prepare_pbs_files(country_code, base_dir, logger)
    
    jobname_target = f"{country_code}_single_run"
    
=======
def submit_and_wait_pbs(cmds, country_code, logger, run_type,
                        max_active_jobs=500, sleep_sec=15,
                        rotate_logs=True):

    import os
    import subprocess, time, getpass
    from pathlib import Path

    if not cmds:
        logger.warning("No commands to submit (cmds is empty). Nothing to do.")
        return

    user = getpass.getuser()
    os.makedirs("jobs", exist_ok=True)

    base_dir = os.path.join("jobs", country_code, run_type)
    os.makedirs(base_dir, exist_ok=True)

    cmds_path = os.path.join(base_dir, "cmds.txt")

    # write cmds.txt (one command per line)
    with open(cmds_path, "w") as f:
        for c in cmds:
            f.write(c.rstrip() + "\n")

    logger.info(f"Commands written to {cmds_path}")

    script_dir, log_dir = prepare_pbs_files(country_code, base_dir, logger, rotate_logs)

    # IMPORTANT: this must match the -N used by your dispatcher for the jobs it creates
    jobname_target = f"{country_code}_single_run"

>>>>>>> e3a5426 (Refactor code for v4.2)
    env = (
        f"CMD_FILE=../cmds.txt,"
        f"MAX_ACTIVE_JOBS={max_active_jobs},"
        f"SLEEP={sleep_sec},"
<<<<<<< HEAD
        f"PROJECT_DIR={Path.cwd().resolve()}"
    )

    logger.info("Submitting PBS dispatcher job...")
    
    subprocess.run(
        [
            "qsub",
            "-v", env,
            "-o", "submit.output",
            "-e", "submit.error",
            "submit_jobs.pbs",
        ],
=======
        f"PROJECT_DIR={Path.cwd().resolve()},"
        f"JOBNAME_TARGET={jobname_target}"
    )

    logger.info("Submitting PBS dispatcher job...")

    # write submit logs into base_dir so they don't overwrite each time
    submit_out = os.path.join(base_dir, "submit.output")
    submit_err = os.path.join(base_dir, "submit.error")

    subprocess.run(
        ["qsub", "-v", env, "-o", submit_out, "-e", submit_err, "submit_jobs.pbs"],
>>>>>>> e3a5426 (Refactor code for v4.2)
        cwd=script_dir,
        check=True,
    )

    logger.info(f"Waiting for all PBS jobs named '{jobname_target}' to finish...")
<<<<<<< HEAD

    time.sleep(5)  # allow jobs to appear
    
    total_jobs = len(cmds)

    while True:
        running, ended, total = pbs_counts_qselect(user, country_code, jobname_target)
    
        if total == 0:
            break
        
        #count output files in output dir
        output_dir = os.path.join("output",country_code,type)
        finished_outputs = len([name for name in os.listdir(output_dir) if name.endswith(".db")])
    
        logger.info(f"Submitted jobs {jobname_target}: Running (R): {running} | Finished (E): {ended} | Total: {total} | Outputs: {finished_outputs}/{total_jobs} ({(finished_outputs/total_jobs)*100:.2f}%)")
        time.sleep(sleep_sec)
    
    logger.info(f"All jobs for {jobname_target} finished.")

def check_error_cmds(log_dir: str, logger) -> list[str]:
    error_inputs = []
    for file in os.listdir(log_dir):
        if file.endswith(".error") and os.path.getsize(os.path.join(log_dir, file)) > 0:
            input = file.replace(".error", ".yml")
            error_inputs.append(input)
                
    return error_inputs
=======
    time.sleep(5)  # allow jobs to appear

    total_jobs = len(cmds)
    output_dir = os.path.join("output", country_code, run_type)

    while True:
        running, ended, total = pbs_counts_qselect(user, country_code, jobname_target)

        # If qselect sees none of the target jobs, assume they're done
        if total == 0:
            break

        if os.path.isdir(output_dir):
            finished_outputs = sum(
                1 for name in os.listdir(output_dir) if name.endswith(".db")
            )
        else:
            finished_outputs = 0

        pct = 100.0 * finished_outputs / total_jobs if total_jobs else 0.0
        logger.info(
            f"Jobs {jobname_target}: R={running} | E={ended} | Total={total} | "
            f"Outputs: {finished_outputs}/{total_jobs} ({pct:.2f}%)"
        )

        time.sleep(sleep_sec)

    logger.info(f"All jobs for {jobname_target} finished.")

import os, re

def check_error_cmds(log_dir: str, cmds_path: str, logger) -> list[str]:
    if not os.path.isdir(log_dir):
        return []

    with open(cmds_path, "r") as f:
        cmd_lines = [line.rstrip("\n") for line in f if line.strip()]

    rerun_cmds = []
    seen_idx = set()

    # helper: find cmd indices that reference a given yml base
    def find_indices_by_base(base: str) -> list[int]:
        # matches ... -i .../<base>.yml (token-based-ish)
        pat = re.compile(rf'(^|\s)-i\s+\S*/{re.escape(base)}\.yml(\s|$)')
        hits = [i for i, cmd in enumerate(cmd_lines) if pat.search(cmd)]
        return hits

    for fn in os.listdir(log_dir):
        if not fn.endswith(".error"):
            continue
        full = os.path.join(log_dir, fn)
        if os.path.getsize(full) <= 0:
            continue

        # Case A: base__123.error -> exact index
        m = re.search(r"__(\d+)\.error$", fn)
        if m:
            idx = int(m.group(1))
            if 0 <= idx < len(cmd_lines) and idx not in seen_idx:
                seen_idx.add(idx)
                rerun_cmds.append(cmd_lines[idx])
            continue

        # Case B: base.error -> map by base name
        base = fn[:-len(".error")]
        idxs = find_indices_by_base(base)
        if not idxs:
            logger.warning(f"Cannot map error file to any cmd by base name: {fn}")
            continue

        for idx in idxs:
            if idx not in seen_idx:
                seen_idx.add(idx)
                rerun_cmds.append(cmd_lines[idx])

    return rerun_cmds


>>>>>>> e3a5426 (Refactor code for v4.2)

