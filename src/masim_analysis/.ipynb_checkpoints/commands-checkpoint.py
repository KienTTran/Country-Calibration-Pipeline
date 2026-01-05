"""Command-generation utilities for MaSim calibration and batch runs.

This module exposes helpers to generate shell commands that invoke the
MaSim binary (`bin/MaSim`) across parameter sweeps and to build PBS job
scripts for cluster submission. Typical workflow:

- Use `generate_commands` to create per-strategy command lists.
- Use `batch_generate_commands` to collect commands for a directory of YAML
    configuration files.
- Use `generate_job_file` to convert a commands file into a PBS submission
    script that runs commands in parallel while respecting a per-node slot
    limit.

Examples
--------
Create commands for a single YAML and write them to a file::

        filename, cmds = generate_commands('conf/moz/AL5.yml', 'output/moz', repetitions=5)
        with open(filename, 'w') as fh:
                fh.writelines(cmds)

Create a PBS job wrapper for the generated commands::

        generate_job_file(filename, node=Cluster.ONE, job_name='moz_cal', time_override=24)

Note: the module assumes `bin/MaSim` exists and that MaSim produces `.db`
files as outputs; it does not run or validate MaSim itself.
"""

import argparse
import os
from enum import Enum
from pathlib import Path
from typing import Optional


class Cluster(Enum):
    ONE = "nd01"
    TWO = "nd02"
    THREE = "nd03"
    FOUR = "nd04"


# Multiprocessing notes:
# - We can specify the number of repetitions for each strategy using the -j flag
# - Previous implementation seems to have caused the cluster to run out of memory
# Need to investigate adding nodes to the job and maybe automatically calculating
# the number of nodes and clusters needed based on the number of repetitions. If
# you specify multiple nodes, the jobs system will requisition cores equal to the
# number of nodes times the number of cores per node. For example, if you specify
# 2 nodes and 15 cores per node, the job system will requisition 30 cores.
# - There is no reason NOT to parallelize the jobs, so I don't know to what extent
# using the -j flag is useful on the cluster. It might be useful for local testing.

# We can run approximately 16 jobs simultaneously on a single node
# We can run approximately 10 sequential job-batches on a single node (48 / 4.5)
# We can run approximately 160 jobs on a single node (16 * 10) for the max wall time


def generate_commands(
    input_configuration_file: Path | str,
    output_directory: Path | str,
    repetitions: int = 1,
    use_pixel_reporter: bool = False,
) -> tuple[str, list[str]]:
    """Create MaSim command strings for one configuration file.

    The function returns a filename (suggested local text file name) and a
    list of shell command strings. Each command uses the pattern:

    ``./bin/MaSim -i <input.yml> -o <output_prefix>_ -r <Reporter> -j <rep_index>``

    Parameters
    ----------
    input_configuration_file
        Path to a single MaSim YAML configuration file (e.g. in ``conf/<country>/``).
    output_directory
        Directory where MaSim `.db` outputs should be written.
    repetitions
        Number of independent repetitions to generate commands for.
    use_pixel_reporter
        If True use the `SQLitePixelReporter`, otherwise use
        `SQLiteDistrictReporter`.

    Returns
    -------
    tuple[str, list[str]]
        Suggested commands filename (string) and a list of commands.
    """
    # Convert to Path objects for consistent handling
    input_path = Path(input_configuration_file)
    output_path = Path(output_directory)

    # Check if the output directory exists
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    # Parse the strategy name
    strategy_name = input_path.stem
    commands_filename = f"{strategy_name}_{repetitions}reps.txt"
    commands_path = Path(commands_filename)
    if commands_path.exists():
        commands_path.unlink()
    commands = []
    output_file = output_path / strategy_name
    if use_pixel_reporter:
        reporter = "SQLitePixelReporter"
    else:
        reporter = "SQLiteMonthlyReporter"
    for i in range(repetitions):
        commands.append(f"./bin/MaSim -i {input_path} -o {output_file}_ -r {reporter} -j {i}\n")
    return commands_filename, commands


def batch_generate_commands(
    input_configuration_directory: Path | str,
    output_directory: Path | str,
    repetitions: int = 1,
) -> list[str]:
    """Discover YAML files under ``input_configuration_directory`` and
    generate a flattened list of MaSim commands for all of them.

    Parameters
    ----------
    input_configuration_directory
        Root directory containing one or more configuration ``.yml`` files.
    output_directory
        Where MaSim outputs should be written for each configuration.
    repetitions
        Number of repetitions per configuration.

    Returns
    -------
    list[str]
        Flattened list of shell command strings.
    """
    # Convert to Path objects for consistent handling
    input_path = Path(input_configuration_directory)
    output_path = Path(output_directory)

    commands = []
    for yml_file in input_path.rglob("*.yml"):
        _, new_commands = generate_commands(
            yml_file,
            output_path,
            repetitions,
            # in_parallel=True,
            use_pixel_reporter=False,
        )
        commands.extend(new_commands)
    return commands


def generate_job_file(
    commands_filename: str,
    node: Cluster = Cluster.ONE,
    job_name: Path | str = Path("MyJob"),
    cores_override: Optional[int] = None,
    time_override: Optional[int] = 48,
    std_output_location: Optional[Path | str] = Path("."),
    std_error_location: Optional[Path | str] = Path("."),
    email: Optional[str] = None,
) -> None:
    """Write a PBS job script that executes commands from a file in parallel.

    The produced ``.pbs`` script will read the commands file line-by-line,
    launching commands as background jobs while ensuring the number of
    concurrent jobs does not exceed the requested ``cores_requested``.

    Parameters
    ----------
    commands_filename
        Path to a newline-delimited file containing shell commands to run.
    node
        Which cluster host to request (values defined in the ``Cluster`` Enum).
    job_name
        Base job name used for PBS metadata and for per-command stdout/stderr
        files produced by the job wrapper.
    cores_override
        Optional hint for how many concurrent jobs to allow; if omitted the
        local CPU count is used.
    time_override
        Walltime in hours requested for the PBS job.
    std_output_location, std_error_location
        Directories where per-task stdout/stderr files are written.
    email
        Optional email for PBS job events.

    Side effects
    ------------
    - Writes ``{job_name}.pbs`` in the current working directory.
    - Prints instructions to submit the produced script with ``qsub``.
    """
    # Generate the job file
    job_filename = f"{job_name}.pbs"
    if os.path.exists(job_filename):
        os.remove(job_filename)

    # Normalize available CPU count to a non-None int
    available_cpus = os.cpu_count() or 1

    if cores_override is None:
        cores_requested = available_cpus
    elif cores_override is not None and cores_override <= available_cpus:
        cores_requested = cores_override
    else:
        cores_requested = available_cpus

    wait_for_available_slot = (
        "function wait_for_available_slot() {\n"
        f"while (( $(jobs -r | wc -l) >= {cores_requested} )); do\n"
        "    sleep 0.5 \n"  # <<< check every 0.5s if a slot is free
        "done\n"
        "}\n"
    )
    command_loop = (
        'while IFS= read -r line || [[ -n "$line" ]]; do\n'
        '    if [[ -n "$line" ]]; then\n'
        "        wait_for_available_slot\n"  # <<< wait if too many jobs
        '        echo "Running: $line" \n'
        f'        eval "$line" > {std_output_location}/{job_name}_${{INDEX}}.output 2> {std_error_location}/{job_name}_${{INDEX}}.error & \n'
        "        ((INDEX++))\n"
        "    fi\n"
        f"done < {commands_filename}\n"
    )

    with open(job_filename, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"#PBS -l walltime={time_override}:00:00\n")
        f.write(f"#PBS -N {job_name}\n")
        f.write("#PBS -q nd\n")
        f.write(f"#PBS -l select=1:ncpus={cores_requested}:host={node.value}\n")
        f.write(f"#PBS -o {std_output_location}/{job_name}.output\n")
        f.write(f"#PBS -e {std_error_location}/{job_name}.error\n")
        if email:
            f.write("#PBS -m bae\n")
            f.write(f"#PBS -M {email}\n")
        f.write("\n")
        f.write("cd $PBS_O_WORKDIR\n")
        f.write(wait_for_available_slot)
        f.write("\n")
        f.write(command_loop)
        f.write("\n")
    # Display the commands file
    print(f"Job file generate for {commands_filename} and saved to: {job_filename}")
    run_command = f"qsub {job_filename}"
    print(f"To submit the job, run: {run_command}")


def setup_directories(country_code: str) -> None:
    """Create the standard per-country directory layout used by workflows.

    The function creates the following directories under the repository root
    (if they do not already exist):

    - ``conf/{country_code}``
    - ``data/{country_code}``
    - ``images/{country_code}``
    - ``log/{country_code}``
    - ``output/{country_code}`` (and subfolders ``calibration`` and ``validation``)

    Parameters
    ----------
    country_code
        Short country code used to name the directories (e.g. ``'moz'``).
    """
    os.makedirs(f"./conf/{country_code}", exist_ok=True)
    os.makedirs(f"./data/{country_code}", exist_ok=True)
    os.makedirs(f"./images/{country_code}", exist_ok=True)
    os.makedirs(f"./log/{country_code}", exist_ok=True)
    os.makedirs(f"./output/{country_code}", exist_ok=True)
    os.makedirs(f"./output/{country_code}/calibration", exist_ok=True)
    os.makedirs(f"./output/{country_code}/validation", exist_ok=True)
    # os.makedirs(f"./scripts/{country_code}", exist_ok=True)

    # Check for required raster files in ./data/{country_code}
    data_dir = Path(f"./data/{country_code}")
    required_patterns = ["*_pfpr2to10.asc", "*_treatmentseeking.asc", "*_initialpopulation.asc", "*_traveltime.asc"]
    missing_files = []
    for pattern in required_patterns:
        if not any(data_dir.glob(pattern)):
            missing_files.append(pattern)
    if missing_files:
        print(f"Error: Missing required raster files in {data_dir}: {', '.join(missing_files)}")
    else:
        print(f"All required raster files found in {data_dir}.")


def main():
    parser = argparse.ArgumentParser(description="MaSim Simulation Control and Analysis Command Utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)  # Commands set

    # ---- Basic utility commands contained in this module ----
    # Generate commands
    gen_cmd = subparsers.add_parser("generate", help="Generate MaSim commands")
    gen_cmd.add_argument("-c", "--configuration", type=str, help="Path to configuration (.yml) file", required=True)
    gen_cmd.add_argument(
        "-o", "--output_directory", type=str, default="", help="Output directory for simulation results"
    )
    gen_cmd.add_argument(
        "-n", "--name", type=str, default=None, help="Name for the strategy (overrides configuration file name)"
    )
    gen_cmd.add_argument(
        "-r",
        "--repetitions",
        type=int,
        default=1,
        help="Number of repetitions per parameter combination.",
    )
    gen_cmd.add_argument(
        "-p", "--reporter", action="store_true", default=False, help="Use pixel reporter instead of district reporter"
    )

    # Batch generate commands
    batch_cmd = subparsers.add_parser("batch", help="Batch generate MaSim commands from multiple configuration files")
    batch_cmd.add_argument("-i", "--input", type=str, help="Input configuration directory", required=True)
    batch_cmd.add_argument(
        "-o", "--output", type=str, help="Output directory for simulation results", default="./output"
    )
    batch_cmd.add_argument("-r", "--repetitions", type=int, default=1)
    batch_cmd.add_argument("-n", "--name", type=str, default="batch_commands.txt", help="Name for the commands file")

    # Generate job file
    job_cmd = subparsers.add_parser("job", help="Generate a PBS job file")
    job_cmd.add_argument("-f", "--filename", type=str, help="Commands filename", required=True)
    job_cmd.add_argument(
        "-d",
        "--node",
        type=str,
        choices=[c.value for c in Cluster],
        default=Cluster.ONE.value,
        help="Cluster node to use",
    )
    job_cmd.add_argument("-n", "--job_name", type=str, default="MyJob", help="Name of the job")
    job_cmd.add_argument("-c", "--cores_override", type=int, default=None, help="Override number of cores per node")
    job_cmd.add_argument("-t", "--time_override", type=int, default=48, help="Override wall time in hours")
    job_cmd.add_argument("-o", "--std_output_location", type=str, default="", help="Standard output location")
    job_cmd.add_argument("-e", "--std_error_location", type=str, default="", help="Standard error location")
    job_cmd.add_argument("-m", "--email", type=str, default=None, help="Email for job notifications")

    # ---- Additional commands for MaSimAnalysis processes and procedures defined in separate modules ----
    # === Full calibration ===
    # calibrate_cmd = subparsers.add_parser("calibrate", help="Calibrate model parameters for a specific country")
    # calibrate_cmd.add_argument("country_code", type=str, help="Country code for calibration (e.g., 'UGA').")
    # calibrate_cmd.add_argument(
    #     "-r",
    #     "--repetitions",
    #     type=int,
    #     default=20,
    #     help="Number of repetitions per parameter combination (default: 20).",
    # )
    # calibrate_cmd.add_argument(
    #     "-o",
    #     "--output_dir",
    #     type=str,
    #     default="output",
    #     help="Directory to store output files (default: 'output').",
    # )
    # Note: The calibration process is several steps with potential failure points along the way. It may or
    # may not be useful to break out that process into several sub-process commands accessible from this
    # command line interface. That said, the WHOLE POINT of the the calibration process reform is to make it
    # so that it is a one-shot fire and forget it process that generates the beta mapping models. Given that,
    # if it has the correct inputs and has been thoroughly tested, it should just work. Failures should be
    # rare and likely due to bad input data or an edge case bug.

    setup_cmd = subparsers.add_parser("setup", help="Set up directory structure for a new country model")
    setup_cmd.add_argument(
        "country_code", type=str, help="Country code for the new model (e.g., 'uga', 'ago', 'moz', 'rwa', et cetera)."
    )

    args = parser.parse_args()

    if args.command == "generate":
        filename, commands = generate_commands(
            args.configuration,
            args.output_directory,
            args.repetitions,
            args.reporter,
        )
        if args.name:
            filename = args.name
        with open(filename, "w") as f:
            f.writelines(commands)
        print(f"Commands written to {filename}")

    elif args.command == "batch":
        commands = batch_generate_commands(
            args.configuration,
            args.output_directory,
            args.repetitions,
        )
        with open(args.name, "w") as f:
            f.writelines(commands)
        print(f"Batch commands written to {args.name}")

    elif args.command == "job":
        node = Cluster(args.node)
        generate_job_file(
            args.commands_filename,
            node=node,
            job_name=args.job_name,
            cores_override=args.cores_override,
            time_override=args.time_override,
            std_output_location=args.std_output_location,
            std_error_location=args.std_error_location,
            email=args.email,
        )
    # elif args.command == "calibrate":
    #     calibrate(args.country_code, args.repetitions, args.output_dir)
    elif args.command == "setup":
        setup_directories(args.country_code.lower())
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
