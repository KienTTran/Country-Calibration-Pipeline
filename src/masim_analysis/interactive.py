"""
Interactive terminal application for MaSimAnalysis commands.

Has the following commands from the masim_analysis package:

- Pipelines
    - Calibration: `calibrate.calibrate()`
    - Validation: `validate.validate()`
    - Directory setup: `commands.setup_directories()`
- General commands
    - Generate commands: `commands.generate_commands()`
    - Batch generate commands: `commands.batch_generate_commands()`
    - Generate job file: `commands.generate_job_file()`
    - Get average summary statistics from simulation output: `analysis.get_average_summary_statistics()`
    - Post-process validation results: `validate.post_process()`


"""

from rich.console import Console
from rich.prompt import Prompt, IntPrompt, Confirm
from rich.panel import Panel
from rich.table import Table

from masim_analysis.commands import (
    generate_commands,
    batch_generate_commands,
    generate_job_file,
    Cluster,
)
from masim_analysis.calibrate import calibrate
from masim_analysis.commands import setup_directories


console = Console()


def show_menu():
    """Display the main menu."""
    table = Table(title="MaSimAnalysis Interactive Menu", show_header=True)
    table.add_column("Option", style="cyan", width=8)
    table.add_column("Command", style="magenta")

    table.add_row("1", "Generate simulation commands")
    table.add_row("2", "Batch generate commands")
    table.add_row("3", "Generate PBS job file")
    table.add_row("4", "Calibrate model")
    table.add_row("5", "Setup new country directories")
    table.add_row("q", "Quit")

    console.print(table)


def interactive_generate():
    """Interactive command generation."""
    console.print(Panel("[bold cyan]Generate Simulation Commands[/bold cyan]"))

    config_file = Prompt.ask("Configuration file path")
    output_dir = Prompt.ask("Output directory", default="./output")
    repetitions = IntPrompt.ask("Number of repetitions", default=1)
    use_pixel = Confirm.ask("Use pixel reporter?", default=True)

    log_dir = Prompt.ask("Log directory", default="./jobs/logs")
    filename, commands = generate_commands(config_file, output_dir, log_dir, repetitions, use_pixel)

    with open(filename, "w") as f:
        f.writelines(commands)

    console.print(f"[green]✓[/green] Commands written to {filename}")


def interactive_batch():
    """Interactive batch command generation."""
    console.print(Panel("[bold cyan]Batch Generate Commands[/bold cyan]"))

    input_dir = Prompt.ask("Input configuration directory")
    output_dir = Prompt.ask("Output directory", default="./output")
    repetitions = IntPrompt.ask("Number of repetitions", default=1)
    filename = Prompt.ask("Output filename", default="batch_commands.txt")

    commands = batch_generate_commands(input_dir, output_dir, repetitions)

    with open(filename, "w") as f:
        f.writelines(commands)

    console.print(f"[green]✓[/green] Batch commands written to {filename}")


def interactive_job():
    """Interactive job file generation."""
    console.print(Panel("[bold cyan]Generate PBS Job File[/bold cyan]"))

    commands_file = Prompt.ask("Commands filename")
    job_name = Prompt.ask("Job name", default="MyJob")
    node = Prompt.ask("Cluster node", choices=["nd01", "nd02", "nd03", "nd04"], default="nd01")
    cores = IntPrompt.ask("Number of cores (0 for max)", default=0)
    time_hrs = IntPrompt.ask("Wall time (hours)", default=48)
    email = Prompt.ask("Email for notifications (optional)", default="")

    generate_job_file(
        commands_file,
        node=Cluster(node),
        job_name=job_name,
        cores_override=cores if cores > 0 else None,
        time_override=time_hrs,
        email=email if email else None,
    )


def interactive_calibrate():
    """Interactive model calibration."""
    console.print(Panel("[bold cyan]Calibrate Model[/bold cyan]"))

    country_code = Prompt.ask("Country code (e.g., UGA, RWA)").upper()
    repetitions = IntPrompt.ask("Number of repetitions", default=20)
    output_dir = Prompt.ask("Output directory", default="output")

    console.print(f"[yellow]Starting calibration for {country_code}...[/yellow]")
    calibrate(country_code, repetitions, output_dir)
    console.print("[green]✓[/green] Calibration complete!")


def interactive_setup():
    """Interactive directory setup."""
    console.print(Panel("[bold cyan]Setup New Country Directories[/bold cyan]"))

    country_code = Prompt.ask("Country code (e.g., uga, rwa)").lower()

    setup_directories(country_code)
    console.print(f"[green]✓[/green] Directories created for {country_code}")


def main():
    """Main interactive loop."""
    console.print("[bold green]MaSimAnalysis Interactive Terminal[/bold green]\n")

    while True:
        show_menu()
        choice = Prompt.ask("\nSelect an option", choices=["1", "2", "3", "4", "5", "q"])

        console.print()  # blank line

        if choice == "1":
            interactive_generate()
        elif choice == "2":
            interactive_batch()
        elif choice == "3":
            interactive_job()
        elif choice == "4":
            interactive_calibrate()
        elif choice == "5":
            interactive_setup()
        elif choice == "q":
            console.print("[yellow]Goodbye![/yellow]")
            break

        console.print()  # blank line


if __name__ == "__main__":
    main()
