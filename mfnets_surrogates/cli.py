"""The command-line interface for MFNets-Surrogates."""
from pathlib import Path
from typing import Any

import typer
import yaml
from rich.console import Console

from mfnets_surrogates.config import Config
from mfnets_surrogates.net_jax import MFNetJax

# Create a Typer application
app = typer.Typer()

# Create a console for rich text output
console = Console()


def _load_and_validate_config(config_path: Path) -> Config:
    """Load and validate the YAML config file using Pydantic."""
    console.print(f"Loading configuration from: [bold cyan]{config_path}[/]")
    try:
        with open(config_path, "r") as f:
            raw_config = yaml.safe_load(f)
        return Config(**raw_config)
    except Exception as e:
        console.print(f"[bold red]Error parsing configuration file:[/]\n{e}")
        raise typer.Exit(code=1)


def _build_mfnet_from_config(config: Config) -> MFNetJax:
    """Build the MFNetJax object from the configuration."""
    console.print("Building MFNetJax graph...")
    # TODO: Implement the logic to parse the config and initialize models.
    console.print("[yellow]Skipping graph build (not implemented).[/]")
    # Placeholder return
    return None  # type: ignore


def _load_training_data(config: Config) -> Any:
    """Load all training datasets specified in the config."""
    console.print("Loading training data...")
    # TODO: Loop through config.datasets, find type=='training', load .npz
    console.print("[yellow]Skipping data loading (not implemented).[/]")
    # Placeholder return
    return None


def _train_network(mfnet: MFNetJax, data: Any, params: Any) -> MFNetJax:
    """Run the training loop."""
    console.print("Starting training process...")
    # TODO: Implement the Optax training loop.
    console.print("[yellow]Skipping training (not implemented).[/]")
    # Placeholder return
    return mfnet


def _process_prediction_tasks(mfnet: MFNetJax, config: Config) -> None:
    """Run and save all prediction tasks specified in the config."""
    console.print("Processing prediction tasks...")
    # TODO: Loop through config.datasets, find type=='prediction', run, save.
    console.print("[yellow]Skipping predictions (not implemented).[/]")


@app.command()
def run(
    config_path: Path = typer.Option(
        ...,
        "--config",
        "-c",
        help="Path to the YAML configuration file.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    )
) -> None:
    """
    Build, train, and run predictions for an MFNets surrogate model.
    """
    # 1. Load and validate the configuration
    config = _load_and_validate_config(config_path)

    # 2. Build the MFNetJax object from the config
    mfnet = _build_mfnet_from_config(config)

    # 3. Load training data
    training_data = _load_training_data(config)

    # 4. Train the network
    trained_mfnet = _train_network(mfnet, training_data, config.training)

    # 5. Run prediction tasks
    _process_prediction_tasks(trained_mfnet, config)

    console.print("\n[bold green]CLI tool finished successfully.[/]")


if __name__ == "__main__":
    app()
