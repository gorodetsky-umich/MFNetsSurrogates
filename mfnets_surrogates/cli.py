"""The command-line interface for MFNets-Surrogates."""

import inspect
import time
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np
import typer
import yaml
from rich.console import Console
from rich.table import Table

from mfnets_surrogates import net_jax
from mfnets_surrogates.config import Config, TrainingParams

app = typer.Typer(pretty_exceptions_show_locals=False)
console = Console()


def _discover_initializers() -> dict[str, Callable[..., Any]]:
    """Scan the net_jax module to find all model initializer functions."""
    initializers: dict[str, Callable[..., Any]] = {}
    for name, func in inspect.getmembers(net_jax, inspect.isfunction):
        if name.startswith("init_"):
            model_name = name.replace("init_", "").replace("_", " ").title()
            model_name = model_name.replace(" ", "")
            initializers[model_name] = func
    return initializers


MODEL_INITIALIZERS = _discover_initializers()

ACTIVATION_FUNCTIONS: dict[str, Callable[[jnp.ndarray], jnp.ndarray]] = {
    "relu": jax.nn.relu,
    "tanh": jax.nn.tanh,
}


def _load_and_validate_config(config_path: Path) -> Config:
    """Load and validate the YAML config file using Pydantic."""
    console.print(f"Loading configuration from: [bold cyan]{config_path}[/]")
    try:
        with open(config_path) as f:
            raw_config = yaml.safe_load(f)
        return Config(**raw_config)
    except Exception as e:
        console.print(f"[bold red]Error parsing configuration file:[/]\n{e}")
        raise typer.Exit(code=1) from e


def _load_training_data(
    config: Config,
) -> tuple[dict[str, Any], dict[int | str, tuple[int, int]]]:
    """Load all training datasets and derive model dimensions."""
    console.print("Loading training data...")
    training_datasets = [d for d in config.datasets if d.type == "training"]
    if not training_datasets:
        console.print("[bold red]No training datasets found in config.[/]")
        raise typer.Exit(code=1)

    all_data = {}
    dim_info = {}
    for dataset in training_datasets:
        data = jnp.load(dataset.data_path)
        all_data[dataset.name] = data
        for node_id in dataset.nodes:
            x_key = f"x_train_{node_id}"
            y_key = f"y_train_{node_id}"
            if x_key not in data or y_key not in data:
                console.print(
                    f"[bold red]Error: {x_key} or {y_key} not found in "
                    f"{dataset.data_path}[/]"
                )
                raise typer.Exit(code=1)
            d_in = data[x_key].shape[1]
            d_out = data[y_key].shape[1]
            dim_info[node_id] = (d_in, d_out)
    console.print(
        "Derived model dimensions from data for nodes: "
        f"{list(dim_info.keys())}"
    )
    return all_data, dim_info


def _build_mfnet_from_config(
    config: Config, dim_info: dict[int | str, tuple[int, int]]
) -> net_jax.MFNetJax:
    """Build the MFNetJax object from the configuration."""
    console.print("Building MFNetJax graph from configuration...")
    key = jax.random.PRNGKey(42)

    graph = nx.DiGraph(config.graph["edges"])
    graph.add_nodes_from(config.graph["nodes"])

    for node_id, model_config in config.models.items():
        key, model_key = jax.random.split(key)
        d_in, d_out = dim_info[node_id]

        initializer = next(
            (
                func
                for key, func in MODEL_INITIALIZERS.items()
                if key.lower() == model_config.type.lower()
            ),
            None,
        )
        if initializer is None:
            console.print(
                f"[bold red]Unknown model type: {model_config.type}[/]"
            )
            raise typer.Exit(code=1)

        init_kwargs = model_config.params.copy()
        init_kwargs["d_in"] = d_in
        init_kwargs["d_out"] = d_out

        predecessors = sorted(graph.predecessors(node_id))
        if predecessors:
            d_parent = sum(dim_info[p][1] for p in predecessors)
            if "scaleshift" in model_config.type.lower():
                init_kwargs["d_parent"] = d_parent
            elif "enhancement" in model_config.type.lower():
                init_kwargs["layer_sizes"].insert(0, d_in + d_parent)

        if "activation" in init_kwargs:
            act_str = init_kwargs.pop("activation")
            activation_fn = ACTIVATION_FUNCTIONS.get(act_str)
            if activation_fn is None:
                console.print(f"[bold red]Unknown activation: {act_str}[/]")
                raise typer.Exit(code=1)
            init_kwargs["activation"] = activation_fn

        model_instance = initializer(key=model_key, **init_kwargs)
        graph.add_node(node_id, func=model_instance)

    console.print("[green]Successfully built MFNetJax graph.[/]")
    return net_jax.MFNetJax(graph)


def _train_network(
    mfnet: net_jax.MFNetJax,
    training_data: dict[str, Any],
    training_params: TrainingParams,
    config: Config,
) -> net_jax.MFNetJax:
    """Run the training loop using the high-level .fit() method."""
    console.print(
        f"\nStarting training with {training_params.num_steps} steps..."
    )
    start_time = time.time()

    # 1. Find the training dataset configuration
    dataset_config = next(
        (d for d in config.datasets if d.type == "training"), None
    )
    if not dataset_config:
        console.print(
            "[bold red]Critical error: No training dataset found.[/]"
        )
        raise typer.Exit(code=1)

    # 2. Prepare training data in the format required by .fit():
    # A list of (x_i, y_i) tuples, one for each node.
    data_file = training_data[dataset_config.name]
    target_nodes = tuple(sorted(dataset_config.nodes))
    train_data_for_fit = []
    console.print("Preparing training data for the following nodes:")
    try:
        for node_id in target_nodes:
            x_train = data_file[f"x_train_{node_id}"]
            y_train = data_file[f"y_train_{node_id}"]
            train_data_for_fit.append((x_train, y_train))
            console.print(
                f"  - Node {node_id}: x_shape={x_train.shape}, "
                f"y_shape={y_train.shape}"
            )
    except KeyError as e:
        console.print(f"[bold red]Data key not found in NPZ file: {e}[/]")
        raise typer.Exit(code=1) from e

    # 3. Call the high-level .fit() method
    mfnet.fit(
        train_data=train_data_for_fit,
        n_iters=training_params.num_steps,
        learning_rate=training_params.learning_rate,
        verbose=True,  # Use the fit method's internal progress bar
    )

    duration = time.time() - start_time
    console.print(f"Training completed in {duration:.2f} seconds.")

    # The mfnet object is trained in-place
    return mfnet


def _process_prediction_tasks(mfnet: net_jax.MFNetJax, config: Config) -> None:
    """Run and save all prediction tasks specified in the config."""
    console.print("\nProcessing prediction tasks...")
    prediction_datasets = [
        d for d in config.datasets if d.type == "prediction"
    ]

    if not prediction_datasets:
        console.print(
            "[yellow]No prediction tasks found in configuration.[/yellow]"
        )
        return

    for task in prediction_datasets:
        console.print(
            f"- Running prediction task: [bold cyan]{task.name}[/bold cyan] -"
        )
        try:
            input_data = np.load(task.data_path)
            x_predict = input_data["x_predict"]
            console.print(
                f"Loaded prediction inputs from '{task.data_path}' "
                f"with shape {x_predict.shape}"
            )
        except Exception as e:
            console.print(
                f"[bold red]Error loading data for task '{task.name}': {e}[/]"
            )
            continue

        if len(task.nodes) != 1:
            console.print(
                "[bold red]Prediction tasks must specify exactly one target "
                f"node. Task '{task.name}' has {len(task.nodes)}.[/]"
            )
            continue
        target_node = task.nodes[0]
        console.print(f"Generating predictions for node: {target_node}")

        (y_predict,) = mfnet.run((target_node,), x_predict)
        console.print(f"Generated predictions with shape {y_predict.shape}")

        if task.output_path:
            output_path = Path(task.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(output_path, y_predict=y_predict, x_predict=x_predict)
            console.print(
                f"[green]Success. saving pred. to '{output_path}'[/green]"
            )
        else:
            console.print(
                f"[yellow]No output_path for task '{task.name}'. "
                "Skipping save.[/yellow]"
            )


@app.command()
def list_models() -> None:
    """List all available models and their required parameters."""
    console.print("[bold]Available Models for Configuration:[/bold]")
    table = Table(title="Model Initializers")
    table.add_column("Config Name", style="cyan", no_wrap=True)
    table.add_column("Parameters", style="magenta")

    for name, func in MODEL_INITIALIZERS.items():
        sig = inspect.signature(func)
        params = [
            p
            for p in sig.parameters
            if p not in ["key", "d_in", "d_out", "d_parent"]
        ]
        table.add_row(name, ", ".join(params))

    console.print(table)


@app.command()
def run(
    config_path: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            help="Path to the YAML configuration file.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
) -> None:
    """Build, train, and run predictions for an MFNets surrogate model."""
    config = _load_and_validate_config(config_path)
    training_data, dim_info = _load_training_data(config)
    mfnet = _build_mfnet_from_config(config, dim_info)
    trained_mfnet = _train_network(
        mfnet, training_data, config.training, config
    )
    _process_prediction_tasks(trained_mfnet, config)
    console.print("\n[bold green]CLI tool finished successfully.[/]")


if __name__ == "__main__":
    app()
