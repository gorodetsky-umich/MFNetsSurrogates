"""The command-line interface for MFNets-Surrogates."""
import os
import time
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np
import optax
import typer
import yaml
from jax import tree_util
from rich.console import Console
from rich.progress import track

from mfnets_surrogates.config import Config, TrainingParams
from mfnets_surrogates.net_jax import (
    MFNetJax,
    init_mlp_enhancement_model,
    init_pce_model,
    init_pce_scale_shift_model,
    mse_loss_graph,
)

app = typer.Typer(
    pretty_exceptions_show_locals=False
)
console = Console()

MODEL_INITIALIZERS: dict[str, Callable[..., Any]] = {
    "PCEModel": init_pce_model,
    "PCEScaleShiftModel": init_pce_scale_shift_model,
    "MLPEnhancementModel": init_mlp_enhancement_model,
}
ACTIVATION_FUNCTIONS: dict[str, Callable[[jnp.ndarray], jnp.ndarray]] = {
    "relu": jax.nn.relu,
    "tanh": jax.nn.tanh,
}


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
        f"Derived model dimensions from data for nodes: {list(dim_info.keys())}"
    )
    return all_data, dim_info


def _build_mfnet_from_config(
    config: Config, dim_info: dict[int | str, tuple[int, int]]
) -> MFNetJax:
    """Build the MFNetJax object from the configuration."""
    console.print("Building MFNetJax graph from configuration...")
    key = jax.random.PRNGKey(42)

    graph = nx.DiGraph(config.graph["edges"])
    graph.add_nodes_from(config.graph["nodes"])

    for node_id, model_config in config.models.items():
        key, model_key = jax.random.split(key)
        d_in, d_out = dim_info[node_id]

        initializer = MODEL_INITIALIZERS.get(model_config.type)
        if initializer is None:
            console.print(f"[bold red]Unknown model type: {model_config.type}[/]")
            raise typer.Exit(code=1)

        init_kwargs = model_config.params.copy()

        if model_config.type == "PCEModel":
            init_kwargs["dim_in"] = d_in
            init_kwargs["dim_out"] = d_out
        else:
            init_kwargs["d_in"] = d_in
            init_kwargs["d_out"] = d_out

        predecessors = sorted(list(graph.predecessors(node_id)))
        if predecessors:
            d_parent = sum(dim_info[p][1] for p in predecessors)
            if model_config.type == "PCEScaleShiftModel":
                init_kwargs["d_parent"] = d_parent
            elif model_config.type == "MLPEnhancementModel":
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
    return MFNetJax(graph)


def _train_network(
    mfnet: MFNetJax,
    training_data: dict[str, Any],
    training_params: TrainingParams,
    config: Config,
) -> MFNetJax:
    """Run the training loop."""
    console.print(
        f"\nStarting training with {training_params.num_steps} steps..."
    )
    start_time = time.time()

    dataset_config = next(
        (d for d in config.datasets if d.type == "training"), None
    )
    if not dataset_config:
        console.print("[bold red]Critical error: No training dataset found.[/]")
        raise typer.Exit(code=1)

    data_file = training_data[dataset_config.name]
    target_nodes = tuple(sorted(dataset_config.nodes))

    try:
        x_train = data_file[f"x_train_{target_nodes[0]}"]
        y_train_tuple = tuple(data_file[f"y_train_{n}"] for n in target_nodes)
    except KeyError as e:
        console.print(f"[bold red]Data key not found in NPZ file: {e}[/]")
        raise typer.Exit(code=1)

    console.print(f"Training on nodes: [bold cyan]{target_nodes}[/]")
    console.print(f"Input data shape: {x_train.shape}")
    for i, y_arr in enumerate(y_train_tuple):
        console.print(
            f"  - Target data shape for node {target_nodes[i]}: {y_arr.shape}"
        )

    params, treedef = tree_util.tree_flatten(mfnet)
    optimizer = optax.adam(learning_rate=training_params.learning_rate)
    opt_state = optimizer.init(params)

    def loss_fn(
        p: list[Any], x: jnp.ndarray, y: tuple[jnp.ndarray, ...]
    ) -> jnp.ndarray:
        model = treedef.unflatten(p)
        return mse_loss_graph(model, nodes=target_nodes, x=x, y=y)

    @jax.jit
    def step(
        p: list[Any], opt_s: Any, x: jnp.ndarray, y: tuple[jnp.ndarray, ...]
    ) -> tuple[list[Any], Any, jnp.ndarray]:
        loss_val, grads = jax.value_and_grad(loss_fn)(p, x, y)
        updates, opt_s = optimizer.update(grads, opt_s)
        p = optax.apply_updates(p, updates)
        return p, opt_s, loss_val

    initial_loss = loss_fn(params, x_train, y_train_tuple)
    console.print(f"Initial Loss: [bold yellow]{initial_loss:.6f}[/]")

    for _ in track(range(training_params.num_steps), description="Training..."):
        params, opt_state, _ = step(params, opt_state, x_train, y_train_tuple)

    final_loss = loss_fn(params, x_train, y_train_tuple)
    duration = time.time() - start_time
    console.print(f"Final Loss:   [bold green]{final_loss:.6f}[/]")
    console.print(f"Training completed in {duration:.2f} seconds.")

    return treedef.unflatten(params)


def _process_prediction_tasks(mfnet: MFNetJax, config: Config) -> None:
    """Run and save all prediction tasks specified in the config."""
    console.print("\nProcessing prediction tasks...")
    prediction_datasets = [d for d in config.datasets if d.type == "prediction"]

    if not prediction_datasets:
        console.print("[yellow]No prediction tasks found in configuration.[/yellow]")
        return

    for task in prediction_datasets:
        console.print(f"--- Running prediction task: [bold cyan]{task.name}[/bold cyan] ---")
        try:
            input_data = np.load(task.data_path)
            x_predict = input_data["x_predict"]
            console.print(
                f"Loaded prediction inputs from '{task.data_path}' "
                f"with shape {x_predict.shape}"
            )
        except Exception as e:
            console.print(f"[bold red]Error loading data for task '{task.name}': {e}[/]")
            continue

        if len(task.nodes) != 1:
            console.print(
                f"[bold red]Prediction tasks must specify exactly one target "
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
            np.savez(output_path, y_predict=y_predict)
            console.print(f"[green]Successfully saved predictions to '{output_path}'[/green]")
        else:
            console.print(
                f"[yellow]No output_path for task '{task.name}'. Skipping save.[/yellow]"
            )


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
    config = _load_and_validate_config(config_path)
    training_data, dim_info = _load_training_data(config)
    mfnet = _build_mfnet_from_config(config, dim_info)
    trained_mfnet = _train_network(mfnet, training_data, config.training, config)
    _process_prediction_tasks(trained_mfnet, config)
    console.print("\n[bold green]CLI tool finished successfully.[/]")


if __name__ == "__main__":
    app()

