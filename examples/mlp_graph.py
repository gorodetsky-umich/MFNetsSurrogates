"""A comprehensive example that trains and visually compares graph structures.

This script performs the following steps:
1. Defines a "true" data-generating process with a specific graph structure.
2. Generates training and testing data from this true process.
3. Defines three different candidate model architectures:
   - A Peer model (all low fidelities are independent inputs to the highest).
   - A Hierarchical/Recursive model (a simple serial chain).
   - The Exact model (matching the true graph structure).
4. For each candidate model:
   - Trains the entire graph using Optax.
   - Generates predictions on unseen test data.
5. Creates a single, combined 2x3 plot showing the graph and prediction
   performance for each of the three candidate models.
"""
import os
from typing import Callable, Dict

import jax
import jax.numpy as jnp
import networkx as nx
import optax
from jax import tree_util
from matplotlib import pyplot as plt

from mfnets_surrogates import (
    MFNetJax,
    MLPModel,
    init_mlp_enhancement_model,
    init_mlp_params,
    mse_loss_graph,
)

# --- Plotting Helpers ---


def plot_graph_on_ax(ax, graph: nx.DiGraph, title: str):
    """Draw a NetworkX graph on a given Matplotlib Axes object."""
    pos = nx.spring_layout(graph, seed=42)
    nx.draw(
        graph,
        pos,
        ax=ax,
        with_labels=True,
        node_size=2000,
        node_color="#a2d2ff",
        font_size=14,
        font_weight="bold",
        arrowsize=20,
    )
    ax.set_title(title, fontsize=16)


def plot_predictions_on_ax(ax, y_true, y_pred, mse: float, title: str):
    """Draw a predicted vs. actual scatter plot on a given Axes object."""
    ax.scatter(y_true, y_pred, alpha=0.6, label="Model Predictions")
    ax.plot(
        [y_true.min(), y_true.max()],
        [y_true.min(), y_true.max()],
        "r--",
        lw=2,
        label="Perfect Fit",
    )
    ax.text(
        0.95,
        0.05,
        f"Test MSE: {mse:1.2E}",
        verticalalignment="bottom",
        horizontalalignment="right",
        transform=ax.transAxes,
        fontsize=12,
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "wheat",
            "alpha": 0.5,
        },
    )
    ax.set_xlabel("True High-Fidelity Values", fontsize=12)
    ax.set_ylabel("Predicted High-Fidelity Values", fontsize=12)
    ax.set_title(title, fontsize=16)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    ax.set_aspect("equal", adjustable="box")


# --- Training and Model Creation ---


def train_graph(mfnet: MFNetJax, x_train, y_train, num_steps=5000):
    """Run the Optax training loop for a given graph."""
    target_nodes = tuple(sorted(mfnet.graph.nodes))
    params, treedef = tree_util.tree_flatten(mfnet)
    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(params)

    def _calculate_loss(current_params, x, y):
        model = treedef.unflatten(current_params)
        return mse_loss_graph(model, target_nodes, x, y)

    @jax.jit
    def step(p, opt_s, x, y):
        loss_val, grads = jax.value_and_grad(_calculate_loss)(p, x, y)
        updates, opt_s = optimizer.update(grads, opt_s)
        p = optax.apply_updates(p, updates)
        return p, opt_s, loss_val

    initial_loss = _calculate_loss(params, x_train, y_train)
    print(f"  Initial MSE Loss: {initial_loss:.6f}")
    for i in range(num_steps):
        params, opt_state, loss = step(params, opt_state, x_train, y_train)
        if (i + 1) % 1000 == 0:
            print(f"    Step {i+1}, Loss: {loss:.6f}")

    mfnet_fitted = treedef.unflatten(params)
    final_loss = mse_loss_graph(
        mfnet_fitted, target_nodes, x_train, y_train
    )
    print(f"  Final MSE Loss:   {final_loss:.6f}")
    return mfnet_fitted


def create_mfnet_from_graph(
    graph_struct: nx.DiGraph,
    model_builders: Dict[int, Callable[[jax.Array], Callable]],
    key: jax.Array,
) -> MFNetJax:
    """Create an MFNetJax instance from a graph and model builder functions."""
    mfnet_graph = graph_struct.copy()
    for node_id, builder in model_builders.items():
        key, subkey = jax.random.split(key)
        mfnet_graph.add_node(node_id, func=builder(subkey))
    return MFNetJax(mfnet_graph)


def _build_peer_models(d_in, d_out, activation):
    """Return a dictionary of model builders for the Peer architecture."""
    return {
        1: lambda k: MLPModel(
            init_mlp_params(k, [d_in, 16, 16, d_out]), activation
        ),
        2: lambda k: MLPModel(
            init_mlp_params(k, [d_in, 16, 16, d_out]), activation
        ),
        3: lambda k: MLPModel(
            init_mlp_params(k, [d_in, 16, 16, d_out]), activation
        ),
        4: lambda k: init_mlp_enhancement_model(
            k, [d_in + 3 * d_out, 32, 32, d_out], activation
        ),
    }


def _build_hierarchical_models(d_in, d_out, activation):
    """Return a dictionary of model builders for the Hierarchical architecture."""
    return {
        1: lambda k: MLPModel(
            init_mlp_params(k, [d_in, 16, 16, d_out]), activation
        ),
        2: lambda k: init_mlp_enhancement_model(
            k, [d_in + d_out, 16, 16, d_out], activation
        ),
        3: lambda k: init_mlp_enhancement_model(
            k, [d_in + d_out, 16, 16, d_out], activation
        ),
        4: lambda k: init_mlp_enhancement_model(
            k, [d_in + d_out, 32, 32, d_out], activation
        ),
    }


def _build_exact_models(d_in, d_out, activation):
    """Return a dictionary of model builders for the Exact architecture."""
    return {
        1: lambda k: MLPModel(
            init_mlp_params(k, [d_in, 16, 16, d_out]), activation
        ),
        2: lambda k: init_mlp_enhancement_model(
            k, [d_in + d_out, 16, 16, d_out], activation
        ),
        3: lambda k: MLPModel(
            init_mlp_params(k, [d_in, 16, 16, d_out]), activation
        ),
        4: lambda k: init_mlp_enhancement_model(
            k, [d_in + 2 * d_out, 32, 32, d_out], activation
        ),
    }


# --- Main Experiment ---


def main():
    """Run the graph comparison experiment."""
    key = jax.random.PRNGKey(42)
    d_in, d_out = 1, 1
    os.makedirs("plots", exist_ok=True)

    # 1. Define True Data-Generating Process
    print("--- 1. Generating synthetic data ---")
    x_all = jnp.linspace(-jnp.pi, jnp.pi, 400).reshape(-1, d_in)
    y1_all = 0.5 * jnp.cos(0.8 * x_all) - 0.2
    y2_all = y1_all**2 + 0.1 * jnp.sin(x_all)
    y3_all = 0.8 * jnp.sin(x_all) + 0.1
    y4_all = jnp.sin(y2_all * 2) + y3_all + 0.05 * x_all
    y_all = (y1_all, y2_all, y3_all, y4_all)

    # Create data splits
    train_indices = jax.random.permutation(key, 400)[:200]
    test_indices = jnp.setdiff1d(jnp.arange(400), train_indices)
    x_train, x_test = x_all[train_indices], x_all[test_indices]
    y_train = tuple(y[train_indices] for y in y_all)
    y_test = tuple(y[test_indices] for y in y_all)
    y_true_hf = y_test[3]

    # Define the architectures to be tested
    architectures = {
        "Peer": {
            "structure": nx.DiGraph([(1, 4), (2, 4), (3, 4)]),
            "builder": _build_peer_models,
        },
        "Hierarchical": {
            "structure": nx.DiGraph([(1, 2), (2, 3), (3, 4)]),
            "builder": _build_hierarchical_models,
        },
        "Exact": {
            "structure": nx.DiGraph([(1, 2), (2, 4), (3, 4)]),
            "builder": _build_exact_models,
        },
    }

    # Setup the combined plot
    fig, axes = plt.subplots(2, 3, figsize=(22, 12), dpi=120)
    fig.suptitle(
        "Comparison of Multi-Fidelity Graph Architectures", fontsize=24
    )
    ax_map = {
        "Peer": axes[:, 0],
        "Hierarchical": axes[:, 1],
        "Exact": axes[:, 2],
    }

    # 2. Train and evaluate each architecture
    for name, config in architectures.items():
        print(f"\n--- 2. Training {name} Model ---")
        key, subkey = jax.random.split(key)

        model = create_mfnet_from_graph(
            config["structure"],
            config["builder"](d_in, d_out, jax.nn.tanh),
            subkey,
        )
        trained_model = train_graph(model, x_train, y_train)

        y_pred = trained_model.run((4,), x_test)[0]
        mse = jnp.mean((y_true_hf - y_pred) ** 2)
        print(f"  {name} Model Test MSE: {mse:1.6E}")

        plot_graph_on_ax(
            ax_map[name][0], config["structure"], f"{name} Model Graph"
        )
        plot_predictions_on_ax(
            ax_map[name][1],
            y_true_hf,
            y_pred,
            mse,
            f"{name} Model Predictions",
        )

    # Finalize and save the combined plot
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    save_path = "plots/model_comparison.png"
    fig.savefig(save_path)
    plt.close(fig)
    print(f"\nExperiment complete. Check '{save_path}' for results.")


if __name__ == "__main__":
    main()
