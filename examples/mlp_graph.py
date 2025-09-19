"""
A comprehensive example that trains and visually compares different graph
structures for a 4-fidelity non-linear problem.

This script performs the following steps:
1. Defines a "true" data-generating process with a specific graph structure.
2. Generates training and testing data from this true process.
3. Visualizes the true data-generating graph.
4. Defines three different candidate model architectures:
   - A Peer model (all low fidelities are independent inputs to the highest).
   - A Hierarchical/Recursive model (a simple serial chain).
   - The Exact model (matching the true graph structure).
5. For each candidate model:
   - Trains the entire graph using Optax.
   - Generates predictions on unseen test data.
6. Creates a single, combined 2x3 plot showing the graph and prediction
   performance for each of the three candidate models.
"""
import os
from functools import partial

import jax
import jax.numpy as jnp
import networkx as nx
import optax
from jax import tree_util
from matplotlib import pyplot as plt

from mfnets_surrogates import (
    MFNetJax,
    MLPModel,
    MLPEnhancementModel,
    init_mlp_params,
    init_mlp_enhancement_model,
    mse_loss_graph,
)

# --- Plotting and Graph Helpers ---


def plot_graph(graph: nx.DiGraph, title: str, filename: str):
    """Visualize and save a single NetworkX graph to a file."""
    plt.figure(figsize=(6, 4), dpi=120)
    pos = nx.spring_layout(graph, seed=42)
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_size=2000,
        node_color="#a2d2ff",
        font_size=14,
        font_weight="bold",
        arrowsize=20,
    )
    plt.title(title, fontsize=16)
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
    print(f"  Saved graph visualization to '{filename}'")


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
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "wheat", "alpha": 0.5},
    )
    ax.set_xlabel("True High-Fidelity Values", fontsize=12)
    ax.set_ylabel("Predicted High-Fidelity Values", fontsize=12)
    ax.set_title(title, fontsize=16)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    ax.set_aspect("equal", adjustable="box")


def train_graph(mfnet: MFNetJax, x_train, y_train, num_steps=5000):
    """A helper function to run the Optax training loop for a given graph."""
    target_nodes = tuple(sorted(mfnet.graph.nodes))
    params, treedef = tree_util.tree_flatten(mfnet)
    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(params)

    loss_fn = partial(_calculate_loss, treedef=treedef, target_nodes=target_nodes)

    @jax.jit
    def step(p, opt_s, x, y):
        loss_val, grads = jax.value_and_grad(loss_fn)(p, x, y)
        updates, opt_s = optimizer.update(grads, opt_s)
        p = optax.apply_updates(p, updates)
        return p, opt_s, loss_val

    print(f"  Initial MSE Loss: {loss_fn(params, x_train, y_train):.6f}")
    for i in range(num_steps):
        params, opt_state, loss = step(params, opt_state, x_train, y_train)
        if (i + 1) % 1000 == 0:
            print(f"    Step {i+1}, Loss: {loss:.6f}")

    mfnet_fitted = treedef.unflatten(params)
    final_loss = mse_loss_graph(mfnet_fitted, target_nodes, x_train, y_train)
    print(f"  Final MSE Loss:   {final_loss:.6f}")
    return mfnet_fitted


def _calculate_loss(current_params, x, y, treedef, target_nodes):
    """Helper for loss calculation, compatible with jax.grad."""
    model = treedef.unflatten(current_params)
    return mse_loss_graph(model, target_nodes, x, y)


# --- Main Experiment ---


def main():
    """Run the graph comparison experiment."""
    key = jax.random.PRNGKey(42)
    d_in, d_out = 1, 1
    os.makedirs("plots", exist_ok=True)

    # 1. Define True Data-Generating Process
    print("--- 1. Defining True Data-Generating Process ---")
    true_graph_structure = nx.DiGraph([(1, 2), (2, 4), (3, 4)])
    # plot_graph(
    #     true_graph_structure, "True Data-Generating Graph", "plots/true_graph.png"
    # )

    # Generate training and testing data splits
    x_all = jnp.linspace(-jnp.pi, jnp.pi, 400).reshape(-1, d_in)
    y1_all = 0.5 * jnp.cos(0.8 * x_all) - 0.2
    y2_all = y1_all**2 + 0.1 * jnp.sin(x_all)
    y3_all = 0.8 * jnp.sin(x_all) + 0.1
    y4_all = jnp.sin(y2_all * 2) + y3_all + 0.05 * x_all

    key, subkey = jax.random.split(key)
    train_indices = jax.random.permutation(subkey, 400)[:200]
    test_indices = jnp.setdiff1d(jnp.arange(400), train_indices)

    x_train, x_test = x_all[train_indices], x_all[test_indices]
    y_train = (
        y1_all[train_indices],
        y2_all[train_indices],
        y3_all[train_indices],
        y4_all[train_indices],
    )
    y_test = (
        y1_all[test_indices],
        y2_all[test_indices],
        y3_all[test_indices],
        y4_all[test_indices],
    )

    # --- Setup for the combined 2x3 plot ---
    fig, axes = plt.subplots(2, 3, figsize=(22, 12), dpi=120)
    fig.suptitle("Comparison of Multi-Fidelity Graph Architectures", fontsize=24)
    ax_map = {
        "Peer": axes[:, 0],
        "Hierarchical": axes[:, 1],
        "Exact": axes[:, 2],
    }

    # 2. Define Model Architectures and Train
    activation = jax.nn.tanh
    key, p_key, h_key, e_key = jax.random.split(key, 4)

    # --- Candidate 1: Peer Model (1,2,3 -> 4) ---
    print("\n--- 2a. Training Peer Model ---")
    peer_graph_structure = nx.DiGraph([(1, 4), (2, 4), (3, 4)])
    plot_graph_on_ax(ax_map["Peer"][0], peer_graph_structure, "Peer Model Graph")
    m1_p = MLPModel(init_mlp_params(p_key, [d_in, 16, 16, d_out]), activation)
    m2_p = MLPModel(init_mlp_params(p_key, [d_in, 16, 16, d_out]), activation)
    m3_p = MLPModel(init_mlp_params(p_key, [d_in, 16, 16, d_out]), activation)
    m4_p = init_mlp_enhancement_model(
        p_key, [d_in + 3 * d_out, 32, 32, d_out], activation
    )
    peer_mfnet_graph = nx.DiGraph()
    peer_mfnet_graph.add_nodes_from([
        (1, {"func": m1_p}), (2, {"func": m2_p}),
        (3, {"func": m3_p}), (4, {"func": m4_p})
    ])
    peer_mfnet_graph.add_edges_from([(1, 4), (2, 4), (3, 4)])
    peer_mfnet = MFNetJax(peer_mfnet_graph)
    peer_mfnet_trained = train_graph(peer_mfnet, x_train, y_train)

    # --- Candidate 2: Hierarchical Model (1 -> 2 -> 3 -> 4) ---
    print("\n--- 2b. Training Hierarchical Model ---")
    hier_graph_structure = nx.DiGraph([(1, 2), (2, 3), (3, 4)])
    plot_graph_on_ax(ax_map["Hierarchical"][0], hier_graph_structure, "Hierarchical Model Graph")
    m1_h = MLPModel(init_mlp_params(h_key, [d_in, 16, 16, d_out]), activation)
    m2_h = init_mlp_enhancement_model(h_key, [d_in + d_out, 16, 16, d_out], activation)
    m3_h = init_mlp_enhancement_model(h_key, [d_in + d_out, 16, 16, d_out], activation)
    m4_h = init_mlp_enhancement_model(h_key, [d_in + d_out, 32, 32, d_out], activation)
    hier_mfnet_graph = nx.DiGraph()
    hier_mfnet_graph.add_nodes_from([
        (1, {"func": m1_h}), (2, {"func": m2_h}),
        (3, {"func": m3_h}), (4, {"func": m4_h})
    ])
    hier_mfnet_graph.add_edges_from([(1, 2), (2, 3), (3, 4)])
    hier_mfnet = MFNetJax(hier_mfnet_graph)
    hier_mfnet_trained = train_graph(hier_mfnet, x_train, y_train)

    # --- Candidate 3: Exact Model (1 -> 2 -> 4, 3 -> 4) ---
    print("\n--- 2c. Training Exact Model ---")
    plot_graph_on_ax(ax_map["Exact"][0], true_graph_structure, "Exact Model Graph")
    m1_e = MLPModel(init_mlp_params(e_key, [d_in, 16, 16, d_out]), activation)
    m2_e = init_mlp_enhancement_model(e_key, [d_in + d_out, 16, 16, d_out], activation)
    m3_e = MLPModel(init_mlp_params(e_key, [d_in, 16, 16, d_out]), activation)
    m4_e = init_mlp_enhancement_model(e_key, [d_in + 2 * d_out, 32, 32, d_out], activation)
    exact_mfnet_graph = nx.DiGraph()
    exact_mfnet_graph.add_nodes_from([
        (1, {"func": m1_e}), (2, {"func": m2_e}),
        (3, {"func": m3_e}), (4, {"func": m4_e})
    ])
    exact_mfnet_graph.add_edges_from([(1, 2), (2, 4), (3, 4)])
    exact_mfnet = MFNetJax(exact_mfnet_graph)
    exact_mfnet_trained = train_graph(exact_mfnet, x_train, y_train)

    # 3. Evaluate and Plot Predictions on the subplots
    print("\n--- 3. Evaluating Models on Test Data ---")
    y_true_hf = y_test[3]
    # Peer Model Predictions
    y_pred_peer = peer_mfnet_trained.run((4,), x_test)[0]
    mse_peer = jnp.mean((y_true_hf - y_pred_peer) ** 2)
    print(f"  Peer Model Test MSE: {mse_peer:1.6E}")
    plot_predictions_on_ax(ax_map["Peer"][1], y_true_hf, y_pred_peer, mse_peer, "Peer Model Predictions")

    # Hierarchical Model Predictions
    y_pred_hier = hier_mfnet_trained.run((4,), x_test)[0]
    mse_hier = jnp.mean((y_true_hf - y_pred_hier) ** 2)
    print(f"  Hierarchical Model Test MSE: {mse_hier:1.2E}")
    plot_predictions_on_ax(ax_map["Hierarchical"][1], y_true_hf, y_pred_hier, mse_hier, "Hierarchical Model Predictions")

    # Exact Model Predictions
    y_pred_exact = exact_mfnet_trained.run((4,), x_test)[0]
    mse_exact = jnp.mean((y_true_hf - y_pred_exact) ** 2)
    print(f"  Exact Model Test MSE: {mse_exact:1.2E}")
    plot_predictions_on_ax(ax_map["Exact"][1], y_true_hf, y_pred_exact, mse_exact, "Exact Model Predictions")

    # --- Finalize and Save the Combined Plot ---
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])  # Adjust for suptitle
    fig.savefig("plots/model_comparison.png")
    plt.close(fig)

    print("\nExperiment complete. Check the 'plots/' directory for results.")


if __name__ == "__main__":
    main()

