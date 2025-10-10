"""A simple example demonstrating direct usage of the MFNetJax.fit() method."""

import jax
import jax.numpy as jnp
import networkx as nx

from mfnets_surrogates.net_jax import (
    MFNetJax,
    init_mlp_enhancement_model,
    init_mlp_model,
)


def main():
    """Run a simple MFNetJax training and prediction example."""
    key = jax.random.PRNGKey(42)
    d_in, d_out = 1, 1

    # 1. Define a "true" model and generate training data
    # True model is a simple hierarchical structure: 0 -> 1
    # Node 0 is a simple sine wave
    # Node 1 is an enhancement of Node 0's output
    x_train = jnp.linspace(-jnp.pi, jnp.pi, 100).reshape(-1, 1)
    y_train_0 = jnp.sin(x_train)
    y_train_1 = y_train_0 + 0.1 * jnp.cos(4 * x_train)
    train_data = [(x_train, y_train_0), (x_train, y_train_1)]

    # 2. Define the graph structure and initialize models to be trained
    key, m0_key, m1_key = jax.random.split(key, 3)
    graph = nx.DiGraph()
    graph.add_node(
        0, func=init_mlp_model(m0_key, layer_sizes=[d_in, 16, 16, d_out])
    )
    graph.add_node(
        1,
        func=init_mlp_enhancement_model(
            m1_key, layer_sizes=[d_in + d_out, 16, 16, d_out]
        ),
    )
    graph.add_edge(0, 1)

    # 3. Create the MFNetJax object
    mfnet = MFNetJax(graph)

    # 4. Train the model using the high-level .fit() method
    print("--- Starting Training ---")
    mfnet.fit(
        train_data=train_data,
        n_iters=5000,
        learning_rate=1e-3,
    )
    print("--- Training Complete ---")

    # 5. Make predictions with the final trained model
    x_test = jnp.linspace(-jnp.pi, jnp.pi, 10).reshape(-1, 1)
    # The run method returns a tuple of predictions, one for each target node
    (y_pred_0,) = mfnet.run((0,), x_test)
    (y_pred_1,) = mfnet.run((1,), x_test)

    print("\n--- Predictions from Node 0 (Low Fidelity) ---")
    print(y_pred_0.flatten())

    print("\n--- Predictions from Node 1 (High Fidelity) ---")
    print(y_pred_1.flatten())


if __name__ == "__main__":
    main()
