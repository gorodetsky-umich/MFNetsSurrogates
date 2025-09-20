# Tutorial: Building Your First MFNet

This tutorial walks through a complete example of defining, training, and evaluating a simple two-fidelity hierarchical model (`1 -> 2`).

This is the same example found in the project's `README.md`.

```python
import jax
import jax.numpy as jnp
import networkx as nx
import optax
from jax import tree_util

from mfnets_surrogates import (
    MFNetJax,
    LinearModel,
    LinearScaleShiftModel,
    init_linear_params,
    init_linear_scale_shift_model,
    mse_loss_graph,
)

def main():
    """A complete example of building, training, and running an MFNet."""
    key = jax.random.PRNGKey(0)
    d_in, d1_out, d2_out = 2, 2, 3
    key, true_key, train_key, data_key = jax.random.split(key, 4)

    # 1. Define a "true" model and generate some training data
    true_m1 = LinearModel(init_linear_params(true_key, d_in, d1_out))
    true_m2 = init_linear_scale_shift_model(true_key, d_in, d1_out, d2_out)
    true_graph = nx.DiGraph([(1, 2)])
    true_graph.add_node(1, func=true_m1)
    true_graph.add_node(2, func=true_m2)
    true_mfnet = MFNetJax(true_graph)

    x_train = jax.random.normal(data_key, (100, d_in))
    y_train = true_mfnet.run((1, 2), x_train)

    # 2. Create the MFNetJax model to be trained
    train_m1 = LinearModel(init_linear_params(train_key, d_in, d1_out))
    train_m2 = init_linear_scale_shift_model(train_key, d_in, d1_out, d2_out)
    train_graph_struct = nx.DiGraph([(1, 2)])
    train_graph_struct.add_node(1, func=train_m1)
    train_graph_struct.add_node(2, func=train_m2)
    mfnet_to_train = MFNetJax(train_graph_struct)

    # 3. Set up the training loop with Optax
    params, treedef = tree_util.tree_flatten(mfnet_to_train)
    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(params)

    def loss_fn(p, x, y):
        model = treedef.unflatten(p)
        return mse_loss_graph(model, nodes=(1, 2), x=x, y=y)

    @jax.jit
    def step(p, opt_s, x, y):
        loss_val, grads = jax.value_and_grad(loss_fn)(p, x, y)
        updates, opt_s = optimizer.update(grads, opt_s)
        p = optax.apply_updates(p, updates)
        return p, opt_s, loss_val

    print("--- Starting Training ---")
    initial_loss = loss_fn(params, x_train, y_train)
    print(f"Initial Loss: {initial_loss:.4f}")

    for i in range(2000):
        params, opt_state, loss = step(params, opt_state, x_train, y_train)
        if (i + 1) % 500 == 0:
            print(f"  Step {i+1}, Loss: {loss:.4f}")

    # 4. Make predictions with the final trained model
    mfnet_fitted = treedef.unflatten(params)
    x_test = jax.random.normal(data_key, (10, d_in))
    predictions = mfnet_fitted.run((1, 2), x_test)

    print("\n--- Predictions from Node 2 (Highest Fidelity) ---")
    print(predictions[1])

if __name__ == "__main__":
    main()
```
