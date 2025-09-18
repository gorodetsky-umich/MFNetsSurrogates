"""
An example script demonstrating how to build and train an MFNetJax model.

This script follows JAX best practices for random number generation by
explicitly managing and splitting a PRNGKey.

To run this example from the project root:
  1. Install the project in editable mode: `pip install -e ".[dev]"`
  2. Run the script: `python examples/basic_training.py`
"""
import jax
import jax.numpy as jnp
import jaxopt

# Import the components from your library
from mfnets_surrogates.net_jax import (
    MFNetJax,
    LinearModel,
    init_linear_params,
    init_linear_scale_shift_model,
    make_graph_2gen,
    resid_loss_graph,
    mse_loss_graph
)


def run_example():
    """
    Defines, trains, and evaluates a simple two-fidelity network.
    """
    # --- 1. Setup and Configuration ---
    key = jax.random.PRNGKey(42)
    dim_in = 10
    d1_out = 3  # Output dimension of the low-fidelity model
    d2_out = 4  # Output dimension of the high-fidelity model
    n_samples = 1000

    # --- 2. Generate Ground Truth Data ---
    # Create a "true" network with known parameters to generate our dataset.
    key, true_key, train_key, data_key = jax.random.split(key, 4)

    # True low-fidelity model (Node 1)
    true_model1 = LinearModel(init_linear_params(true_key, dim_in, d1_out))

    # True high-fidelity model (Node 2)
    true_model2 = init_linear_scale_shift_model(
        key=true_key, d_in=dim_in, d_parent=d1_out, d_out=d2_out
    )

    # Build the true graph and generate training data
    true_graph = make_graph_2gen(true_model1, true_model2)
    true_mfnet = MFNetJax(true_graph)
    
    x_train = jax.random.normal(data_key, (n_samples, dim_in))
    y_train_node1, y_train_node2 = true_mfnet.run(target_nodes=(1, 2), xinput=x_train)
    y_train = (y_train_node1, y_train_node2)

    print(f"Generated {n_samples} samples of training data.")

    # --- 3. Initialize a Trainable Model ---
    # Create a new, randomly initialized network that we will train.
    
    # Trainable low-fidelity model (Node 1)
    train_model1 = LinearModel(init_linear_params(train_key, dim_in, d1_out))

    # Trainable high-fidelity model (Node 2)
    train_model2 = init_linear_scale_shift_model(
        key=train_key, d_in=dim_in, d_parent=d1_out, d_out=d2_out
    )

    # Build the trainable graph
    train_graph = make_graph_2gen(train_model1, train_model2)
    mfnet_to_train = MFNetJax(train_graph)

    # --- 4. Train the Model ---
    print("\nStarting model training...")
    initial_mse = mse_loss_graph(mfnet_to_train, (1, 2), x_train, y_train)
    print(f"Initial MSE: {initial_mse:.6f}")

    # Use a least-squares solver like Gauss-Newton, which is often efficient
    # for problems of this nature. It uses the residual function.
    solver = jaxopt.GaussNewton(
        residual_fun=lambda m, x, y: resid_loss_graph(m, (1, 2), x, y),
        maxiter=50,
        tol=1e-6,
    )

    res = solver.run(init_params=mfnet_to_train, x=x_train, y=y_train)

    # The optimal parameters are in the `res.params` attribute
    mfnet_fitted = res.params

    # --- 5. Evaluate the Fitted Model ---
    final_mse = mse_loss_graph(mfnet_fitted, (1, 2), x_train, y_train)
    print(f"Final MSE:   {final_mse:.6f}")
    print("\nTraining complete.")


if __name__ == "__main__":
    run_example()
