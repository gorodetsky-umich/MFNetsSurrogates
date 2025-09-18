"""Tests for the MFNetJax library.

Unit and integration tests for the MFNetJax framework.

To run these tests from the project root directory:
pip install -e .
pytest
"""

import jax
import jax.numpy as jnp
import networkx as nx
import optax
import pytest
from jax import tree_util
from jax.tree_util import register_pytree_node_class

# Import everything directly from your package's public API
from mfnets_surrogates import (
    LinearModel,
    LinearParams,
    MFNetJax,
    MLPModel,
    Model,
    init_linear_params,
    init_linear_scale_shift_model,
    init_mlp_params,
    make_graph_2gen,
    mse_loss_graph,
)


# A simple model just for testing fan-in logic. This can stay in the test file
# as it's not part of the public library.
@register_pytree_node_class
class ParentInputConcatenationModel(Model):
    """
    A test model that concatenates parent values and the primary input.

    It then applies a single linear transformation. Its purpose is to directly
    test the fan-in mechanism.
    """

    def __init__(self, linear_model: LinearModel):
        """Initialize the model with its internal linear model."""
        self.linear_model = linear_model

    def tree_flatten(self):
        """Flatten the model's parameters into a list of arrays (leaves)."""
        return (self.linear_model,), {}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Unflatten parameter arrays back into a model instance."""
        return cls(children[0])

    def run(self, xin: jnp.ndarray, parent_val: jnp.ndarray) -> jnp.ndarray:
        """Run the model by combining inputs and applying a linear layer."""
        # Concatenate all inputs along the feature axis
        combined_input = jnp.concatenate([xin, parent_val], axis=-1)
        return self.linear_model.run(combined_input)


@pytest.fixture
def key():
    """Provide a reusable, reproducible JAX random key for tests."""
    return jax.random.PRNGKey(0)


def test_linear_model_output():
    """Unit Test: Verify a LinearModel computes the correct mathematical result."""
    # Define known, non-random weights and bias
    weight = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    bias = jnp.array([10.0, 20.0])
    params = LinearParams(weight, bias)
    model = LinearModel(params)

    # Define a single input vector
    xin = jnp.array([1.0, 1.0, 2.0])

    # Manually calculate the expected output
    # (1*1 + 2*1 + 3*2) + 10 = 9 + 10 = 19
    # (4*1 + 5*1 + 6*2) + 20 = 21 + 20 = 41
    expected_output = jnp.array([19.0, 41.0])

    # JAX vmaps the run method, so we need to add a batch dimension to xin
    actual_output = model.run(xin[None, :])[0]

    assert jnp.allclose(actual_output, expected_output)


def test_mfnet_pytree_roundtrip(key):
    """Unit Test: Ensure MFNetJax objects can be flattened and unflattened."""
    d_in, d1_out, d2_out = 2, 2, 3
    key1, key2 = jax.random.split(key)

    # 1. Create an initial MFNetJax instance
    model1 = LinearModel(init_linear_params(key1, d_in, d1_out))
    model2 = init_linear_scale_shift_model(key2, d_in, d1_out, d2_out)
    graph = make_graph_2gen(model1, model2)
    mfnet_original = MFNetJax(graph)

    # 2. Flatten and then unflatten the object
    leaves, treedef = tree_util.tree_flatten(mfnet_original)
    mfnet_rebuilt = tree_util.tree_unflatten(treedef, leaves)

    # 3. Verify that the rebuilt object behaves identically
    x_test = jax.random.normal(key, (1, d_in))

    # Check that the outputs are the same
    original_output = mfnet_original.run((1, 2), x_test)
    rebuilt_output = mfnet_rebuilt.run((1, 2), x_test)

    assert len(original_output) == len(rebuilt_output)
    for orig, reb in zip(original_output, rebuilt_output, strict=False):
        assert jnp.allclose(orig, reb)


def test_graph_concatenates_fan_in_inputs():
    """MFNetJax.run correctly concats outputs from mult. parents."""
    # --- 1. Setup Graph: (1 -> 3) and (2 -> 3) ---
    graph = nx.DiGraph()
    # Node 1: input dim 2, output dim 2
    graph.add_node(
        1, func=LinearModel(LinearParams(jnp.ones((2, 2)), jnp.array([1.0, 1.0])))
    )
    # Node 2: input dim 2, output dim 3
    graph.add_node(
        2,
        func=LinearModel(LinearParams(jnp.ones((3, 2)), jnp.array([2.0, 2.0, 2.0]))),
    )

    # --- 2. Setup Child Node (Node 3) ---
    # Total input dimension for its linear model is (parent1 + parent2 + xin)
    # which is (2 + 3 + 2 = 7).
    child_linear_model = LinearModel(
        LinearParams(weight=jnp.ones((1, 7)), bias=jnp.array([100.0]))
    )
    graph.add_node(3, func=ParentInputConcatenationModel(child_linear_model))
    graph.add_edge(1, 3)
    graph.add_edge(2, 3)
    mfnet = MFNetJax(graph)

    # --- 3. Define Input and Calculate Expected Outputs ---
    x_test = jnp.ones((1, 2))

    # Manually calculate the final expected output of Node 3
    # Parent 1 output: [3, 3]
    # Parent 2 output: [4, 4, 4]
    # Combined input to child's linear layer: [1, 1, 3, 3, 4, 4, 4]
    # Sum of combined input = 20
    # Child output: 20 * 1 + 100 = 120
    expected_out3 = jnp.array([[120.0]])

    # --- 4. Run and Assert ---
    (out3,) = mfnet.run((3,), x_test)

    assert jnp.allclose(out3, expected_out3)


def test_end_to_end_training_overfit_with_optax(key):
    """Integration Test: Check if the graph can overfit a small dataset using Optax."""
    d_in, d1_out, d2_out = 2, 2, 3
    true_key, train_key, data_key = jax.random.split(key, 3)

    # 1. Generate a small, noiseless dataset from a "true" model
    true_model1 = LinearModel(init_linear_params(true_key, d_in, d1_out))
    true_model2 = init_linear_scale_shift_model(true_key, d_in, d1_out, d2_out)
    true_mfnet = MFNetJax(make_graph_2gen(true_model1, true_model2))
    x_train = jax.random.normal(data_key, (10, d_in))
    y_train = true_mfnet.run((1, 2), x_train)

    # 2. Create a randomly initialized model to be trained
    train_model1 = LinearModel(init_linear_params(train_key, d_in, d1_out))
    train_model2 = init_linear_scale_shift_model(train_key, d_in, d1_out, d2_out)
    mfnet_to_train = MFNetJax(make_graph_2gen(train_model1, train_model2))

    # 3. Flatten the model into its parameters (leaves) and structure (treedef).
    # Optax optimizers operate on the raw parameters (the leaves).
    params, treedef = tree_util.tree_flatten(mfnet_to_train)
    initial_params = params

    # 4. Define the optimizer
    optimizer = optax.adam(learning_rate=5e-3)
    opt_state = optimizer.init(initial_params)

    # 5. Define a loss function that takes the raw parameters (leaves) as its
    # first argument. This is required for `jax.grad`.
    def loss_fn(current_params, x, y):
        # Reconstruct the full MFNetJax model from the current parameters
        model = treedef.unflatten(current_params)
        # The 'nodes' argument is now hard-coded, solving the static arg issue
        return mse_loss_graph(model, nodes=(1, 2), x=x, y=y)

    initial_loss = loss_fn(initial_params, x_train, y_train)

    # 6. Define a JIT-compiled training step for performance
    @jax.jit
    def step(params, opt_state, x, y):
        grads = jax.grad(loss_fn)(params, x, y)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    # 7. Run the explicit training loop
    for _ in range(8000):  # Adam often requires more steps than LBFGS
        params, opt_state = step(params, opt_state, x_train, y_train)

    # 8. Reconstruct the final fitted model and calculate final loss
    mfnet_fitted = treedef.unflatten(params)
    final_loss = mse_loss_graph(mfnet_fitted, (1, 2), x_train, y_train)

    # 9. Assert that the final loss is significantly smaller than the initial
    assert final_loss < initial_loss / 100
    assert final_loss < 1e-5


def test_mlp_model_output_shape(key):
    """Unit Test: Verify the MLP model produces the correct output shape."""
    layer_sizes = [10, 32, 5]  # 10-dim in, 32-dim hidden, 5-dim out
    params = init_mlp_params(key, layer_sizes)
    model = MLPModel(params)

    x_test = jax.random.normal(key, (100, 10))  # 100 samples
    y_pred = model.run(x_test)

    assert y_pred.shape == (100, 5)
