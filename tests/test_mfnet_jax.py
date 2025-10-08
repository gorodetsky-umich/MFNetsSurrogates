"""Tests for the MFNetJax library."""

import jax
import jax.numpy as jnp
import networkx as nx
import optax
import pytest
from jax import tree_util
from jax.tree_util import register_pytree_node_class

from mfnets_surrogates import (
    LinearModel,
    LinearParams,
    MFNetJax,
    MLPEnhancementModel,
    MLPModel,
    Model,
    build_poly_basis,
    init_linear_params,
    init_linear_scale_shift_model,
    init_mlp_enhancement_model,
    init_mlp_params,
    init_pc_additive_model,
    init_pce_model,
    init_pce_scale_shift_model,
    make_graph_2gen,
    mse_loss_graph,
)


# A simple model just for testing fan-in logic. This can stay in the test file
# as it's not part of the public library.
@register_pytree_node_class
class ParentInputConcatenationModel(Model):
    """A test model that concatenates parent and primary inputs."""

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
    """Verify a LinearModel computes the correct mathematical result."""
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
    expected_output = jnp.array([[19.0, 41.0]])

    actual_output = model.run(xin[None, :])

    assert jnp.allclose(actual_output, expected_output)


def test_mfnet_pytree_roundtrip(key):
    """Ensure MFNetJax objects can be flattened and unflattened."""
    d_in, d1_out, d2_out = 2, 2, 3
    key1, key2 = jax.random.split(key)

    # 1. Create an initial MFNetJax instance
    model1 = LinearModel(init_linear_params(key1, d_in, d1_out))
    model2 = init_linear_scale_shift_model(key2, d_in, d1_out, d2_out)
    graph = make_graph_2gen(model1, model2)
    mfnet_original = MFNetJax(graph)

    # 2. Flatten and then unflatten the object
    leaves, treedef = tree_util.tree_flatten(mfnet_original)
    mfnet_rebuilt = treedef.unflatten(leaves)

    # 3. Verify that the rebuilt object behaves identically
    x_test = jax.random.normal(key, (1, d_in))

    # Check that the outputs are the same
    original_output = mfnet_original.run((1, 2), x_test)
    rebuilt_output = mfnet_rebuilt.run((1, 2), x_test)

    assert len(original_output) == len(rebuilt_output)
    for orig, reb in zip(original_output, rebuilt_output, strict=False):
        assert jnp.allclose(orig, reb)


def test_graph_concatenates_peer_inputs():
    """Test MFNetJax.run correctly concatenates multiple parent outputs."""
    # --- 1. Setup Graph: (1 -> 3) and (2 -> 3) ---
    graph = nx.DiGraph()
    # Node 1: input dim 2, output dim 2
    graph.add_node(
        1,
        func=LinearModel(
            LinearParams(jnp.ones((2, 2)), jnp.array([1.0, 1.0]))
        ),
    )
    # Node 2: input dim 2, output dim 3
    graph.add_node(
        2,
        func=LinearModel(
            LinearParams(jnp.ones((3, 2)), jnp.array([2.0, 2.0, 2.0]))
        ),
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
    """Check if the graph can overfit a small dataset using Optax."""
    d_in, d1_out, d2_out = 2, 2, 3
    true_key, train_key, data_key = jax.random.split(key, 3)

    # 1. Generate a small, noiseless dataset from a "true" model
    true_model1 = LinearModel(init_linear_params(true_key, d_in, d1_out))
    true_model2 = init_linear_scale_shift_model(true_key, d_in, d1_out, d2_out)
    true_mfnet = MFNetJax(make_graph_2gen(true_model1, true_model2))
    x_train = jax.random.normal(data_key, (10, d_in))
    y_train = true_mfnet.run((1, 2), x_train)

    x_train_list = [x_train, x_train]

    # 2. Create a randomly initialized model to be trained
    train_model1 = LinearModel(init_linear_params(train_key, d_in, d1_out))
    train_model2 = init_linear_scale_shift_model(
        train_key, d_in, d1_out, d2_out
    )
    mfnet_to_train = MFNetJax(make_graph_2gen(train_model1, train_model2))

    # 3. Flatten the model into its parameters and structure.
    params, treedef = tree_util.tree_flatten(mfnet_to_train)
    initial_params = params

    # 4. Define the optimizer and training state
    optimizer = optax.adam(learning_rate=5e-3)
    opt_state = optimizer.init(initial_params)

    # 5. Define a loss function that takes the raw parameters
    def loss_fn(current_params, x_list, y):
        model = treedef.unflatten(current_params)
        return mse_loss_graph(model, nodes=(1, 2), x_list=x_list, y_list=y)

    initial_loss = loss_fn(initial_params, x_train_list, y_train)

    # 6. Define a JIT-compiled training step
    @jax.jit
    def step(p, opt_s, x, y):
        grads = jax.grad(loss_fn)(p, x, y)
        updates, opt_s = optimizer.update(grads, opt_s)
        p = optax.apply_updates(p, updates)
        return p, opt_s

    # 7. Run the explicit training loop
    for _ in range(5000):
        params, opt_state = step(params, opt_state, x_train_list, y_train)

    # 8. Reconstruct the final fitted model and calculate final loss
    mfnet_fitted = treedef.unflatten(params)
    final_loss = mse_loss_graph(mfnet_fitted, (1, 2), x_train_list, y_train)

    # 9. Assert that the final loss is significantly smaller than the initial
    assert final_loss < initial_loss / 100
    assert final_loss < 1e-4


def test_mlp_model_output_shape(key):
    """Verify the MLP model produces the correct output shape."""
    layer_sizes = [10, 32, 5]  # 10-dim in, 32-dim hidden, 5-dim out
    params = init_mlp_params(key, layer_sizes)
    model = MLPModel(params)

    x_test = jax.random.normal(key, (100, 10))  # 100 samples
    y_pred = model.run(x_test)

    assert y_pred.shape == (100, 5)


def test_mlp_enhancement_model_output_shape(key):
    """Verify the MLPEnhancementModel has the correct output shape."""
    d_in, d_parent, d_out = 5, 3, 7
    batch_size = 100

    layer_sizes = [d_in + d_parent, 32, d_out]
    model = init_mlp_enhancement_model(key, layer_sizes)

    x_test = jax.random.normal(key, (batch_size, d_in))
    parent_val = jax.random.normal(key, (batch_size, d_parent))
    y_pred = model.run(x_test, parent_val)

    assert y_pred.shape == (batch_size, d_out)


def test_mlp_enhancement_pytree_roundtrip(key):
    """Ensure MLPEnhancementModel can be flattened and unflattened."""
    layer_sizes = [5, 16, 2]
    original_model = init_mlp_enhancement_model(key, layer_sizes)

    leaves, treedef = tree_util.tree_flatten(original_model)
    rebuilt_model = treedef.unflatten(leaves)

    x_test = jax.random.normal(key, (1, 3))
    parent_val = jax.random.normal(key, (1, 2))
    original_output = original_model.run(x_test, parent_val)
    rebuilt_output = rebuilt_model.run(x_test, parent_val)

    assert jnp.allclose(original_output, rebuilt_output)


def test_mlp_enhancement_concatenation_logic(key):
    """Verify the MLPEnhancementModel correctly combines inputs."""
    d_in, d_parent, d_out = 2, 3, 1

    internal_mlp_params = [
        LinearParams(
            weight=jnp.ones((d_out, d_in + d_parent)),
            bias=jnp.zeros(d_out),
        )
    ]
    internal_mlp = MLPModel(internal_mlp_params)
    model = MLPEnhancementModel(internal_mlp)

    xin = jnp.array([[1.0, 2.0]])
    parent_val = jnp.array([[3.0, 4.0, 5.0]])
    expected_output = jnp.array([[15.0]])
    actual_output = model.run(xin, parent_val)

    assert jnp.allclose(actual_output, expected_output)


def test_pce_basis_hermite_correctness():
    """Verify Hermite basis matrix for a known simple case."""
    # For 1D, degree 2, the basis functions are: H0, H1, H2
    # H0n=1, H1n=x, H2n=(x^2-1)/sqrt(2)
    x = jnp.array([[2.0]])  # A single sample at x=2
    multi_indices = jnp.array([[0], [1], [2]])
    degree = 2

    basis = build_poly_basis(x, multi_indices, "hermite", degree)

    expected_basis = jnp.array(
        [
            [
                1.0,  # H0
                2.0,  # H1
                (2.0**2 - 1) / jnp.sqrt(2.0),  # H2
            ]
        ]
    )

    assert jnp.allclose(basis, expected_basis, atol=1e-6)


def test_pce_model_pytree_roundtrip(key):
    """Ensure PCEModel can be flattened and unflattened."""
    d_in, d_out, degree = 3, 2, 2
    original_model = init_pce_model(key, d_in, d_out, degree)

    leaves, treedef = tree_util.tree_flatten(original_model)
    rebuilt_model = treedef.unflatten(leaves)

    x_test = jax.random.normal(key, (1, d_in))

    original_output = original_model.run(x_test)
    rebuilt_output = rebuilt_model.run(x_test)

    assert jnp.allclose(original_output, rebuilt_output)


def test_pc_additive_model_pytree_roundtrip(key):
    """Ensure PCEAdditiveModel can be flattened and unflattened."""
    d_in, d_parent, d_out, degree = 3, 2, 4, 2
    original_model = init_pc_additive_model(key, d_in, d_parent, d_out, degree)

    leaves, treedef = tree_util.tree_flatten(original_model)
    rebuilt_model = treedef.unflatten(leaves)

    x_test = jax.random.normal(key, (1, d_in))
    parent_val = jax.random.normal(key, (1, d_parent))

    original_output = original_model.run(x_test, parent_val)
    rebuilt_output = rebuilt_model.run(x_test, parent_val)

    assert jnp.allclose(original_output, rebuilt_output)


def test_pce_scale_shift_model_pytree_roundtrip(key):
    """Ensure PCEScaleShiftModel can be flattened and unflattened."""
    d_in, d_parent, d_out, degree = 3, 2, 4, 2
    original_model = init_pce_scale_shift_model(
        key, d_in, d_parent, d_out, degree
    )

    leaves, treedef = tree_util.tree_flatten(original_model)
    rebuilt_model = treedef.unflatten(leaves)

    x_test = jax.random.normal(key, (1, d_in))
    parent_val = jax.random.normal(key, (1, d_parent))

    original_output = original_model.run(x_test, parent_val)
    rebuilt_output = rebuilt_model.run(x_test, parent_val)

    assert jnp.allclose(original_output, rebuilt_output)


def test_mfnet_fit_method_overfits(key):
    """Check if the new MFNetJax.fit() method can overfit a small dataset."""
    d_in, d1_out, d2_out = 2, 2, 3
    true_key, train_key, data_key = jax.random.split(key, 3)

    # 1. Generate a small, noiseless dataset from a "true" model
    true_model1 = LinearModel(init_linear_params(true_key, d_in, d1_out))
    true_model2 = init_linear_scale_shift_model(true_key, d_in, d1_out, d2_out)
    true_mfnet = MFNetJax(make_graph_2gen(true_model1, true_model2))
    x_train = jax.random.normal(data_key, (10, d_in))
    y_train_tuple = true_mfnet.run((1, 2), x_train)

    # The fit method expects a list of (x, y) tuples. For this graph,
    # both models are trained on the same input data.
    train_data = [
        (x_train, y_train_tuple[0]),
        (x_train, y_train_tuple[1]),
    ]

    # 2. Create a randomly initialized model to be trained
    train_model1 = LinearModel(init_linear_params(train_key, d_in, d1_out))
    train_model2 = init_linear_scale_shift_model(
        train_key, d_in, d1_out, d2_out
    )
    mfnet_to_train = MFNetJax(make_graph_2gen(train_model1, train_model2))

    # 3. Calculate initial loss before training
    x_train_list = [x_train, x_train]
    initial_loss = mse_loss_graph(
        mfnet_to_train, (1, 2), x_train_list, y_train_tuple
    )

    # 4. Call the new fit method to train the model
    mfnet_to_train.fit(
        train_data, n_iters=5000, learning_rate=5e-3, verbose=False
    )

    # 5. Calculate final loss after training
    final_loss = mse_loss_graph(
        mfnet_to_train, (1, 2), x_train_list, y_train_tuple
    )

    # 6. Assert that the final loss is significantly smaller
    assert final_loss < initial_loss / 100
    assert final_loss < 1e-4


def test_optimizable_flag_effect(key):
    """Test that only optimizable parameters are updated during training."""
    d_in, d_out = 2, 2
    train_key, data_key = jax.random.split(key)

    # Initialize a simple linear model
    params = init_linear_params(train_key, d_in, d_out)
    model = LinearModel(params)

    # Create a graph with a single node
    graph = nx.DiGraph()
    graph.add_node(1, func=model)
    mfnet = MFNetJax(graph)

    # Generate a small dataset
    x_train = jax.random.normal(data_key, (10, d_in))
    y_train = jax.random.normal(data_key, (10, d_out))
    train_data = [(x_train, y_train)]

    # Set the model as non-optimizable
    model.set_optimizable(False)

    # Capture initial parameters
    initial_params = jax.tree_util.tree_leaves(mfnet)

    # Train the model
    mfnet.fit(train_data, n_iters=100, learning_rate=1e-3, verbose=False)

    # Capture parameters after training
    final_params = jax.tree_util.tree_leaves(mfnet)

    # Assert that parameters have not changed
    for initial, final in zip(initial_params, final_params, strict=False):
        assert jnp.allclose(initial, final), (
            "Parameters should not change when non-optimizable"
        )

    # Set the model as optimizable
    model.set_optimizable(True)

    # Train the model again
    mfnet.fit(train_data, n_iters=100, learning_rate=1e-3, verbose=False)

    # Capture parameters after training
    updated_params = jax.tree_util.tree_leaves(mfnet)

    # Assert that parameters have changed
    for initial, updated in zip(initial_params, updated_params, strict=False):
        assert not jnp.allclose(initial, updated), (
            "Parameters should change when optimizable"
        )


def test_three_node_subset_optimizable(key):
    """Nodes 1 and 2 fixed, only node 3 should learn."""
    import networkx as nx
    from jax.tree_util import tree_leaves

    # Dimensions
    d_in, d1, d2, d3 = 2, 2, 3, 2

    # Random keys
    k1, k2, k3, kd = jax.random.split(key, 4)

    # Build models
    m1 = LinearModel(init_linear_params(k1, d_in, d1))

    # Node 2: MLP enhancement of m1
    mlp2 = MLPModel(init_mlp_params(k2, [d_in + d1, 8, d2]))
    m2 = MLPEnhancementModel(mlp2)

    # Node 3: MLP enhancement of m2
    mlp3 = MLPModel(init_mlp_params(k3, [d_in + d2, 8, d3]))
    m3 = MLPEnhancementModel(mlp3)

    # Make graph 1→2→3
    graph = nx.DiGraph()
    graph.add_node(1, func=m1)
    graph.add_node(2, func=m2)
    graph.add_node(3, func=m3)
    graph.add_edge(1, 2)
    graph.add_edge(2, 3)
    mfnet = MFNetJax(graph)

    # Synthetic training data: make y3 = current output of node 3
    x = jax.random.normal(kd, (50, d_in))
    (y3,) = mfnet.run((3,), x)

    # Train only node 3: others fixed
    m1.set_optimizable(False)
    m2.set_optimizable(False)
    m3.set_optimizable(True)

    # Prepare train_data: [None, None, (x,y3)]
    train_data = [None, None, (x, y3)]

    # Record initial leaves
    init_leaves = tree_leaves(mfnet)

    # Fit for a small number of steps
    mfnet.fit(train_data, n_iters=200, learning_rate=1e-2, verbose=False)

    # Record post-fit leaves
    post_leaves = tree_leaves(mfnet)

    # Count leaves per node
    n1 = len(tree_leaves(m1))
    n2 = len(tree_leaves(m2))

    # 1) Node 1 leaves unchanged
    for before, after in zip(init_leaves[:n1], post_leaves[:n1], strict=False):
        assert jnp.allclose(before, after), "Node1 params should remain fixed"

    # 2) Node 2 leaves unchanged
    for before, after in zip(
        init_leaves[n1 : n1 + n2], post_leaves[n1 : n1 + n2], strict=False
    ):
        assert jnp.allclose(before, after), "Node2 params should remain fixed"

    # 3) Node 3 leaves should change
    for before, after in zip(
        init_leaves[n1 + n2 :], post_leaves[n1 + n2 :], strict=False
    ):
        assert not jnp.allclose(before, after), "Node3 params should update"
