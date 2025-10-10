import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest
from jax import tree_util
from unittest.mock import Mock

from mfnets_surrogates import (
    AutoMFNet,
    LinearModel,
    LinearParams,
    MFNetStructureLearner,
    Model, # Added import for Mock
    init_linear_model, # Added import for convenience functions
)


@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


def test_tree_flatten_unflatten_roundtrip(key):
    # Build a two-node learner with simple linear base models
    params0 = LinearParams(jnp.eye(2), jnp.zeros(2))
    params1 = LinearParams(jnp.ones((3, 2)), jnp.ones(3))
    m0 = LinearModel(params0)
    m1 = LinearModel(params1)
    learner = MFNetStructureLearner(node_ids=[0, 1], base_models=[m0, m1], sink_node=1, alpha=0.5, beta=2.0)

    leaves, aux = tree_util.tree_flatten(learner)
    rebuilt = tree_util.tree_unflatten(aux, leaves)

    # The static attributes should match
    assert rebuilt.n_nodes == learner.n_nodes
    assert rebuilt.alpha == pytest.approx(learner.alpha)
    assert rebuilt.beta == pytest.approx(learner.beta)
    # The constraint mask row for sink_node must be zeros
    assert jnp.all(rebuilt.constraint_mask[1, :] == 0.0)
    # And run on a sample input should match
    x = jnp.ones((5, 2))
    out_orig = learner.run(x)
    out_rebuilt = rebuilt.run(x)
    assert jnp.allclose(out_orig, out_rebuilt, atol=1e-6)


def test_single_node_forward_identity(key):
    # One node: delta(x) = 2 * x + 3
    weight = jnp.eye(2) * 2.0
    bias = jnp.ones(2) * 3.0
    delta = LinearModel(LinearParams(weight, bias))
    learner = MFNetStructureLearner(node_ids=[0], base_models=[delta], sink_node=None)

    # With W=0, F == Δ
    x = jax.random.normal(key, (4, 2))
    f = learner.run(x)
    expected = delta.run(x)
    assert jnp.allclose(f, expected)


def test_sink_node_mask_enforced(key):
    # Two nodes, force node 0 as sink (no outgoing edges)
    params0 = LinearParams(jnp.zeros((1, 1)), jnp.zeros(1))
    params1 = LinearParams(jnp.zeros((1, 1)), jnp.zeros(1))
    m0 = LinearModel(params0)
    m1 = LinearModel(params1)
    learner = MFNetStructureLearner(node_ids=[0, 1], base_models=[m0, m1], sink_node=0)

    # Mask should zero out row 0
    mask = learner.constraint_mask
    assert mask.shape == (2, 2)
    assert jnp.all(mask[0, :] == 0.0)
    # Other rows should remain ones
    assert jnp.all(mask[1, :] == 1.0)


# ----------------------------------------------------------------------
# New tests for the fit(train_data, ...) API


def test_structure_learner_single_node_fit(key):
    # One node: δ(x) = 2*x + 3
    weight = jnp.eye(2) * 2.0
    bias = jnp.ones(2) * 3.0
    delta = LinearModel(LinearParams(weight, bias))

    node_ids = [0]  # External node ID
    base_models = [delta]  # Ordered list of models

    # Prepare training data: model 0 is supervised
    x = jax.random.normal(key, (8, 2))
    y = delta.run(x)
    train_data = {0: (x, y)}  # Use dictionary for train_data

    # Fit with a few iterations at a high lr (should recover exact δ)
    learner = MFNetStructureLearner(
        node_ids=node_ids,
        base_models=base_models,
        sink_node=0,
        alpha=0.0,
        beta=0.0,
    )
    learner = learner.fit(train_data, n_iters=50, learning_rate=1.0)

    # Check that run(x) matches y exactly
    out = learner.run(x)
    assert out.shape == y.shape
    assert jnp.allclose(out, y, atol=1e-6)

    # Test `to_graph` method
    dag = learner.to_graph(threshold=0.0)
    assert len(dag.nodes) == 1
    assert 0 in dag.nodes
    assert len(dag.edges) == 0
    # Check that the model is an instance of LinearModel and its parameters are close
    assert isinstance(dag.nodes[0]["func"], LinearModel)
    npt.assert_allclose(dag.nodes[0]["func"].params.w, delta.params.w, atol=1e-6)
    npt.assert_allclose(dag.nodes[0]["func"].params.b, delta.params.b, atol=1e-6)


def test_structure_learner_partial_supervision(key):
    # Two nodes with same output dim: δ0(x)=x, δ1(x)=2x
    m0 = LinearModel(LinearParams(jnp.eye(3), jnp.zeros(3)))
    m1 = LinearModel(LinearParams(jnp.eye(3) * 2, jnp.zeros(3)))

    node_ids = [0, 1]  # External node IDs
    base_models = [m0, m1]  # Ordered list of models

    # Only supervise node 1
    x = jax.random.normal(key, (5, 3))
    y1 = m1.run(x)
    train_data = {
        1: (x, y1)
    }  # Use dictionary for train_data, key is external node ID

    # Fit; with single edge W[0,1] learnable, it should route node0 -> node1
    learner = MFNetStructureLearner(
        node_ids=node_ids,
        base_models=base_models,
        sink_node=None,
        alpha=0.0,
        beta=0.0,
    )
    learner = learner.fit(train_data, n_iters=100, learning_rate=0.5)

    # After training, if we run on x:
    F = learner.run(x)  # shape (2, batch, dim)
    F0, F1 = F[0], F[1]

    # Node 0 still equals its δ0
    assert jnp.allclose(F0, m0.run(x), atol=1e-5)
    # Node 1 matches its δ1
    assert jnp.allclose(F1, y1, atol=1e-5)


def test_structure_learner_recovers_known_dag(key):
    """Stage 1: learn W for a 3-node linear DAG.

    True structure:
        0 → 1 with weight 0.5
        0 → 2 with weight 0.2
        1 → 2 with weight 0.7
    """
    k0, k1, k2, k_data = jax.random.split(key, 4)
    n_nodes, d_in, d_out = 3, 5, 1
    # Build base linear models δ_j(x) = x @ (c_j I) with c_j=1,2,3
    x_train = jax.random.normal(k_data, (2000, d_in))
    model0 = LinearModel(init_linear_params(k0, d_in, d_out))
    model1 = LinearModel(init_linear_params(k1, d_in, d_out))
    model2 = LinearModel(init_linear_params(k2, d_in, d_out))

    node_ids = [0, 1, 2]  # External node IDs
    base_models = [model0, model1, model2]  # Ordered list of base models

    # Define a 'true' MFNet graph and generate data from it
    def true_run(x):
        # A simple linear model to simulate some data
        return jnp.dot(x, jnp.array([[1.0], [2.0], [3.0], [4.0], [5.0]])) + 0.5

    # Generate training data for all nodes
    y_train_0 = true_run(x_train)
    y_train_1 = 0.5 * y_train_0 + true_run(x_train)
    y_train_2 = 0.2 * y_train_0 + 0.7 * y_train_1 + true_run(x_train)

    # Training data as a dictionary mapping external node IDs to (x,y)
    train_data = {
        0: (x_train, y_train_0),
        1: (x_train, y_train_1),
        2: (x_train, y_train_2),
    }

    learner = MFNetStructureLearner(
        node_ids=node_ids,
        base_models=base_models,
        sink_node=2,
        alpha=1.0,
        beta=1.0,
    )
    learner = learner.fit(train_data, n_iters=10000, learning_rate=1e-3)

    # Check the learned adjacency matrix.
    # We expect W[0,1] and W[1,2] to be strong, W[0,2] weaker, and others small.
    # Node 2 is sink, so W[2,:] should be near zero after mask.
    npt.assert_array_less(
        learner.adjacency_matrix[2, :], 1e-2
    )  # 2 is sink

    # Check recovered DAG
    dag = learner.to_graph(
        threshold=0.1
    )  # to_graph no longer needs node_ids/node_funcs
    assert len(dag.nodes) == n_nodes
    # Assert edges using the external node IDs
    assert dag.has_edge(0, 1)  # 0 -> 1
    assert dag.has_edge(0, 2)  # 0 -> 2
    assert dag.has_edge(1, 2)  # 1 -> 2
    assert not dag.has_edge(1, 0)
    assert not dag.has_edge(2, 0)
    assert not dag.has_edge(2, 1)
    assert isinstance(dag.nodes[0]["func"], Model) # Check that base models are attached


# ----------------------------------------------------------------------
# Phase 2: DAG extraction methods


def test_get_weights_and_mask():
    # Setup for get_weights and adjacency_mask
    d = 1
    k0, k1, k2 = jax.random.split(jax.random.PRNGKey(0), 3)
    base_models_list = [
        init_linear_model(k0, d, d),
        init_linear_model(k1, d, d),
        init_linear_model(k2, d, d),
    ]
    node_ids = [0, 1, 2]
    learner = MFNetStructureLearner(
        node_ids=node_ids, base_models=base_models_list, sink_node=2
    )
    # Manually set adjacency matrix to simulate learning
    learner.adjacency_matrix = jnp.array(
        [[0.0, 0.5, 0.2], [0.0, 0.0, 0.7], [0.0, 0.0, 0.0]]
    )
    # constraint_mask is set by sink_node during __init__ (node 2 is sink)
    # expected mask is [[1., 1., 1.], [1., 1., 1.], [0., 0., 0.]] at internal indices
    W = learner.get_weights()
    expected_W = jnp.array([[0.0, 0.5, 0.2], [0.0, 0.0, 0.7], [0.0, 0.0, 0.0]])
    npt.assert_allclose(W, expected_W, atol=1e-6)

    mask_strict = learner.adjacency_mask(threshold=0.6)
    expected_mask_strict = jnp.array(
        [[False, False, False], [False, False, True], [False, False, False]]
    )
    npt.assert_array_equal(mask_strict, expected_mask_strict)

    mask_loose = learner.adjacency_mask(threshold=0.1)
    expected_mask_loose = jnp.array(
        [[False, True, True], [False, False, True], [False, False, False]]
    )
    npt.assert_array_equal(mask_loose, expected_mask_loose)


def test_to_graph_constructs_correct_dag():
    # Create a mock adjacency matrix for a 3-node graph with arbitrary node IDs
    # Node IDs: 10, 20, 30
    mock_model = Mock(spec=Model)
    mock_model.output_dim.return_value = 1

    node_ids = [10, 20, 30]  # External IDs for nodes
    base_models = [
        mock_model,
        mock_model,
        mock_model,
    ]  # Ordered list of base models

    learner = MFNetStructureLearner(
        node_ids=node_ids,
        base_models=base_models,
        sink_node=30,  # Sink is external ID
    )
    # Internal adjacency_matrix is 0-indexed: 0->1, 0->2, 1->2
    learner.adjacency_matrix = jnp.array(
        [[0.0, 0.8, 0.1], [0.0, 0.0, 0.6], [0.0, 0.0, 0.0]]
    )

    dag = learner.to_graph(
        threshold=0.5
    )  # to_graph no longer takes node_ids/node_funcs

    assert len(dag.nodes) == len(node_ids)
    assert set(dag.nodes) == set(node_ids)  # Check external node IDs
    assert dag.edges == {(10, 20), (20, 30)}  # Edges should use external IDs
    assert dag.nodes[10]["func"] is mock_model  # Check base model is attached


# ----------------------------------------------------------------------
# Tests for AutoMFNet end-to-end pipeline


def test_auto_mfnet_single_node_pipeline(key):
    # Tests a single-node setup with AutoMFNet
    d = 1
    x, y = jnp.array([[1.0], [2.0]]), jnp.array([[2.0], [4.0]])

    model0 = init_linear_model(key, d_in=1, d_out=1)

    node_ids = [0]  # External node ID
    base_models_map = {0: model0}  # Map external ID to base model
    structure_data = {0: (x, y)}  # Map external ID to training data
    param_data = {0: (x, y)}  # Map external ID to parameter fitting data

    # AutoMFNet setup
    auto = AutoMFNet(sink_node=0)
    learner = auto.fit_structure(
        node_ids=node_ids,
        base_models=base_models_map,
        structure_data=structure_data,
        n_iters=100,
    )

    # leaf_model_fn and edge_model_fn now accept external node ID (Any)
    def leaf_model_fn(nid: Any, dim: int):
        return init_linear_model(
            jax.random.PRNGKey(hash(nid)), d_in=dim, d_out=dim
        )

    def edge_model_fn(nid: Any, dim: int, parent_dims: list[int]):
        return init_linear_model(
            jax.random.PRNGKey(hash(nid)),
            d_in=dim + sum(parent_dims),
            d_out=dim,
        )

    dag = auto.extract_dag(
        threshold=0.1, leaf_model_fn=leaf_model_fn, edge_model_fn=edge_model_fn
    )
    assert len(dag.nodes) == 1
    assert 0 in dag.nodes
    assert len(dag.edges) == 0
    assert isinstance(
        dag.nodes[0]["func"], LinearModel
    )  # Should now be a leaf model

    mfnet = auto.fit_parameters(
        dag, param_data, n_iters=100, learning_rate=1.0, verbose=False
    )
    (pred,) = mfnet.run((0,), x)
    npt.assert_allclose(pred, y, atol=1e-5)


def test_auto_mfnet_two_node_no_edge(key):
    # Tests a two-node AutoMFNet where no edge is expected due to sparsity
    k1, k2 = jax.random.split(key, 2)
    # Data for two nodes, no interdependence
    x1, y1 = jnp.array([[1.0], [2.0]]), jnp.array([[2.0], [4.0]])
    x2, y2 = jnp.array([[3.0], [4.0]]), jnp.array([[6.0], [8.0]])

    model1 = init_linear_model(k1, d_in=1, d_out=1)
    model2 = init_linear_model(k2, d_in=1, d_out=1)

    node_ids = [1, 2]  # External node IDs
    base_models_map = {1: model1, 2: model2}  # Map external ID to base model
    structure_data = {
        1: (x1, y1),
        2: (x2, y2),
    }  # Map external ID to training data
    param_data = {
        1: (x1, y1),
        2: (x2, y2),
    }  # Map external ID to parameter fitting data

    # AutoMFNet setup: force sparsity to get no edges. sink_node=2 (external ID)
    auto = AutoMFNet(
        sink_node=2, alpha=0.0, beta=1.0
    )  # beta=1.0 promotes sparsity
    learner = auto.fit_structure(
        node_ids=node_ids,
        base_models=base_models_map,
        structure_data=structure_data,
        n_iters=1000,
    )

    # leaf_model_fn and edge_model_fn now accept external node ID (Any)
    def leaf_model_fn(nid: Any, dim: int):
        return init_linear_model(
            jax.random.PRNGKey(hash(nid)), d_in=dim, d_out=dim
        )

    def edge_model_fn(nid: Any, dim: int, parent_dims: list[int]):
        return init_linear_model(
            jax.random.PRNGKey(hash(nid)),
            d_in=dim + sum(parent_dims),
            d_out=dim,
        )

    dag = auto.extract_dag(
        threshold=0.1, leaf_model_fn=leaf_model_fn, edge_model_fn=edge_model_fn
    )

    assert len(dag.nodes) == 2
    assert set(dag.nodes) == {1, 2}
    assert len(dag.edges) == 0  # No edges due to sparsity penalty

    mfnet = auto.fit_parameters(
        dag, param_data, n_iters=2000, learning_rate=1.0, verbose=False
    )
    (pred1,) = mfnet.run((1,), x1)
    (pred2,) = mfnet.run((2,), x2)
    npt.assert_allclose(pred1, y1, atol=1e-5)
    npt.assert_allclose(pred2, y2, atol=1e-5)
