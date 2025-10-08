import jax
import jax.numpy as jnp
import pytest
from jax import tree_util

from mfnets_surrogates import (
    AutoMFNet,
    LinearModel,
    LinearParams,
    MFNetStructureLearner,
    init_linear_params,
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
    learner = MFNetStructureLearner([m0, m1], sink_node=1, alpha=0.5, beta=2.0)

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
    learner = MFNetStructureLearner([delta], sink_node=None)

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
    learner = MFNetStructureLearner([m0, m1], sink_node=0)

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
    learner = MFNetStructureLearner([delta], sink_node=None)

    # Prepare training data: model 0 is supervised
    x = jax.random.normal(key, (8, 2))
    y = delta.run(x)
    train_data = [(x, y)]

    # Fit with a few iterations at a high lr (should recover exact δ)
    learner.fit(train_data, n_iters=50, learning_rate=1.0)

    # Check that run(x) matches y exactly
    out = learner.run(x)
    assert out.shape == y.shape
    assert jnp.allclose(out, y, atol=1e-6)


def test_structure_learner_partial_supervision(key):
    # Two nodes with same output dim: δ0(x)=x, δ1(x)=2x
    m0 = LinearModel(LinearParams(jnp.eye(3), jnp.zeros(3)))
    m1 = LinearModel(LinearParams(jnp.eye(3) * 2, jnp.zeros(3)))
    learner = MFNetStructureLearner([m0, m1], sink_node=None)

    # Only supervise node 1
    x = jax.random.normal(key, (5, 3))
    y1 = m1.run(x)
    train_data = [None, (x, y1)]

    # Fit; with single edge W[0,1] learnable, it should route node0 -> node1
    learner.fit(train_data, n_iters=100, learning_rate=0.5)

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
    # Build base linear models δ_j(x) = x @ (c_j I) with c_j=1,2,3
    d = 5
    identity = jnp.eye(d)
    m0 = LinearModel(LinearParams(identity, jnp.zeros(d)))
    m1 = LinearModel(LinearParams(2 * identity, jnp.zeros(d)))
    m2 = LinearModel(LinearParams(3 * identity, jnp.zeros(d)))
    # True adjacency
    W_true = jnp.array(
        [
            [0.0, 0.5, 0.2],
            [0.0, 0.0, 0.7],
            [0.0, 0.0, 0.0],
        ]
    )

    def true_run(x):
        Δ = jnp.stack([m.run(x) for m in (m0, m1, m2)], axis=0)
        A = jnp.eye(3) - W_true.T
        flat = Δ.reshape(3, -1)
        sol = jnp.linalg.solve(A, flat)
        return sol.reshape(3, *Δ.shape[1:])

    # Generate data
    x = jax.random.normal(key, (2000, d))
    sol = true_run(x)
    train_data = [(x, sol[i]) for i in range(3)]
    # Fit
    learner = MFNetStructureLearner(
        [m0, m1, m2], sink_node=None, alpha=0.1, beta=0.01
    )
    learner = learner.fit(train_data, n_iters=2000, learning_rate=0.5)
    # Threshold and compare
    threshold = 0.15
    W_learned = learner.adjacency_matrix
    adj_mask = jnp.abs(W_learned) > threshold
    expected = jnp.array(
        [
            [False, True, True],
            [False, False, True],
            [False, False, False],
        ]
    )
    assert jnp.array_equal(adj_mask, expected)


# ----------------------------------------------------------------------
# Phase 2: DAG extraction methods


def test_get_weights_and_mask():
    # Two-node learner: manually set W, then test get_weights & adjacency_mask
    params0 = LinearParams(jnp.zeros((1, 1)), jnp.zeros(1))
    params1 = LinearParams(jnp.zeros((1, 1)), jnp.zeros(1))
    m0 = LinearModel(params0)
    m1 = LinearModel(params1)
    learner = MFNetStructureLearner([m0, m1], sink_node=None)
    learner.adjacency_matrix = jnp.array([[0.0, 0.2], [0.5, 0.0]])
    W = learner.get_weights()
    assert jnp.allclose(W, learner.adjacency_matrix)
    mask = learner.adjacency_mask(threshold=0.3)
    assert mask.tolist() == [[False, False], [True, False]]


def test_to_graph_constructs_correct_dag():
    # Three-node learner: set W and build graph with to_graph()
    base = [
        LinearModel(LinearParams(jnp.zeros((1, 1)), jnp.zeros(1)))
        for _ in range(3)
    ]
    learner = MFNetStructureLearner(base, sink_node=None)
    learner.adjacency_matrix = jnp.array(
        [
            [0.0, 0.6, 0.0],
            [0.0, 0.0, 0.7],
            [0.0, 0.0, 0.0],
        ]
    )
    funcs = {i: base[i] for i in range(3)}
    G = learner.to_graph(node_ids=[0, 1, 2], node_funcs=funcs, threshold=0.5)
    # nodes should be [0,1,2], edges only at (0,1) and (1,2)
    assert list(G.nodes) == [0, 1, 2]
    assert (0, 1) in G.edges
    assert (1, 2) in G.edges
    assert (0, 2) not in G.edges


# ----------------------------------------------------------------------
# Tests for AutoMFNet end-to-end pipeline


def test_auto_mfnet_single_node_pipeline(key):
    # single-node identity mapping
    d = 4
    x = jax.random.normal(key, (20, d))
    base = LinearModel(init_linear_params(key, d, d))
    y = base.run(x)
    auto = AutoMFNet([base], full_model_fn=lambda nid, base, parents: base)
    learner = auto.fit_structure([(x, y)], n_iters=20, learning_rate=1.0)
    assert isinstance(learner, MFNetStructureLearner)
    dag = auto.extract_dag(threshold=0.0)
    assert list(dag.nodes) == [0]
    # Sink node (0) should have no outgoing edges
    assert dag.out_degree(0) == 0
    mfnet = auto.fit_parameters(
        [(x, y)], n_iters=20, learning_rate=1.0, verbose=False
    )
    (pred,) = mfnet.run((0,), x)
    assert jnp.allclose(pred, y)


def test_auto_mfnet_two_node_no_edge(key):
    # two-node chain with only second node supervised; expect no edges
    d = 3
    base0 = LinearModel(init_linear_params(key, d, d))
    base1 = LinearModel(init_linear_params(key, d, d * 2))
    x = jax.random.normal(key, (30, d))
    y1 = base1.run(x)
    auto = AutoMFNet(
        [base0, base1], full_model_fn=lambda nid, base, parents: base
    )
    auto.fit_structure([None, (x, y1)], n_iters=50, learning_rate=0.5)
    dag = auto.extract_dag(threshold=0.1)
    assert set(dag.nodes) == {0, 1}
    # Sink node (1) must have no outgoing edges
    assert dag.out_degree(1) == 0
    mfnet = auto.fit_parameters(
        [None, (x, y1)], n_iters=50, learning_rate=0.5, verbose=False
    )
    (pred1,) = mfnet.run((1,), x)
    assert jnp.allclose(pred1, y1, atol=1e-6)
