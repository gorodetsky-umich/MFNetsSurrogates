import jax
import jax.numpy as jnp
import pytest
from jax import tree_util

from mfnets_surrogates import (
    LinearModel,
    LinearParams,
    MFNetStructureLearner,
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
