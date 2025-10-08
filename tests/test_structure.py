import jax
import jax.numpy as jnp
import pytest
from jax import tree_util

from mfnets_surrogates import (
    MFNetStructureLearner,
    LinearModel,
    LinearParams,
)

@pytest.fixture
def key():
    return jax.random.PRNGKey(42)

def test_tree_flatten_unflatten_roundtrip(key):
    # Build a two-node learner with simple linear base models
    params0 = LinearParams(jnp.eye(2), jnp.zeros(2))
    params1 = LinearParams(jnp.ones((3,2)), jnp.ones(3))
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
    x = jnp.ones((5,2))
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
    x = jax.random.normal(key, (4,2))
    f = learner.run(x)
    expected = delta.run(x)
    assert jnp.allclose(f, expected)

def test_sink_node_mask_enforced(key):
    # Two nodes, force node 0 as sink (no outgoing edges)
    params0 = LinearParams(jnp.zeros((1,1)), jnp.zeros(1))
    params1 = LinearParams(jnp.zeros((1,1)), jnp.zeros(1))
    m0 = LinearModel(params0)
    m1 = LinearModel(params1)
    learner = MFNetStructureLearner([m0, m1], sink_node=0)

    # Mask should zero out row 0
    mask = learner.constraint_mask
    assert mask.shape == (2,2)
    assert jnp.all(mask[0,:] == 0.0)
    # Other rows should remain ones
    assert jnp.all(mask[1,:] == 1.0)
