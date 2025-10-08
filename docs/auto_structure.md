# AutoMFNet Two-Stage Pipeline

AutoMFNet turns a set of simple *base* models into a fully–trained
multi-fidelity surrogate in two explicit stages:

1. **Structure learning**  
   Learns a sparse, acyclic adjacency matrix `W` using linear base models and
   penalties for acyclicity (`α·h(W)`) and sparsity (`β·‖W‖₁`).

2. **Parameter learning**  
   Prunes `W` with a user-chosen threshold, replaces each node with a *leaf* or
   *edge* model built by user factories, and trains all parameters on the fixed
   DAG via `MFNetJax.fit`.

---

## Quick example

```python
import jax, jax.numpy as jnp
from mfnets_surrogates import (
    AutoMFNet, init_linear_model,
    init_mlp_model, init_mlp_enhancement_model,
)

key = jax.random.PRNGKey(0)
d_in, d_out = 3, 1
x = jax.random.normal(key, (128, d_in))
y_hf = jnp.sin(jnp.sum(x, -1, keepdims=True))

# Stage-1 base models (linear)
bases = [init_linear_model(key, d_in, d_out) for _ in range(3)]
auto = AutoMFNet(sink_node=2, alpha=0.5, beta=0.05)

# structure_data: only highest-fidelity node (2) supervised
auto.fit_structure(bases, [None, None, (x, y_hf)], n_iters=2000)

# Build DAG with factories
leaf_fn = lambda nid, dim: init_mlp_model(key, [d_in, 32, dim])
edge_fn = lambda nid, dim, pd: init_mlp_enhancement_model(
    key, [d_in + sum(pd), 32, 32, dim]
)
dag = auto.extract_dag(0.1, leaf_fn, edge_fn)

# Stage-2 training
mfnet = auto.fit_parameters(dag, [None, None, (x, y_hf)], n_iters=5000)
print("final MSE", jnp.mean((mfnet.run((2,), x)[0] - y_hf) ** 2))
```

---

## API

| Step                | Method                               | Notes |
|---------------------|--------------------------------------|-------|
| Structure learning  | `fit_structure(base_models, data)`   | `base_models`: simple models<br>`data`: per-node `(x,y)` or `None` |
| DAG extraction      | `extract_dag(τ, leaf_fn, edge_fn)`   | `leaf_fn(node_id, d)`<br>`edge_fn(node_id, d, parent_dims)` |
| Parameter learning  | `fit_parameters(dag, data)`          | `dag`: output of `extract_dag` |

Choose a pruning threshold τ that balances sparsity vs. accuracy.  You can
call `extract_dag` multiple times with different thresholds without repeating
Stage 1.

For detailed usage see `examples/mlp_graph.py`.
