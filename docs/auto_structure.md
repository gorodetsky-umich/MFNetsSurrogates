# AutoMFNet Two-Stage Pipeline

AutoMFNet turns a set of simple *base* models into a fully–trained
multi-fidelity surrogate in two explicit stages:

1. **Structure learning**  
   Learns a sparse, acyclic adjacency matrix `W` using **any** `Model`
   subclass (Linear, MLP, PCE, or custom).  
   Loss = data-MSE + `α·h(W)` (acyclicity) + `β·‖W‖₁` (sparsity).

2. **Parameter learning**  
   Prunes `W` with a user-chosen threshold, replaces each node with a *leaf* or
   *edge* model built by user factories, and trains all parameters on the fixed
   DAG via `MFNetJax.fit`.

---

## How the structure learning works

The core idea is to learn a causal adjacency matrix `W` for the graph of fidelities.
Each node `j` has a base model `δ_j(x)` (a neural network or other surrogate)
that represents its intrinsic output given the primary input `x`.

The full output `F_j` of node `j` in the multi-fidelity network is modeled as:

`F_j = δ_j(x) + Σ_i W_ij F_i`

Where `W_ij` is the learned weight of the edge from node `i` to node `j`.
In matrix form, this becomes `F = Δ + Wᵀ F`, which can be rearranged as:

`(I – Wᵀ) F = Δ   ⇒   F = (I – Wᵀ)⁻¹ Δ`

This equation represents a single linear solve which is fully differentiable
with respect to `W` and the parameters of the `δ` models.

* `α·h(W)` – NOTEARS trace-exponential keeps W acyclic  
    *   `h(W) = tr(exp(W ∘ W)) - N`, where `N` is the number of nodes.
        This acyclicity constraint ensures that the learned graph is a
        Directed Acyclic Graph (DAG), which is fundamental for consistent
        multi-fidelity evaluation.
* `β·‖W‖₁` – L1 pushes W toward sparsity  
    *   The L1 norm `‖W‖₁ = Σ |W_ij|` encourages many `W_ij` elements to be
        exactly zero, leading to a sparse graph structure where only the most
        important connections are retained.
* *Sink* node mask freezes outgoing edges from the highest-fidelity node
    *   A designated "sink" node (typically the highest fidelity) will have all
        its outgoing edge weights `W_sink,j` masked to zero. This ensures that
        the highest fidelity node does not influence any other nodes.

### Thresholding
After learning the adjacency matrix `W`, a `threshold` `τ` is used to
binarize the continuous weights `W_ij` into a discrete graph. Edges `(i,j)`
are kept in the graph if `|W_ij| > τ`, otherwise they are removed. This step
is part of `extract_dag`.

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

### Troubleshooting

* **Graph too dense** – raise the pruning threshold τ.  
* **Training stalls** – lower `α` so the acyclicity penalty is weaker.  
* **Pre-training MSE already tiny** – base models are too expressive; use
  simpler bases (e.g. `LinearModel`).

You may call `extract_dag` with several τ values without repeating Stage 1.

For detailed usage see `examples/mlp_graph.py`.
