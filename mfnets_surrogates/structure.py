"""Structure learning module for MFNets."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import networkx as nx
import optax
from jax import tree_util
from jax.tree_util import register_pytree_node_class

from mfnets_surrogates.net_jax import MFNetJax, Model, mse_loss_graph


@register_pytree_node_class
class MFNetStructureLearner:
    """
    Learns adjacency matrix W and base model parameters.

    For a fully-connected graph, discovering a sparse DAG structure.
    """

    def __init__(
        self,
        base_models: list[Model],
        sink_node: int | None = None,
        alpha: float = 1.0,
        beta: float = 1.0,
    ) -> None:
        """
        Initialize the structure learning engine.

        Args:
            base_models: A list of Model instances for δ_j outputs.
            sink_node: Index of a node to force as sink (no outgoing edges).
            alpha: Weight for acyclicity penalty.
            beta: Weight for L1 sparsity penalty.
        """
        self.n_nodes = len(base_models)
        self.adjacency_matrix = jnp.zeros((self.n_nodes, self.n_nodes))
        mask = jnp.ones_like(self.adjacency_matrix)
        if sink_node is not None:
            mask = mask.at[sink_node, :].set(0.0)
        self.constraint_mask = mask
        self.base_models = base_models
        self.alpha = alpha
        self.beta = beta

    def tree_flatten(self) -> tuple[list[jnp.ndarray], tuple]:
        """Flatten parameters (W and base_models) for JAX transformations."""
        leaves: list[jnp.ndarray] = [self.adjacency_matrix]
        treedefs = []

        for model in self.base_models:
            m_leaves, m_def = tree_util.tree_flatten(model)
            leaves.extend(m_leaves)
            treedefs.append(m_def)

        aux_data = (
            self.n_nodes,
            self.constraint_mask,
            treedefs,
            self.alpha,
            self.beta,
        )
        return leaves, aux_data

    @classmethod
    def tree_unflatten(
        cls, aux_data: tuple, children: list[jnp.ndarray]
    ) -> "MFNetStructureLearner":
        """Reconstruct instance from leaves and static data."""
        n_nodes, constraint_mask, treedefs, alpha, beta = aux_data
        # First child is adjacency_matrix
        adj_matrix = children[0]
        # Next children correspond to base_models
        idx = 1
        base_models = []
        for tdef in treedefs:
            n_leaves = tdef.num_leaves
            m_leaves = children[idx : idx + n_leaves]
            idx += n_leaves
            model = tdef.unflatten(m_leaves)
            base_models.append(model)

        # Identify sink_node by mask
        sink_node = None  # mask only stored, not original sink
        inst = cls(base_models, sink_node=sink_node, alpha=alpha, beta=beta)
        inst.adjacency_matrix = adj_matrix
        inst.constraint_mask = constraint_mask
        return inst

    def run(self, x_input: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass: solve (I - W^T) F = Δ.

        Each scalar W_ij is applied to every coordinate of δ_i(x).
        """
        # Shortcut for single node: just return its raw output
        if self.n_nodes == 1:
            return self.base_models[0].run(x_input)

        # 1) Compute each base-model output δ_j(x) with shape (batch, d_j)
        outputs = [m.run(x_input) for m in self.base_models]
        dims = [o.shape[-1] for o in outputs]
        max_dim = max(dims)

        # 2) Pad each δ_j to width max_dim along last axis
        padded = [
            o
            if o.shape[-1] == max_dim
            else jnp.pad(o, ((0, 0), (0, max_dim - o.shape[-1])))
            for o in outputs
        ]

        # 3) Stack into Δ of shape (n_nodes, batch, max_dim)
        delta = jnp.stack(padded, axis=0)

        # 4) Form A = I - W^T and solve for F in each flattened coordinate
        W = self.adjacency_matrix * self.constraint_mask
        A = jnp.eye(self.n_nodes) - W.T

        # Flatten batch and feature dims -> (n_nodes, batch*max_dim)
        flat_delta = delta.reshape(self.n_nodes, -1)
        flat_F = jnp.linalg.solve(A, flat_delta)

        # Reshape back to (n_nodes, batch, max_dim)
        F = flat_F.reshape(self.n_nodes, *delta.shape[1:])
        return F

    def structure_learning_loss(
        self,
        train_data: list[tuple[jnp.ndarray, jnp.ndarray] | None],
    ) -> jnp.ndarray:
        """Compute loss over datasets: data-fit, DAG & sparsity penalties."""
        # Accumulate MSE only for supervised nodes
        mse_total = 0.0
        for j, entry in enumerate(train_data):
            if entry is not None:
                x_j, y_j = entry
                # run ⇒ shape (n_nodes, batch_j, max_dim)
                # or (batch_j, d) if single node
                F = self.run(x_j)
                if F.ndim == 2:
                    # make it (1, batch, dim) so F[j, ...] works
                    F = F[None, ...]
                d_j = y_j.shape[-1]
                pred_j = F[j, :, :d_j]
                mse_total += jnp.mean((pred_j - y_j) ** 2)

        # Acyclicity penalty
        W = self.adjacency_matrix * self.constraint_mask
        H = W * W
        expm = jax.scipy.linalg.expm(H)
        h_pen = jnp.trace(expm) - self.n_nodes
        # Sparsity penalty
        l1 = jnp.sum(jnp.abs(W))

        return mse_total + self.alpha * h_pen + self.beta * l1

    def fit(
        self,
        train_data: list[tuple[jnp.ndarray, jnp.ndarray] | None],
        n_iters: int = 1000,
        learning_rate: float = 1e-3,
    ) -> "MFNetStructureLearner":
        """
        Train the structure learner.

        Args:
            train_data: List where each element is either (x_j, y_j) or None.
        """
        optimizer = optax.adam(learning_rate)
        state = optimizer.init(self)

        @jax.jit
        def train_step(model, opt_state):
            loss, grads = jax.value_and_grad(
                lambda m: m.structure_learning_loss(train_data)
            )(model)
            updates, opt_state = optimizer.update(grads, opt_state, model)
            model = optax.apply_updates(model, updates)
            return model, opt_state, loss

        model = self
        for _ in range(n_iters):
            model, state, _ = train_step(model, state)
        return model

    def get_weights(self) -> jnp.ndarray:
        """Return learned adjacency matrix W with mask applied."""
        return self.adjacency_matrix * self.constraint_mask

    def adjacency_mask(self, threshold: float) -> jnp.ndarray:
        """Return boolean mask of edges where |W_ij| > threshold."""
        W = self.get_weights()
        return jnp.abs(W) > threshold

    def to_graph(
        self,
        node_ids: Sequence[Any],
        node_funcs: Mapping[Any, Model],
        threshold: float,
    ) -> nx.DiGraph:
        """Convert learned structure to a NetworkX DAG with provided models."""
        mask = self.adjacency_mask(threshold)
        G = nx.DiGraph()
        # 1) Add nodes
        for nid in node_ids:
            if nid not in node_funcs:
                raise KeyError(f"No model provided for node {nid!r}")
            G.add_node(nid, func=node_funcs[nid])
        # 2) Add edges where mask is True, skipping any self-loops
        for i, src in enumerate(node_ids):
            for j, dst in enumerate(node_ids):
                if i == j or not mask[i, j]:
                    continue
                G.add_edge(src, dst)
        return G


class AutoMFNet:
    """
    Two-stage Auto-MFNet orchestrator with separate fit & extract steps.

    1) fit_structure(...) learns W and δ-models.
    2) extract_dag(...) prunes W at any threshold
       and builds a DAG of full models.
    3) fit_parameters(...) trains that DAG with MFNetJax.fit.
    """

    def __init__(
        self,
        base_models: Sequence[Model],
        full_model_fn: Callable[[int, Model, Sequence[Model]], Model],
        sink_node: int | None = None,
        alpha: float = 1.0,
        beta: float = 1.0,
    ):
        self.base_models = list(base_models)
        self.full_model_fn = full_model_fn
        self.sink_node = sink_node
        self.alpha = alpha
        self.beta = beta

        self.learner: MFNetStructureLearner | None = None
        self.dag: nx.DiGraph | None = None
        self.trained_mfnet: MFNetJax | None = None

    def fit_structure(
        self,
        structure_data: list[tuple[jnp.ndarray, jnp.ndarray] | None],
        n_iters: int = 1000,
        learning_rate: float = 1e-3,
    ) -> MFNetStructureLearner:
        """Learn adjacency matrix W and base-model parameters."""
        # Choose sink_node: user-supplied or highest-fidelity supervised node
        if self.sink_node is None:
            sup_idxs = [
                i for i, d in enumerate(structure_data) if d is not None
            ]
            primary_sink = sup_idxs[-1] if sup_idxs else None
        else:
            primary_sink = self.sink_node
        learner = MFNetStructureLearner(
            self.base_models,
            sink_node=primary_sink,
            alpha=self.alpha,
            beta=self.beta,
        )
        self.learner = learner.fit(
            structure_data, n_iters=n_iters, learning_rate=learning_rate
        )
        return self.learner

    def extract_dag(
        self,
        threshold: float,
    ) -> nx.DiGraph:
        """Prune W at threshold and build a DAG with full models."""
        if self.learner is None:
            raise RuntimeError("You must call fit_structure(...) first.")

        node_ids = list(range(self.learner.n_nodes))
        # create initial graph with placeholder funcs
        G = self.learner.to_graph(
            node_ids=node_ids,
            node_funcs=dict.fromkeys(node_ids),
            threshold=threshold,
        )
        for nid in G.nodes:
            base = self.learner.base_models[nid]
            parents = [
                self.learner.base_models[p] for p in G.predecessors(nid)
            ]
            G.nodes[nid]["func"] = self.full_model_fn(nid, base, parents)
        self.dag = G
        return G

    def fit_parameters(
        self,
        param_data: list[tuple[jnp.ndarray, jnp.ndarray]],
        n_iters: int = 5000,
        learning_rate: float = 1e-3,
        loss_fn: Callable = mse_loss_graph,
        verbose: bool = True,
        log_every: int = 100,
    ) -> MFNetJax:
        """Train the full-fidelity DAG with MFNetJax.fit."""
        if self.dag is None:
            raise RuntimeError(
                "You must call extract_dag(...) before fit_parameters()."
            )
        mfnet = MFNetJax(self.dag)
        self.trained_mfnet = mfnet.fit(
            param_data,
            n_iters=n_iters,
            learning_rate=learning_rate,
            loss_fn=loss_fn,
            verbose=verbose,
            log_every=log_every,
        )
        return self.trained_mfnet
