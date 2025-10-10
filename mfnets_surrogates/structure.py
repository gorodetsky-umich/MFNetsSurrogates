"""Structure learning module for MFNets."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

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
        node_ids: Sequence[
            Any
        ],  # New: Explicit sequence of node identifiers (external IDs)
        base_models: Sequence[
            Model
        ],  # Sequence must correspond to `node_ids` by index
        sink_node: Any | None = None,  # Now accepts external node ID
        alpha: float = 1.0,
        beta: float = 1.0,
    ) -> None:
        """
        Initialize the structure learning engine.

        Args:
            node_ids: A sequence of identifiers for the nodes in the graph.
                      The order defines the internal 0-indexed mapping used for
                      adjacency.
            base_models: A list of Model instances for δ_j outputs. Must be
                         ordered corresponding to `node_ids`.
            sink_node: Identifier of a node to force as sink (no outgoing
                       edges). Must be one of the `node_ids`. If None, no node
                       is explicitly masked as a sink.
            alpha: Weight for acyclicity penalty.
            beta: Weight for L1 sparsity penalty.
        """
        if len(node_ids) != len(base_models):
            raise ValueError(
                "node_ids and base_models must have the same length."
            )

        self.node_ids = tuple(
            node_ids
        )  # Store node_ids as a tuple for immutability
        self.node_to_idx = {
            node_id: i for i, node_id in enumerate(self.node_ids)
        }
        self.idx_to_node = list(
            self.node_ids
        )  # For unflatten, ensures consistent order

        self.n_nodes = len(self.node_ids)
        self.adjacency_matrix = jnp.zeros((self.n_nodes, self.n_nodes))
        mask = jnp.ones_like(self.adjacency_matrix)

        if sink_node is not None:
            if sink_node not in self.node_to_idx:
                raise ValueError(
                    f"Sink node '{sink_node}' not found in provided node_ids."
                )
            sink_idx = self.node_to_idx[sink_node]
            mask = mask.at[sink_idx, :].set(0.0)
        self.constraint_mask = mask
        self.base_models = list(
            base_models
        )  # Ensure it's a list for internal use
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
            self.node_ids,  # New: Store node_ids for reconstruction
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
        node_ids, n_nodes, constraint_mask, treedefs, alpha, beta = (
            aux_data  # Unpack node_ids
        )
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

        # Reconstruct with the actual node_ids and base_models.
        # Sink node is implicitly handled by the constraint_mask.
        inst = cls(
            node_ids=node_ids,  # Pass node_ids for internal mapping reconstruction
            base_models=base_models,
            sink_node=None,  # When unflattening, sink is encoded in mask
            alpha=alpha,
            beta=beta,
        )
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
            # mypy: explicit cast to satisfy static checker. This relies on
            # base_models[0] being the only model, which aligns with node_ids[0].
            return cast(jnp.ndarray, self.base_models[0].run(x_input))

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
        # Explicit cast so mypy knows this is an Array, not Any
        return cast(jnp.ndarray, F)

    def structure_learning_loss(
        self,
        train_data: Mapping[
            Any, tuple[jnp.ndarray, jnp.ndarray]
        ],  # New: train_data is now a dict
    ) -> jnp.ndarray:
        """Compute loss over datasets: data-fit, DAG & sparsity penalties."""
        # Accumulate MSE only for supervised nodes
        mse_total: jnp.ndarray = jnp.array(0.0)
        num_supervised_nodes = 0

        # Iterate through internal node indices (j) and external node IDs
        # (node_id_ext)
        for j, node_id_ext in enumerate(self.idx_to_node):
            if node_id_ext in train_data:  # Check if this node has supervision
                x_j, y_j = train_data[node_id_ext]

                # self.run(x_j) computes F for all nodes based on this x_j input
                F = self.run(x_j)

                if (
                    self.n_nodes == 1
                ):  # Special case for a single node, run() returns (batch, d)
                    # Make sure F has the correct batch and feature dimensions
                    pred_j = F
                else:  # Multi-node case
                    d_j = y_j.shape[-1]
                    # F has shape (n_nodes, batch, max_dim), select for node j
                    # and trim padding
                    pred_j = F[j, :, :d_j]

                mse_total += jnp.mean((pred_j - y_j) ** 2)
                num_supervised_nodes += 1

        if num_supervised_nodes > 0:
            mse_total /= num_supervised_nodes
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
        train_data: Mapping[
            Any, tuple[jnp.ndarray, jnp.ndarray]
        ],  # New: train_data is now a dict
        n_iters: int = 1000,
        learning_rate: float = 1e-3,
    ) -> "MFNetStructureLearner":
        """
        Train the structure learner.

        Args:
            train_data: Dictionary where keys are external node IDs and values
                        are (x, y) tuples. Only nodes present in this mapping
                        will contribute to the data-fit portion of the loss.
        """
        if not train_data and self.n_nodes > 0:
            raise ValueError(
                "train_data cannot be empty when learning structure for "
                "multiple nodes."
            )
        optimizer = optax.adam(learning_rate)
        state = optimizer.init(self)

        @jax.jit
        def train_step(
            model: "MFNetStructureLearner",
            opt_state: optax.OptState,
            current_train_data: Mapping[
                Any, tuple[jnp.ndarray, jnp.ndarray]
            ],  # Pass data to jitted fn
        ) -> tuple["MFNetStructureLearner", optax.OptState, jnp.ndarray]:
            loss, grads = jax.value_and_grad(
                lambda m: m.structure_learning_loss(current_train_data)
            )(model)
            updates, opt_state = optimizer.update(grads, opt_state, model)
            model = optax.apply_updates(model, updates)
            return model, opt_state, loss

        model = self
        for _step_idx in range(n_iters):  # Renamed step_idx to _step_idx
            model, state, _ = train_step(
                model, state, train_data
            )  # Pass train_data to the jitted step
        return model

    def get_weights(self) -> jnp.ndarray:
        """Return learned adjacency matrix W with mask applied."""
        return self.adjacency_matrix * self.constraint_mask

    def adjacency_mask(self, threshold: float) -> jnp.ndarray:
        """Return boolean mask of edges where |W_ij| > threshold."""
        W = self.get_weights()
        return jnp.abs(W) > threshold

    def to_graph(self, threshold: float) -> nx.DiGraph:
        """Convert learned structure to a NetworkX DAG with associated base
        models.

        The nodes in the returned graph will use the external node IDs stored
        in `self.node_ids`, and each node will have an attribute 'func'
        containing its corresponding base model from `self.base_models`.
        """
        mask = self.adjacency_mask(threshold)
        G = nx.DiGraph()
        # 1) Add nodes with their corresponding base_models
        for i, node_id_ext in enumerate(self.idx_to_node):
            G.add_node(node_id_ext, func=self.base_models[i])
        # 2) Add edges where mask is True, skipping any self-loops
        for i, src_node_id_ext in enumerate(self.idx_to_node):
            for j, dst_node_id_ext in enumerate(self.idx_to_node):
                if i == j or not mask[i, j]:
                    continue
                G.add_edge(src_node_id_ext, dst_node_id_ext)
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
        sink_node: Any | None = None,  # New: now accepts external node ID
        alpha: float = 1.0,
        beta: float = 1.0,
    ):
        """AutoMFNet orchestrator.

        Args:
            sink_node: index of node to freeze as sink (no outgoing edges).
            alpha: weight for acyclicity penalty.
            beta: weight for sparsity penalty.
        """
        self.sink_node = sink_node
        self.alpha = alpha
        self.beta = beta

        self.node_ids: Sequence[Any] | None = (
            None  # Store the ordered external node IDs
        )
        self.base_models_map: Mapping[Any, Model] | None = (
            None  # Store models by external ID
        )
        self.learner: MFNetStructureLearner | None = None
        self.dag: nx.DiGraph | None = None
        self.trained_mfnet: MFNetJax | None = None

    def fit_structure(
        self,
        node_ids: Sequence[Any],  # New: Explicit order of node IDs (external)
        base_models: Mapping[
            Any, Model
        ],  # New: Mapping from external ID to base model
        structure_data: Mapping[
            Any, tuple[jnp.ndarray, jnp.ndarray]
        ],  # New: Mapping from external ID to data
        n_iters: int = 1000,
        learning_rate: float = 1e-3,
    ) -> MFNetStructureLearner:
        """Learn adjacency matrix W and base-model parameters."""
        self.node_ids = node_ids  # Store the canonical node order
        self.base_models_map = (
            base_models  # Store the map of base models by external ID
        )

        # Create an ordered list of base models based on node_ids for the
        # learner's __init__
        ordered_base_models = [base_models[nid] for nid in node_ids]

        # Choose sink_node: user-supplied or highest-fidelity supervised node
        primary_sink_id = self.sink_node
        if primary_sink_id is None:
            # Find the last node in the provided node_ids sequence that has
            # training data
            supervised_node_ids = [
                nid for nid in node_ids if nid in structure_data
            ]
            primary_sink_id = (
                supervised_node_ids[-1] if supervised_node_ids else None
            )
        else:
            primary_sink_id = self.sink_node
        learner = MFNetStructureLearner(
            node_ids=node_ids,  # Pass the ordered list of external node IDs
            base_models=ordered_base_models,  # Pass the ordered list of models
            sink_node=primary_sink_id,  # Pass the external sink node ID
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
        leaf_model_fn: Callable[
            [Any, int], Model
        ],  # New: nid is Any (external ID)
        edge_model_fn: Callable[
            [Any, int, Sequence[int]], Model
        ],  # New: nid is Any (external ID)
    ) -> nx.DiGraph:
        """Prune W at threshold and build a DAG with full models.

        leaf_model_fn(node_id, node_dim) -> Model
        edge_model_fn(node_id, node_dim, parent_dims) -> Model
        """
        if self.learner is None:
            raise RuntimeError("You must call fit_structure(...) first.")
        if (
            self.base_models_map is None or self.node_ids is None
        ):  # pragma: no cover
            raise RuntimeError("fit_structure() must be called first.")

        # `to_graph` now returns a graph with arbitrary external node IDs and
        # base models as funcs
        G = self.learner.to_graph(threshold=threshold)

        # Replace base models with leaf/edge models using the provided functions
        for nid in G.nodes:  # Iterate over external node IDs
            # Lookup the original base model's output dimension using the
            # external ID
            original_base_model = self.base_models_map[nid]
            node_dim = original_base_model.output_dim()
            parent_ids = list(G.predecessors(nid))

            # Lookup parent dimensions using external IDs in base_models_map
            parent_dims = [
                self.base_models_map[p_id].output_dim() for p_id in parent_ids
            ]

            if not parent_ids:
                G.nodes[nid]["func"] = leaf_model_fn(nid, node_dim)
            else:
                G.nodes[nid]["func"] = edge_model_fn(
                    nid, node_dim, parent_dims
                )
        self.dag = G
        return G

    def fit_parameters(
        self,
        dag: nx.DiGraph,
        param_data: Mapping[
            Any, tuple[jnp.ndarray, jnp.ndarray]
        ],  # New: Mapping from external ID to data
        n_iters: int = 5000,
        learning_rate: float = 1e-3,
        loss_fn: Callable = mse_loss_graph,
        verbose: bool = True,
        log_every: int = 100,
    ) -> MFNetJax:
        """Train the full-fidelity DAG by fitting its parameters with
        MFNetJax.fit.
        """
        # Need to convert param_data dict to a list ordered by dag nodes for
        # MFNetJax.fit
        ordered_param_data = []
        for node_id_ext in sorted(
            dag.nodes
        ):  # Ensure a consistent order for MFNetJax.fit
            if node_id_ext not in param_data:
                raise ValueError(
                    f"Training data missing for node {node_id_ext} required "
                    "by DAG for parameter fitting."
                )
            ordered_param_data.append(param_data[node_id_ext])

        mfnet = MFNetJax(dag)
        self.trained_mfnet = mfnet.fit(
            ordered_param_data,  # Pass the ordered list of data
            n_iters=n_iters,
            learning_rate=learning_rate,
            loss_fn=loss_fn,
            verbose=verbose,
            log_every=log_every,
        )
        return self.trained_mfnet
