"""Structure learning module for MFNets."""


import jax
import jax.numpy as jnp
import optax
from jax import tree_util
from jax.tree_util import register_pytree_node_class

from mfnets_surrogates.net_jax import Model


@register_pytree_node_class
class MFNetStructureLearner:
    """
    Learns both the adjacency matrix W and base model parameters for a
    fully-connected graph, discovering a sparse DAG structure.
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
        """
        Flatten parameters (W and base_models) for JAX transformations.
        """
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
        """
        Reconstruct instance from leaves and static data.
        """
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
        Forward pass: compute base outputs, form and solve (I - W^T) F = Δ.
        """
        W = self.adjacency_matrix * self.constraint_mask
        # Compute Δ for each node
        delta = jnp.stack([m.run(x_input) for m in self.base_models], axis=0)
        # Build system matrix A
        A = jnp.eye(self.n_nodes) - W.T
        # Solve linear system for F
        F = jnp.linalg.solve(A, delta)
        return F

    def structure_learning_loss(
        self,
        x_input: jnp.ndarray,
        y_targets: jnp.ndarray,
        supervised_idx: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Loss combining data fit, acyclicity, and sparsity penalties.
        """
        F = self.run(x_input)
        # Data-fit: only supervised nodes
        y_pred = F[supervised_idx]
        mse = jnp.mean((y_pred - y_targets) ** 2)

        W = self.adjacency_matrix * self.constraint_mask
        H = W * W
        expm = jax.scipy.linalg.expm(H)
        h_pen = jnp.trace(expm) - self.n_nodes
        l1 = jnp.sum(jnp.abs(W))

        return mse + self.alpha * h_pen + self.beta * l1

    def fit(
        self,
        x_input: jnp.ndarray,
        y_targets: jnp.ndarray,
        supervised_idx: jnp.ndarray,
        n_iters: int = 1000,
        learning_rate: float = 1e-3,
    ) -> "MFNetStructureLearner":
        """
        Train the structure learner.

        Args:
            x_input: Input features for all samples.
            y_targets: Stacked targets for supervised nodes.
            supervised_idx: Indices of nodes with supervision.
        """
        optimizer = optax.adam(learning_rate)
        state = optimizer.init(self)

        @jax.jit
        def train_step(model, opt_state):
            loss, grads = jax.value_and_grad(
                lambda m: m.structure_learning_loss(
                    x_input, y_targets, supervised_idx
                )
            )(model)
            updates, opt_state = optimizer.update(grads, opt_state, model)
            model = optax.apply_updates(model, updates)
            return model, opt_state, loss

        model = self
        for _ in range(n_iters):
            model, state, _ = train_step(model, state)
        return model
