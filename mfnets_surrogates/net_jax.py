"""
Core JAX implementation for Multi-Fidelity Surrogate Networks (MFNets).

This module defines the main `MFNetJax` class, which acts as a JAX-compatible
PyTree container for a graph of surrogate models. It also provides a set of
basic linear models that can be used as nodes within the graph.

The design philosophy is to make the entire graph structure differentiable and
optimizable with JAX-based tools like `jaxopt`.
"""

from collections.abc import Callable
from functools import partial
from itertools import combinations_with_replacement
from typing import Any, NamedTuple

import jax
import jax.lax as lax
import jax.nn as jnn
import jax.numpy as jnp
import networkx as nx
import numpy as np
from jax import tree_util
from jax.tree_util import register_pytree_node_class
from scipy.special import comb


@register_pytree_node_class
class MFNetJax:
    """
    A JAX-compatible multi-fidelity network represented by a directed graph.

    This class wraps a `networkx.DiGraph` where each node contains a callable
    "func" that represents a surrogate model. It is registered as a JAX PyTree,
    allowing its parameters to be transparently handled by JAX transformations
    like `jax.grad`, `jax.jit`, and optimizers in `jaxopt`.

    Attributes
    ----------
        graph (nx.DiGraph): The graphical representation of the MF network.
        eval_order (list): A topologically sorted list of nodes for execution.
        parents (dict): A mapping from each node to its direct predecessors.
        ancestors (dict): A mapping from each node to the set of all its ancestors.
    """

    def __init__(self, graph: nx.DiGraph):
        """
        Initialize the multifidelity network.

        Args:
            graph: A networkx.DiGraph where each node's data dictionary must contain
                a "func" key pointing to a JAX-compatible model object.
        """
        self.graph = graph
        self.eval_order = list(nx.topological_sort(self.graph))
        self.parents = {n: list(self.graph.predecessors(n)) for n in self.eval_order}
        self.ancestors = {n: set(nx.ancestors(self.graph, n)) for n in self.eval_order}

    def tree_flatten(self):
        """
        Flatten the MFNetJax instance into its dynamic leaves and static auxiliary data.

        This method is required for JAX PyTree registration.

        Returns
        -------
            A tuple containing a list of all model parameters (leaves) and a tuple
            of auxiliary data (graph structure and PyTree definitions).
        """
        leaves = []
        treedefs = []

        for node in self.eval_order:
            func = self.graph.nodes[node]["func"]
            f_leaves, f_treedef = tree_util.tree_flatten(func)
            leaves.extend(f_leaves)
            treedefs.append(f_treedef)

        # Encode graph structure statically
        nodes = tuple(self.eval_order)
        edges = tuple((p, n) for n in self.eval_order for p in self.parents[n])
        aux_data = (nodes, edges, tuple(treedefs))
        return leaves, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Reconstruct an MFNetJax instance from static data and dynamic leaves.

        This method is required for JAX PyTree registration.

        Args:
            aux_data: Static data containing graph structure and treedefs.
            children: A flat list of all model parameters (leaves).

        Returns
        -------
            A reconstructed MFNetJax instance.
        """
        nodes, edges, treedefs = aux_data

        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)

        model = cls(graph)

        # Reconstruct each node's function from its corresponding leaves
        leaf_idx = 0
        for node, tdef in zip(model.eval_order, treedefs, strict=False):
            n_leaves = tdef.num_leaves
            func_leaves = children[leaf_idx : leaf_idx + n_leaves]
            leaf_idx += n_leaves
            func = tdef.unflatten(func_leaves)
            model.graph.nodes[node]["func"] = func

        return model

    @partial(jax.jit, static_argnums=(0, 1))
    def run(
        self, target_nodes: tuple[Any, ...], xinput: jnp.ndarray
    ) -> tuple[jnp.ndarray, ...]:
        """
        Evaluate the graph to compute outputs for the specified target nodes.

        This method efficiently evaluates only the necessary subgraph required to
        compute the outputs for the target nodes.

        Args:
            target_nodes: A tuple of node identifiers for which to compute outputs.
            xinput: The input data array of shape (n_samples, n_features).

        Returns
        -------
            A tuple of output arrays, one for each target node, in the same order.
        """
        # Determine the minimal set of nodes that need to be evaluated
        needed: set[Any] = set()
        for t in target_nodes:
            needed.update(self.ancestors[t])
            needed.add(t)

        evals: dict[Any, jnp.ndarray] = {}
        for node in self.eval_order:
            if node in needed:
                parent_nodes = self.parents[node]
                func = self.graph.nodes[node]["func"]
                if not parent_nodes:
                    # Root node: only takes the primary input
                    val = func.run(xinput)
                else:
                    # Child node: takes primary input and concatenated parent outputs
                    parent_vals = [evals[p] for p in parent_nodes]
                    cat_input = jnp.concatenate(parent_vals, axis=-1)
                    val = func.run(xinput, cat_input)
                evals[node] = val
        return tuple(evals[n] for n in target_nodes)


# --- Model Definitions ---


class LinearParams(NamedTuple):
    """Parameters for a linear model."""

    weight: jnp.ndarray
    bias: jnp.ndarray


class Model:
    """Base class for all models to ensure they are registered as PyTrees."""

    def tree_flatten(self):
        """Flatten the model's parameters into a list of arrays (leaves)."""
        raise NotImplementedError

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Unflatten parameter arrays back into a model instance."""
        raise NotImplementedError


@register_pytree_node_class
class LinearModel(Model):
    """A simple linear model: y = Wx + b."""

    def __init__(self, params: LinearParams):
        """Initialize the model with its parameters."""
        self.params = params

    def tree_flatten(self):
        """Flatten the model's parameters into a list of arrays (leaves)."""
        return (self.params,), {}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Unflatten parameter arrays back into a model instance."""
        return cls(children[0])

    @partial(jax.vmap, in_axes=(None, 0))
    def run(self, xin: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the model on a single input vector."""
        return jnp.dot(self.params.weight, xin) + self.params.bias


@register_pytree_node_class
class LinearModel2D(Model):
    """Linear model with a 2D matrix output, typically for scaling matrices."""

    def __init__(self, params: LinearParams):
        """Initialize the model with its parameters."""
        self.params = params

    def tree_flatten(self):
        """Flatten the model's parameters into a list of arrays (leaves)."""
        return (self.params,), {}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Unflatten parameter arrays back into a model instance."""
        return cls(children[0])

    @partial(jax.vmap, in_axes=(None, 0))
    def run(self, xin: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the model on a single input vector."""
        return jnp.einsum("ijk,k->ij", self.params.weight, xin) + self.params.bias


@register_pytree_node_class
class LinearScaleShiftModel(Model):
    """
    A model that computes a scale-and-shift correction.

    This model is designed for higher-fidelity nodes. It calculates:
    y = scale(x) @ parent_output + shift(x)

    Where 'scale' is a matrix generated by an edge model and 'shift' is a vector
    generated by a node model. Both depend on the primary input `xin`.
    """

    def __init__(self, edge_model: LinearModel2D, node_model: LinearModel):
        """Initialize the model with its edge and node sub-models."""
        self.edge_model = edge_model
        self.node_model = node_model

    def tree_flatten(self):
        """Flatten the model's parameters into a list of arrays (leaves)."""
        return (self.edge_model, self.node_model), {}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Unflatten parameter arrays back into a model instance."""
        return cls(*children)

    def run(self, xin: jnp.ndarray, parent_val: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate the model on a batch of inputs and parent values.

        Args:
            xin: The primary input array of shape (n_samples, n_features).
            parent_val: The output from parent nodes, shape (n_samples,
                n_parent_features).

        Returns
        -------
            The corrected output array.
        """
        edge_val = self.edge_model.run(xin)
        node_val = self.node_model.run(xin)
        # Einsum performs a batched matrix-vector product
        return jnp.einsum("bij,bj->bi", edge_val, parent_val) + node_val


# This defines the parameters for our MLP
MLPParams = list[LinearParams]


@register_pytree_node_class
class MLPModel(Model):
    """
    A Multi-Layer Perceptron (MLP) model.

    This model implements a standard feed-forward neural network.
    """

    # Note: Activation functions are static, so they are passed to __init__
    def __init__(self, params: MLPParams, activation: Callable = jnn.relu):
        """
        Initialize the MLP with its parameters and activation function.

        Args:
            params: A list of LinearParams, one for each layer.
            activation: The activation function to apply between hidden layers
                (e.g., jax.nn.relu, jax.nn.tanh).
        """
        self.params = params
        self.activation = activation

    def tree_flatten(self):
        """Flatten the model into its parameters and static data."""
        # The params are the dynamic children (leaves)
        children = self.params
        # The activation function is static auxiliary data
        aux_data = (self.activation,)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Unflatten the model from its parameters and static data."""
        activation = aux_data[0]
        params = children
        return cls(params, activation)

    # Note: We are not vmapping this run method because MLPs are often
    # written to handle batches of data by default.
    def run(self, xin: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate the MLP on a batch of input data.

        Args:
            xin: An array of shape (n_samples, n_features).

        Returns
        -------
            The model's output array.
        """
        x = xin
        # Apply layers sequentially
        for i, layer_params in enumerate(self.params):
            x = jnp.dot(x, layer_params.weight.T) + layer_params.bias
            # Apply activation to all but the last layer
            if i < len(self.params) - 1:
                x = self.activation(x)
        return x


@register_pytree_node_class
class MLPEnhancementModel(Model):
    """
    An enhancement model that uses an MLP to learn a correction.

    This model is designed for higher-fidelity nodes. It calculates:
    y = MLP(concat(x, parent_output))

    This allows it to learn complex, non-linear relationships between a lower
    fidelity model's output and the higher fidelity data.
    """

    def __init__(self, mlp: MLPModel):
        """Initialize the model with its internal MLP."""
        self.mlp = mlp

    def tree_flatten(self):
        """Flatten the model's internal MLP parameters."""
        return (self.mlp,), {}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Unflatten the model's internal MLP parameters."""
        return cls(children[0])

    def run(self, xin: jnp.ndarray, parent_val: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate the model on a batch of inputs and parent values.

        Args:
            xin: The primary input array of shape (n_samples, n_features).
            parent_val: The output from parent nodes, shape (n_samples,
                n_parent_features).

        Returns
        -------
            The corrected output array.
        """
        combined_input = jnp.concatenate([xin, parent_val], axis=-1)
        return self.mlp.run(combined_input)


# --- Polynomial Chaos Expansion (PCE) Models ---


def _hermite_poly_1d(x: float, degree: int) -> jnp.ndarray:
    """Evaluate 1D normalized Hermite polynomials at a point x."""
    H = jnp.zeros(degree + 1).at[0].set(1.0)
    # Handle degree 0 case
    if degree > 0:
        H = H.at[1].set(x)

    def body_fun(i, H_current):
        val = x * H_current[i - 1] - (i - 1) * H_current[i - 2]
        return H_current.at[i].set(val)

    H = lax.fori_loop(2, degree + 1, body_fun, H)
    # Cast integer array to float before calling lgamma
    factorial_arg = (jnp.arange(degree + 1) + 1).astype(jnp.float32)
    norm = jnp.sqrt(jax.lax.exp(jax.lax.lgamma(factorial_arg)))
    return H / norm


def _legendre_poly_1d(x: float, degree: int) -> jnp.ndarray:
    """Evaluate 1D Legendre polynomials at a point x."""
    P = jnp.zeros(degree + 1).at[0].set(1.0)
    if degree > 0:
        P = P.at[1].set(x)

    def body_fun(i, P_current):
        val = ((2 * i - 1) * x * P_current[i - 1] - (i - 1) * P_current[i - 2]) / i
        return P_current.at[i].set(val)

    return lax.fori_loop(2, degree + 1, body_fun, P)


def _compute_multi_indices(ndim: int, degree: int) -> jnp.ndarray:
    """Compute total-order multi-indices for PCE."""
    if degree == 0:
        return jnp.zeros((1, ndim), dtype=jnp.int32)

    alpha = np.zeros((int(comb(ndim + degree, degree)), ndim), dtype=np.int32)
    count = 0
    for k in range(degree + 1):
        for js in combinations_with_replacement(range(ndim), k):
            for i in range(ndim):
                alpha[count, i] = js.count(i)
            count += 1
    return jnp.array(alpha)


def build_poly_basis(
    x: jnp.ndarray, multi_indices: jnp.ndarray, poly_type: str, degree: int
) -> jnp.ndarray:
    """
    Construct the PCE basis matrix for a batch of inputs.

    Args:
        x: Input data of shape (n_samples, n_features).
        multi_indices: The multi-index set defining the basis terms.
        poly_type: "hermite" or "legendre".
        degree: The maximum polynomial order.

    Returns
    -------
        The PCE basis matrix of shape (n_samples, n_basis_terms).
    """
    n_samples, n_dim = x.shape

    if poly_type == "hermite":
        vmap_poly_eval = jax.vmap(jax.vmap(lambda val: _hermite_poly_1d(val, degree)))
    elif poly_type == "legendre":
        vmap_poly_eval = jax.vmap(jax.vmap(lambda val: _legendre_poly_1d(val, degree)))
    else:
        raise ValueError("poly_type must be 'hermite' or 'legendre'")

    # p_vals shape: (n_samples, n_dim, degree + 1)
    p_vals = vmap_poly_eval(x)

    # Gather the appropriate 1D polynomial values for each basis term
    # multi_indices shape: (n_basis_terms, n_dim) -> (b, d)
    # We want to select from p_vals (s, d, o) using indices of shape (b, d)
    # to produce an intermediate tensor of shape (s, b, d)
    # which we can then prod over the d axis.
    gathered = p_vals[:, jnp.arange(n_dim)[None, :], multi_indices]

    # Compute the tensor product by multiplying across the dimension axis
    # shape: (n_samples, n_basis_terms)
    return jnp.prod(gathered, axis=2)


@register_pytree_node_class
class PCEModel(Model):
    """A Polynomial Chaos Expansion model."""

    def __init__(
        self,
        params: LinearParams,
        poly_type: str,
        degree: int,
        multi_indices: jnp.ndarray,
    ):
        """
        Initialize the PCE model.

        Args:
            params: Linear coefficients for the polynomial basis.
            poly_type: "hermite" or "legendre".
            degree: Maximum polynomial order.
            multi_indices: Precomputed multi-index set.
        """
        self.params = params
        self.poly_type = poly_type
        self.degree = degree
        self.multi_indices = multi_indices

    def tree_flatten(self):
        """Flatten the model into dynamic parameters and static data."""
        children = (self.params,)
        aux_data = (self.poly_type, self.degree, self.multi_indices)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Unflatten the model from its parameters and static data."""
        poly_type, degree, multi_indices = aux_data
        params = children[0]
        return cls(params, poly_type, degree, multi_indices)

    def run(self, xin: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the PCE model on a batch of inputs."""
        basis_matrix = build_poly_basis(
            xin, self.multi_indices, self.poly_type, self.degree
        )
        return jnp.dot(basis_matrix, self.params.weight.T) + self.params.bias


@register_pytree_node_class
class PCEnhancementModel(Model):
    """An enhancement model using a PCE for the node and a linear model for the edge."""

    def __init__(self, edge_model: LinearModel, node_model: PCEModel):
        self.edge_model = edge_model
        self.node_model = node_model

    def tree_flatten(self):
        """Flatten the model's parameters into a list of arrays (leaves)."""
        return (self.edge_model, self.node_model), {}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Unflatten parameter arrays back into a model instance."""
        return cls(*children)

    def run(self, xin: jnp.ndarray, parent_val: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the model on a batch of inputs and parent values."""
        # Note: The edge model in the original torch code had a different shape.
        # This implementation uses a simpler batched linear model for the edge.
        edge_input = jnp.concatenate([xin, parent_val], axis=-1)
        edge_val = self.edge_model.run(edge_input)
        node_val = self.node_model.run(xin)
        return edge_val + node_val


# --- Loss Functions ---


@partial(jax.jit, static_argnums=(1,))
def mse_loss_graph(
    model: MFNetJax,
    nodes: tuple[Any, ...],
    x: jnp.ndarray,
    y: tuple[jnp.ndarray, ...],
) -> jnp.ndarray:
    """Calculate the total mean squared error across multiple graph nodes."""
    pred_nodes = model.run(nodes, x)
    # Breaking the generator expression into a list comprehension for line length
    losses = [
        jnp.mean((pred - true) ** 2) for pred, true in zip(pred_nodes, y, strict=False)
    ]
    return jnp.sum(jnp.array(losses))


@partial(jax.jit, static_argnums=(1,))
def resid_loss_graph(
    model: MFNetJax,
    nodes: tuple[Any, ...],
    x: jnp.ndarray,
    y: tuple[jnp.ndarray, ...],
) -> jnp.ndarray:
    """Calculate the flattened residual vector for least-squares solvers."""
    pred_nodes = model.run(nodes, x)
    residuals = [
        (pred - true).ravel() for pred, true in zip(pred_nodes, y, strict=False)
    ]
    return jnp.concatenate(residuals)


# --- Initializer Functions (JAX Best Practice) ---


def init_linear_params(key: jax.Array, dim_in: int, dim_out: int) -> LinearParams:
    """Initialize parameters for a LinearModel using explicit key splitting."""
    w_key, b_key = jax.random.split(key)
    weight = jax.random.normal(w_key, (dim_out, dim_in))
    bias = jax.random.normal(b_key, (dim_out,))
    return LinearParams(weight, bias)


def init_linear2d_params(
    key: jax.Array, d_out1: int, d_out2: int, d_in: int
) -> LinearParams:
    """Initialize parameters for a LinearModel2D."""
    w_key, b_key = jax.random.split(key)
    weight = jax.random.normal(w_key, (d_out1, d_out2, d_in))
    bias = jax.random.normal(b_key, (d_out1, d_out2))
    return LinearParams(weight, bias)


def init_mlp_params(key: jax.Array, layer_sizes: list[int]) -> MLPParams:
    """
    Initialize all parameters for an MLP.

    Args:
        key: A JAX random key.
        layer_sizes: A list of integers defining the network architecture,
            e.g., [dim_in, hidden1_dim, hidden2_dim, dim_out].

    Returns
    -------
        A list of LinearParams for the MLP.
    """
    params = []
    keys = jax.random.split(key, len(layer_sizes) - 1)
    for i, (dim_in, dim_out) in enumerate(
        zip(layer_sizes[:-1], layer_sizes[1:], strict=False)
    ):
        layer_params = init_linear_params(keys[i], dim_in, dim_out)
        params.append(layer_params)
    return params


def init_linear_scale_shift_model(
    key: jax.Array, d_in: int, d_parent: int, d_out: int
) -> LinearScaleShiftModel:
    """Initialize a complete LinearScaleShiftModel."""
    edge_key, node_key = jax.random.split(key)
    edge_params = init_linear2d_params(edge_key, d_out, d_parent, d_in)
    node_params = init_linear_params(node_key, d_in, d_out)
    return LinearScaleShiftModel(
        edge_model=LinearModel2D(edge_params),
        node_model=LinearModel(node_params),
    )


def init_mlp_enhancement_model(
    key: jax.Array, layer_sizes: list[int], activation: Callable = jnn.relu
) -> MLPEnhancementModel:
    """Initialize a complete MLPEnhancementModel."""
    mlp_params = init_mlp_params(key, layer_sizes)
    mlp_model = MLPModel(mlp_params, activation)
    return MLPEnhancementModel(mlp_model)


def init_pce_model(
    key: jax.Array, dim_in: int, dim_out: int, degree: int, poly_type: str = "hermite"
) -> PCEModel:
    """Initialize a PCEModel."""
    multi_indices = _compute_multi_indices(dim_in, degree)
    num_basis_terms = multi_indices.shape[0]
    pce_coeffs = init_linear_params(key, num_basis_terms, dim_out)
    return PCEModel(pce_coeffs, poly_type, degree, multi_indices)


def init_pc_enhancement_model(
    key: jax.Array,
    d_in: int,
    d_parent: int,
    d_out: int,
    degree: int,
    poly_type: str = "hermite",
) -> PCEnhancementModel:
    """Initialize a PCEnhancementModel."""
    edge_key, node_key = jax.random.split(key)
    edge_params = init_linear_params(edge_key, d_in + d_parent, d_out)
    edge_model = LinearModel(edge_params)
    node_model = init_pce_model(node_key, d_in, d_out, degree, poly_type)
    return PCEnhancementModel(edge_model, node_model)


# --- Graph Helpers ---


def make_graph_2gen(mod1: Model, mod2: Model) -> nx.DiGraph:
    """Create a simple two-node graph: 1 -> 2."""
    graph = nx.DiGraph()
    graph.add_node(1, func=mod1)
    graph.add_node(2, func=mod2)
    graph.add_edge(1, 2)
    return graph
