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
from typing import Any, NamedTuple

import jax
import jax.nn as jnn
import jax.numpy as jnp
import networkx as nx
from jax import tree_util
from jax.tree_util import register_pytree_node_class


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


def make_graph_2gen(mod1: Model, mod2: Model) -> nx.DiGraph:
    """Create a simple two-node graph: 1 -> 2."""
    graph = nx.DiGraph()
    graph.add_node(1, func=mod1)
    graph.add_node(2, func=mod2)
    graph.add_edge(1, 2)
    return graph
