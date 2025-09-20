"""Core JAX implementation for Multi-Fidelity Surrogate Networks (MFNets).

This module defines the main `MFNetJax` class, which acts as a JAX-compatible
PyTree container for a graph of surrogate models. It also provides a set of
basic linear models that can be used as nodes within the graph.

The design philosophy is to make the entire graph structure differentiable and
optimizable with JAX-based tools.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from itertools import combinations_with_replacement
from typing import Any, NamedTuple, Self, cast

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
    """A JAX-compatible multi-fidelity network represented by a directed graph.

    This class wraps a `networkx.DiGraph` where each node contains a callable
    "func" that represents a surrogate model. It is registered as a JAX PyTree,
    allowing its parameters to be transparently handled by JAX transformations
    like `jax.grad` and `jax.jit`.

    Attributes
    ----------
        graph (nx.DiGraph): The graphical representation of the MF network.
        eval_order (list): A topologically sorted list of nodes for execution.
        parents (dict): A mapping from each node to its direct predecessors.
        ancestors (dict): A mapping from each node to all its ancestors.
    """

    def __init__(self, graph: nx.DiGraph) -> None:
        """Initialize the multifidelity network.

        Args:
            graph: A networkx.DiGraph where each node's data dictionary must
                   contain a "func" key pointing to a JAX-compatible model.
        """
        self.graph = graph
        self.eval_order = list(nx.topological_sort(self.graph))
        self.parents = {n: sorted(self.graph.predecessors(n)) for n in self.eval_order}
        self.ancestors = {n: set(nx.ancestors(self.graph, n)) for n in self.eval_order}

    def tree_flatten(self) -> tuple[list[Any], tuple[Any, ...]]:
        """Flatten the MFNetJax into its dynamic leaves and static auxiliary data.

        This method is required for JAX PyTree registration.
        """
        leaves: list[Any] = []
        treedefs: list[tree_util.PyTreeDef] = []

        for node in self.eval_order:
            func = self.graph.nodes[node]["func"]
            f_leaves, f_treedef = tree_util.tree_flatten(func)
            leaves.extend(f_leaves)
            treedefs.append(f_treedef)

        nodes = tuple(self.eval_order)
        edges = tuple((p, n) for n in self.eval_order for p in self.parents[n])
        aux_data = (nodes, edges, tuple(treedefs))
        return leaves, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: tuple[Any, ...], children: list[Any]) -> Self:
        """Reconstruct an MFNetJax from static data and dynamic leaves.

        This method is required for JAX PyTree registration.
        """
        nodes, edges, treedefs = aux_data

        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)

        model = cls(graph)

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
        """Evaluate the graph to compute outputs for the specified target nodes."""
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
                    val = func.run(xinput)
                else:
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

    def tree_flatten(self) -> tuple[list[Any], dict[str, Any]]:
        """Flatten the model's parameters into a list of arrays (leaves)."""
        raise NotImplementedError

    @classmethod
    def tree_unflatten(cls, aux_data: dict[str, Any], children: list[Any]) -> Self:
        """Unflatten parameter arrays back into a model instance."""
        raise NotImplementedError


@register_pytree_node_class
class LinearModel(Model):
    """A simple linear model: y = x @ W.T + b."""

    def __init__(self, params: LinearParams) -> None:
        """Initialize the model with its parameters."""
        self.params = params

    def tree_flatten(self) -> tuple[list[LinearParams], dict[str, Any]]:
        """Flatten the model's parameters into a list of arrays (leaves)."""
        return [self.params], {}

    @classmethod
    def tree_unflatten(
        cls, aux_data: dict[str, Any], children: list[LinearParams]
    ) -> Self:
        """Unflatten parameter arrays back into a model instance."""
        return cls(children[0])

    def run(self, xin: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the model on a batch of input data."""
        return xin @ self.params.weight.T + self.params.bias


@register_pytree_node_class
class LinearModel2D(Model):
    """Linear model with a 2D matrix output, typically for scaling matrices."""

    def __init__(self, params: LinearParams) -> None:
        """Initialize the model with its parameters."""
        self.params = params

    def tree_flatten(self) -> tuple[list[LinearParams], dict[str, Any]]:
        """Flatten the model's parameters into a list of arrays (leaves)."""
        return [self.params], {}

    @classmethod
    def tree_unflatten(
        cls, aux_data: dict[str, Any], children: list[LinearParams]
    ) -> Self:
        """Unflatten parameter arrays back into a model instance."""
        return cls(children[0])

    def run(self, xin: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the model on a batch of input data."""
        return jnp.einsum("opi,si->sop", self.params.weight, xin) + self.params.bias


@register_pytree_node_class
class LinearScaleShiftModel(Model):
    """A model that computes a scale-and-shift correction."""

    def __init__(self, edge_model: LinearModel2D, node_model: LinearModel) -> None:
        """Initialize the model with its edge and node sub-models."""
        self.edge_model = edge_model
        self.node_model = node_model

    def tree_flatten(self) -> tuple[list[Model], dict[str, Any]]:
        """Flatten the model's parameters into a list of arrays (leaves)."""
        return [self.edge_model, self.node_model], {}

    @classmethod
    def tree_unflatten(cls, aux_data: dict[str, Any], children: list[Model]) -> Self:
        """Unflatten parameter arrays back into a model instance."""
        return cls(children[0], children[1])  # type: ignore

    def run(self, xin: jnp.ndarray, parent_val: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the model: y = scale(x) @ parent_val + shift(x)."""
        edge_val = self.edge_model.run(xin)
        node_val = self.node_model.run(xin)
        return jnp.einsum("sop,sp->so", edge_val, parent_val) + node_val


# --- MLP Models ---
MLPParams = list[LinearParams]


@register_pytree_node_class
class MLPModel(Model):
    """A Multi-Layer Perceptron (MLP) model."""

    def __init__(
        self,
        params: MLPParams,
        activation: Callable[[jnp.ndarray], jnp.ndarray] = jnn.relu,
    ) -> None:
        """Initialize the MLP with its parameters and activation function."""
        self.params = params
        self.activation = activation

    def tree_flatten(self) -> tuple[list[Any], dict[str, Any]]:
        """Flatten the model into its parameters and static data."""
        return self.params, {"activation": self.activation}

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: dict[str, Any],
        children: list[Any],
    ) -> Self:
        """Unflatten the model from its parameters and static data."""
        return cls(children, aux_data["activation"])

    def run(self, xin: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the MLP on a batch of input data."""
        x = xin
        for i, layer_params in enumerate(self.params):
            x = x @ layer_params.weight.T + layer_params.bias
            if i < len(self.params) - 1:
                x = self.activation(x)
        return x


@register_pytree_node_class
class MLPEnhancementModel(Model):
    """An MLP that enhances a low-fidelity input with a high-fidelity one."""

    def __init__(self, mlp_model: MLPModel) -> None:
        """Initialize the model with its internal MLP."""
        self.mlp_model = mlp_model

    def tree_flatten(self) -> tuple[list[MLPModel], dict[str, Any]]:
        """Flatten the model's parameters into a list of arrays (leaves)."""
        return [self.mlp_model], {}

    @classmethod
    def tree_unflatten(cls, aux_data: dict[str, Any], children: list[MLPModel]) -> Self:
        """Unflatten parameter arrays back into a model instance."""
        return cls(children[0])

    def run(self, xin: jnp.ndarray, parent_val: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the model on a batch of inputs and parent values."""
        combined_input = jnp.concatenate([xin, parent_val], axis=-1)
        return self.mlp_model.run(combined_input)


# --- Polynomial Chaos Expansion (PCE) Models ---


def _hermite_poly_1d(x: float, degree: int) -> jnp.ndarray:
    """Evaluate 1D normalized Hermite polynomials at a point x."""
    if degree == 0:
        return jnp.ones(1)
    H = jnp.zeros(degree + 1).at[0].set(1.0).at[1].set(x)

    def body_fun(i: int, H_current: jnp.ndarray) -> jnp.ndarray:
        val = x * H_current[i - 1] - (i - 1) * H_current[i - 2]
        return H_current.at[i].set(val)

    H = lax.fori_loop(2, degree + 1, body_fun, H)
    factorial_vals = jnp.exp(jax.lax.lgamma(jnp.arange(degree + 1) + 1.0))
    return cast(jnp.ndarray, H / jnp.sqrt(factorial_vals))


def _legendre_poly_1d(x: float, degree: int) -> jnp.ndarray:
    """Evaluate 1D Legendre polynomials at a point x."""
    if degree == 0:
        return jnp.ones(1)
    P = jnp.zeros(degree + 1).at[0].set(1.0).at[1].set(x)

    def body_fun(i: int, P_current: jnp.ndarray) -> jnp.ndarray:
        val = ((2 * i - 1) * x * P_current[i - 1] - (i - 1) * P_current[i - 2]) / i
        return P_current.at[i].set(val)

    return cast(jnp.ndarray, lax.fori_loop(2, degree + 1, body_fun, P))


def _compute_multi_indices(ndim: int, degree: int) -> np.ndarray:
    """Compute total-order multi-indices for PCE."""
    if degree == 0:
        return np.zeros((1, ndim), dtype=np.int32)
    num_terms = int(comb(ndim + degree, degree))
    alpha = np.zeros((num_terms, ndim), dtype=np.int32)
    count = 0
    for k in range(degree + 1):
        for js in combinations_with_replacement(range(ndim), k):
            for i in range(ndim):
                alpha[count, i] = js.count(i)
            count += 1
    return alpha


def build_poly_basis(
    x: jnp.ndarray,
    multi_indices: jnp.ndarray,
    poly_type: str,
    degree: int,
) -> jnp.ndarray:
    """Construct the PCE basis matrix for a batch of inputs."""
    poly_1d_fn: Callable[[float, int], jnp.ndarray]
    if poly_type == "hermite":
        poly_1d_fn = _hermite_poly_1d
    elif poly_type == "legendre":
        poly_1d_fn = _legendre_poly_1d
    else:
        raise ValueError("poly_type must be 'hermite' or 'legendre'")

    p_vals = jax.vmap(jax.vmap(lambda val: poly_1d_fn(val, degree)))(x)

    def build_for_sample(
        p_vals_sample: jnp.ndarray, multi_indices_local: jnp.ndarray
    ) -> jnp.ndarray:
        n_dim = p_vals_sample.shape[0]
        gathered = p_vals_sample[jnp.arange(n_dim)[:, None], multi_indices_local.T]
        return jnp.prod(gathered, axis=0)

    return jax.vmap(build_for_sample, in_axes=(0, None))(p_vals, multi_indices)


@register_pytree_node_class
class PCEModel(Model):
    """A Polynomial Chaos Expansion model that outputs a vector."""

    def __init__(
        self,
        params: LinearParams,
        poly_type: str,
        degree: int,
        multi_indices: jnp.ndarray,
    ) -> None:
        """Initialize the PCE model."""
        self.params = params
        self.poly_type = poly_type
        self.degree = degree
        self.multi_indices = multi_indices

    def tree_flatten(self) -> tuple[list[LinearParams], dict[str, Any]]:
        """Flatten the model into dynamic parameters and static data."""
        return [self.params], {
            "poly_type": self.poly_type,
            "degree": self.degree,
            "multi_indices": self.multi_indices,
        }

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: dict[str, Any],
        children: list[LinearParams],
    ) -> Self:
        """Unflatten the model from its parameters and static data."""
        return cls(children[0], **aux_data)

    def run(self, xin: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the PCE model on a batch of inputs."""
        basis_matrix = build_poly_basis(
            xin, self.multi_indices, self.poly_type, self.degree
        )
        return basis_matrix @ self.params.weight.T + self.params.bias


@register_pytree_node_class
class PCEModel2D(Model):
    """A Polynomial Chaos Expansion model that outputs a matrix."""

    def __init__(
        self,
        params: LinearParams,
        poly_type: str,
        degree: int,
        multi_indices: jnp.ndarray,
    ) -> None:
        """Initialize the PCE model."""
        self.params = params
        self.poly_type = poly_type
        self.degree = degree
        self.multi_indices = multi_indices

    def tree_flatten(self) -> tuple[list[LinearParams], dict[str, Any]]:
        """Flatten the model into dynamic parameters and static data."""
        return [self.params], {
            "poly_type": self.poly_type,
            "degree": self.degree,
            "multi_indices": self.multi_indices,
        }

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: dict[str, Any],
        children: list[LinearParams],
    ) -> Self:
        """Unflatten the model from its parameters and static data."""
        return cls(children[0], **aux_data)

    def run(self, xin: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the PCE model on a batch of inputs."""
        basis_matrix = build_poly_basis(
            xin, self.multi_indices, self.poly_type, self.degree
        )
        edge_val = jnp.einsum("opb,sb->sop", self.params.weight, basis_matrix)
        return edge_val + self.params.bias


@register_pytree_node_class
class PCEAdditiveModel(Model):
    """An additive enhancement model using PCE and a linear model."""

    def __init__(self, edge_model: LinearModel, node_model: PCEModel) -> None:
        """Initialize the model with its edge and node sub-models."""
        self.edge_model = edge_model
        self.node_model = node_model

    def tree_flatten(self) -> tuple[list[Model], dict[str, Any]]:
        """Flatten the model's parameters into a list of arrays (leaves)."""
        return [self.edge_model, self.node_model], {}

    @classmethod
    def tree_unflatten(cls, aux_data: dict[str, Any], children: list[Model]) -> Self:
        """Unflatten parameter arrays back into a model instance."""
        return cls(children[0], children[1])  # type: ignore

    def run(self, xin: jnp.ndarray, parent_val: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the model on a batch of inputs and parent values."""
        edge_input = jnp.concatenate([xin, parent_val], axis=-1)
        edge_val = self.edge_model.run(edge_input)
        node_val = self.node_model.run(xin)
        return edge_val + node_val


@register_pytree_node_class
class PCEScaleShiftModel(Model):
    """An enhancement model using PCEs for both scale and shift terms."""

    def __init__(self, edge_model: PCEModel2D, node_model: PCEModel) -> None:
        """Initialize the model with its edge and node sub-models."""
        self.edge_model = edge_model
        self.node_model = node_model

    def tree_flatten(self) -> tuple[list[Model], dict[str, Any]]:
        """Flatten the model's parameters into a list of arrays (leaves)."""
        return [self.edge_model, self.node_model], {}

    @classmethod
    def tree_unflatten(cls, aux_data: dict[str, Any], children: list[Model]) -> Self:
        """Unflatten parameter arrays back into a model instance."""
        return cls(children[0], children[1])  # type: ignore

    def run(self, xin: jnp.ndarray, parent_val: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the model: y = PCE_edge(x) @ parent_val + PCE_node(x)."""
        edge_val = self.edge_model.run(xin)
        node_val = self.node_model.run(xin)
        correction = jnp.einsum("sop,sp->so", edge_val, parent_val)
        return correction + node_val


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


# --- Initializer Functions ---


def init_linear_params(key: jax.Array, dim_in: int, dim_out: int) -> LinearParams:
    """Initialize parameters for a LinearModel."""
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


def init_mlp_params(key: jax.Array, layer_sizes: list[int]) -> MLPParams:
    """Initialize all parameters for an MLP."""
    params = []
    keys = jax.random.split(key, len(layer_sizes) - 1)
    layer_pairs = zip(layer_sizes[:-1], layer_sizes[1:], strict=False)
    for i, (dim_in, dim_out) in enumerate(layer_pairs):
        layer_params = init_linear_params(keys[i], dim_in, dim_out)
        params.append(layer_params)
    return params


def init_mlp_enhancement_model(
    key: jax.Array,
    layer_sizes: list[int],
    activation: Callable[[jnp.ndarray], jnp.ndarray] = jnn.relu,
) -> MLPEnhancementModel:
    """Initialize a complete MLPEnhancementModel."""
    mlp_params = init_mlp_params(key, layer_sizes)
    mlp_model = MLPModel(mlp_params, activation)
    return MLPEnhancementModel(mlp_model)


def init_pce_model(
    key: jax.Array,
    dim_in: int,
    dim_out: int,
    degree: int,
    poly_type: str = "hermite",
) -> PCEModel:
    """Initialize a PCEModel."""
    multi_indices = _compute_multi_indices(dim_in, degree)
    num_basis_terms = multi_indices.shape[0]
    pce_coeffs = init_linear_params(key, num_basis_terms, dim_out)
    return PCEModel(pce_coeffs, poly_type, degree, jnp.asarray(multi_indices))


def init_pce_model_2d(
    key: jax.Array,
    d_in: int,
    d_out1: int,
    d_out2: int,
    degree: int,
    poly_type: str = "hermite",
) -> PCEModel2D:
    """Initialize a PCEModel that outputs a 2D matrix."""
    multi_indices = _compute_multi_indices(d_in, degree)
    num_basis_terms = multi_indices.shape[0]
    w_key, b_key = jax.random.split(key)
    weight = jax.random.normal(w_key, (d_out1, d_out2, num_basis_terms))
    bias = jax.random.normal(b_key, (d_out1, d_out2))
    return PCEModel2D(
        LinearParams(weight, bias),
        poly_type,
        degree,
        jnp.asarray(multi_indices),
    )


def init_pc_additive_model(
    key: jax.Array,
    d_in: int,
    d_parent: int,
    d_out: int,
    degree: int,
    poly_type: str = "hermite",
) -> PCEAdditiveModel:
    """Initialize a PCEAdditiveModel."""
    edge_key, node_key = jax.random.split(key)
    edge_params = init_linear_params(edge_key, d_in + d_parent, d_out)
    edge_model = LinearModel(edge_params)
    node_model = init_pce_model(node_key, d_in, d_out, degree, poly_type)
    return PCEAdditiveModel(edge_model, node_model)


def init_pce_scale_shift_model(
    key: jax.Array,
    d_in: int,
    d_parent: int,
    d_out: int,
    degree: int,
    poly_type: str = "hermite",
) -> PCEScaleShiftModel:
    """Initialize a PCEScaleShiftModel."""
    edge_key, node_key = jax.random.split(key)
    edge_model = init_pce_model_2d(edge_key, d_in, d_out, d_parent, degree, poly_type)
    node_model = init_pce_model(node_key, d_in, d_out, degree, poly_type)
    return PCEScaleShiftModel(edge_model, node_model)


# --- Graph Helpers ---


def make_graph_2gen(mod1: Model, mod2: Model) -> nx.DiGraph:
    """Create a simple two-node graph: 1 -> 2."""
    graph = nx.DiGraph()
    graph.add_node(1, func=mod1)
    graph.add_node(2, func=mod2)
    graph.add_edge(1, 2)
    return graph
