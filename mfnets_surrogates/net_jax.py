"""Core JAX implementation for Multi-Fidelity Surrogate Networks (MFNets).

This module defines the main `MFNetJax` class, which acts as a JAX-compatible
PyTree container for a graph of surrogate models. It also provides a set of
basic linear models that can be used as nodes within the graph.

The design philosophy is to make the entire graph structure differentiable and
optimizable with JAX-based tools.
"""

import sys
from collections.abc import Callable
from functools import partial
from itertools import combinations_with_replacement
from typing import Any, NamedTuple, cast

# Conditionally import Self for backward compatibility with Python < 3.11
if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

import jax
import jax.lax as lax
import jax.nn as jnn
import jax.numpy as jnp
import networkx as nx
import numpy as np
import optax
from jax import tree_util
from jax.tree_util import register_pytree_node_class
from optax import OptState
from scipy.special import comb
from tqdm import trange  # progress bar

# --- Loss Functions ---


@partial(jax.jit, static_argnums=(1,))
def mse_loss_graph(
    model: "MFNetJax",
    nodes: tuple[Any, ...],
    x_list: list[jnp.ndarray],
    y_list: list[jnp.ndarray],
) -> jnp.ndarray:
    """Calculate total MSE across nodes with potentially different inputs."""
    losses = []
    for i, node in enumerate(nodes):
        x_i = x_list[i]
        y_i = y_list[i]
        # Run the graph to get the prediction for the i-th node on its data
        (pred_i,) = model.run((node,), x_i)
        losses.append(jnp.mean((pred_i - y_i) ** 2))
    return jnp.sum(jnp.array(losses))


@partial(jax.jit, static_argnums=(1,))
def resid_loss_graph(
    model: "MFNetJax",
    nodes: tuple[Any, ...],
    x_list: list[jnp.ndarray],
    y_list: list[jnp.ndarray],
) -> jnp.ndarray:
    """Calculate the flattened residual vector for least-squares solvers."""
    residuals = []
    for i, node in enumerate(nodes):
        x_i = x_list[i]
        y_i = y_list[i]
        (pred_i,) = model.run((node,), x_i)
        residuals.append((pred_i - y_i).ravel())
    return jnp.concatenate(residuals)


# --- Models ---


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
        self.parents = {
            n: sorted(self.graph.predecessors(n)) for n in self.eval_order
        }
        self.ancestors = {
            n: set(nx.ancestors(self.graph, n)) for n in self.eval_order
        }

    def tree_flatten(self) -> tuple[list[Any], tuple[Any, ...]]:
        """Flatten the MFNetJax into its dynamic leaves and static data.

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
    def tree_unflatten(
        cls, aux_data: tuple[Any, ...], children: list[Any]
    ) -> Self:
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
        """Evaluate the graph for the specified target nodes."""
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

    def fit(
        self,
        train_data: list[tuple[jnp.ndarray, jnp.ndarray]],
        n_iters: int = 10000,
        learning_rate: float = 1e-3,
        loss_fn: Callable = mse_loss_graph,
        verbose: bool = True,
        log_every: int = 100,
        history_callback: list | None = None,
    ) -> Self:
        """Train the network parameters using an Adam optimizer.

        Args:
            train_data: A list of (x, y) tuples for each model fidelity.
            n_iters: The number of optimization iterations.
            learning_rate: The learning rate for the Adam optimizer.
            loss_fn: The loss function to use for training.
            verbose: If True, display a progress bar.
            log_every: The interval at which to record the loss.
            history_callback: An optional list to append loss values to.

        Returns
        -------
            The trained MFNetJax instance.
        """
        target_nodes = tuple(
            self.eval_order[i]
            for i, data in enumerate(train_data)
            if data is not None
        )
        valid_data = [data for data in train_data if data is not None]
        x_data = [d[0] for d in valid_data]
        y_data = [d[1] for d in valid_data]
        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(self)

        @jax.jit
        def train_step(
            model: "MFNetJax",
            opt_state: OptState,
            x_list: list[jnp.ndarray],
            y_list: list[jnp.ndarray],
        ) -> tuple["MFNetJax", OptState, jnp.ndarray]:
            loss, grads = jax.value_and_grad(loss_fn)(
                model, target_nodes, x_list, y_list
            )
            # Filter out non-optimizable gradients
            grad_leaves, aux_data = tree_util.tree_flatten(grads)
            nodes, edges, treedefs = aux_data
            filtered_leaves = []
            idx = 0
            for node, tdef in zip(nodes, treedefs):
                nleaf = tdef.num_leaves
                sub = grad_leaves[idx:idx+nleaf]
                if model.graph.nodes[node]["func"].optimizable:
                    filtered_leaves.extend(sub)
                else:
                    filtered_leaves.extend([jnp.zeros_like(x) for x in sub])
                idx += nleaf
            filtered_grads = tree_util.tree_unflatten(aux_data, filtered_leaves)

            updates, new_opt_state = optimizer.update(
                filtered_grads, opt_state, model
            )
            new_model = optax.apply_updates(model, updates)
            return new_model, new_opt_state, loss

        model = self

        if verbose:
            pbar = trange(n_iters, desc="Training Loss", ascii=True)
            for i in pbar:
                model, opt_state, loss_val = train_step(
                    model, opt_state, x_data, y_data
                )
                if i % log_every == 0:
                    if history_callback is not None:
                        history_callback.append(loss_val)
                    pbar.set_postfix(loss=f"{loss_val:.4e}")
        else:
            for i in range(n_iters):
                model, opt_state, loss_val = train_step(
                    model, opt_state, x_data, y_data
                )
                if i % log_every == 0:
                    if history_callback is not None:
                        history_callback.append(loss_val)

        for node in self.eval_order:
            self.graph.nodes[node]["func"] = model.graph.nodes[node]["func"]

        return self


# --- Model Definitions ---


class LinearParams(NamedTuple):
    """Parameters for a linear model."""

    weight: jnp.ndarray
    bias: jnp.ndarray


class Model:
    """Base class for all models to ensure they are registered as PyTrees."""

    def __init__(self):
        self.optimizable = True

    def set_optimizable(self, optimizable: bool) -> None:
        """Set whether the model's parameters are optimizable."""
        self.optimizable = optimizable

    def tree_flatten(self) -> tuple[list[Any], dict[str, Any]]:
        """Flatten the model's parameters into a list of arrays (leaves)."""
        raise NotImplementedError

    @classmethod
    def tree_unflatten(
        cls, aux_data: dict[str, Any], children: list[Any]
    ) -> Self:
        """Unflatten parameter arrays back into a model instance."""
        raise NotImplementedError


@register_pytree_node_class
class LinearModel(Model):
    """A simple linear model: y = x @ W.T + b."""

    def __init__(self, params: LinearParams) -> None:
        """Initialize the model with its parameters."""
        super().__init__()
        self.params = params

    def tree_flatten(self) -> tuple[list[Any], dict[str, Any]]:
        """Flatten the model's parameters into a list of arrays (leaves)."""
        return [self.params], {"optimizable": self.optimizable}

    @classmethod
    def tree_unflatten(
        cls, aux_data: dict[str, Any], children: list[Any]
    ) -> Self:
        """Unflatten parameter arrays back into a model instance."""
        instance = cls(children[0])  # type: ignore
        instance.optimizable = aux_data["optimizable"]
        return instance

    def run(self, xin: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the model on a batch of input data."""
        return xin @ self.params.weight.T + self.params.bias


@register_pytree_node_class
class LinearModel2D(Model):
    """Linear model with a 2D matrix output for scaling matrices."""

    def __init__(self, params: LinearParams) -> None:
        """Initialize the model with its parameters."""
        super().__init__()
        self.params = params

    def tree_flatten(self) -> tuple[list[Any], dict[str, Any]]:
        """Flatten the model's parameters into a list of arrays (leaves)."""
        return [self.params], {"optimizable": self.optimizable}

    @classmethod
    def tree_unflatten(
        cls, aux_data: dict[str, Any], children: list[Any]
    ) -> Self:
        """Unflatten parameter arrays back into a model instance."""
        instance = cls(children[0])  # type: ignore
        instance.optimizable = aux_data["optimizable"]
        return instance

    def run(self, xin: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the model on a batch of input data."""
        return (
            jnp.einsum("opi,si->sop", self.params.weight, xin)
            + self.params.bias
        )


@register_pytree_node_class
class LinearScaleShiftModel(Model):
    """A model that computes a scale-and-shift correction."""

    def __init__(self, edge_model: LinearModel2D, node_model: LinearModel) -> None:
        """Initialize the model with its edge and node sub-models."""
        super().__init__()
        self.edge_model = edge_model
        self.node_model = node_model

    def tree_flatten(self) -> tuple[list[Any], dict[str, Any]]:
        """Flatten the model's parameters into a list of arrays (leaves)."""
        return [self.edge_model, self.node_model], {
            "optimizable": self.optimizable
        }

    @classmethod
    def tree_unflatten(
        cls, aux_data: dict[str, Any], children: list[Any]
    ) -> Self:
        """Unflatten parameter arrays back into a model instance."""
        instance = cls(children[0], children[1])  # type: ignore
        instance.optimizable = aux_data["optimizable"]
        return instance

    def set_optimizable(self, optimizable: bool) -> None:
        """Set whether the model's parameters are optimizable."""
        super().set_optimizable(optimizable)
        self.edge_model.set_optimizable(optimizable)
        self.node_model.set_optimizable(optimizable)


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
        activation: Callable[[jnp.ndarray], jnp.ndarray] = jnn.gelu,
    ) -> None:
        """Initialize the MLP with its parameters and activation function."""
        super().__init__()
        self.params = params
        self.activation = activation

    def tree_flatten(self) -> tuple[list[Any], dict[str, Any]]:
        """Flatten the model into its parameters and static data."""
        return self.params, {
            "activation": self.activation,
            "optimizable": self.optimizable,
        }

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: dict[str, Any],
        children: list[Any],
    ) -> Self:
        """Unflatten the model from its parameters and static data."""
        instance = cls(children, aux_data["activation"])
        instance.optimizable = aux_data["optimizable"]
        return instance

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
        super().__init__()
        self.mlp_model = mlp_model

    def tree_flatten(self) -> tuple[list[Any], dict[str, Any]]:
        """Flatten the model's parameters into a list of arrays (leaves)."""
        return [self.mlp_model], {"optimizable": self.optimizable}

    @classmethod
    def tree_unflatten(
        cls, aux_data: dict[str, Any], children: list[Any]
    ) -> Self:
        """Unflatten parameter arrays back into a model instance."""
        instance = cls(children[0])  # type: ignore
        instance.optimizable = aux_data["optimizable"]
        return instance

    def set_optimizable(self, optimizable: bool) -> None:
        """Set whether the model's parameters are optimizable."""
        super().set_optimizable(optimizable)
        self.mlp_model.set_optimizable(optimizable)

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
        val = (
            (2 * i - 1) * x * P_current[i - 1] - (i - 1) * P_current[i - 2]
        ) / i
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
        gathered = p_vals_sample[
            jnp.arange(n_dim)[:, None], multi_indices_local.T
        ]
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
        super().__init__()
        self.params = params
        self.poly_type = poly_type
        self.degree = degree
        self.multi_indices = multi_indices

    def tree_flatten(self) -> tuple[list[Any], dict[str, Any]]:
        """Flatten the model into dynamic parameters and static data."""
        return [self.params], {
            "poly_type": self.poly_type,
            "degree": self.degree,
            "multi_indices": self.multi_indices,
            "optimizable": self.optimizable,
        }

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: dict[str, Any],
        children: list[Any],
    ) -> Self:
        """Unflatten the model from its parameters and static data."""
        opt_flag = aux_data.pop("optimizable")
        instance = cls(
            children[0],
            poly_type=aux_data["poly_type"],
            degree=aux_data["degree"],
            multi_indices=aux_data["multi_indices"],
        )
        instance.optimizable = opt_flag
        return instance

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
        super().__init__()
        self.params = params
        self.poly_type = poly_type
        self.degree = degree
        self.multi_indices = multi_indices

    def tree_flatten(self) -> tuple[list[Any], dict[str, Any]]:
        """Flatten the model into dynamic parameters and static data."""
        return [self.params], {
            "poly_type": self.poly_type,
            "degree": self.degree,
            "multi_indices": self.multi_indices,
            "optimizable": self.optimizable,
        }

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: dict[str, Any],
        children: list[Any],
    ) -> Self:
        """Unflatten the model from its parameters and static data."""
        opt_flag = aux_data.pop("optimizable")
        instance = cls(
            children[0],
            poly_type=aux_data["poly_type"],
            degree=aux_data["degree"],
            multi_indices=aux_data["multi_indices"],
        )
        instance.optimizable = opt_flag
        return instance

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
        super().__init__()
        self.edge_model = edge_model
        self.node_model = node_model

    def tree_flatten(self) -> tuple[list[Any], dict[str, Any]]:
        """Flatten the model's parameters into a list of arrays (leaves)."""
        return [self.edge_model, self.node_model], {
            "optimizable": self.optimizable
        }

    @classmethod
    def tree_unflatten(
        cls, aux_data: dict[str, Any], children: list[Any]
    ) -> Self:
        """Unflatten parameter arrays back into a model instance."""
        instance = cls(children[0], children[1])  # type: ignore
        instance.optimizable = aux_data["optimizable"]
        return instance

    def set_optimizable(self, optimizable: bool) -> None:
        """Set whether the model's parameters are optimizable."""
        super().set_optimizable(optimizable)
        self.edge_model.set_optimizable(optimizable)
        self.node_model.set_optimizable(optimizable)

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
        super().__init__()
        self.edge_model = edge_model
        self.node_model = node_model

    def tree_flatten(self) -> tuple[list[Any], dict[str, Any]]:
        """Flatten the model's parameters into a list of arrays (leaves)."""
        return [self.edge_model, self.node_model], {
            "optimizable": self.optimizable
        }

    @classmethod
    def tree_unflatten(
        cls, aux_data: dict[str, Any], children: list[Any]
    ) -> Self:
        """Unflatten parameter arrays back into a model instance."""
        instance = cls(children[0], children[1])  # type: ignore
        instance.optimizable = aux_data["optimizable"]
        return instance

    def set_optimizable(self, optimizable: bool) -> None:
        """Set whether the model's parameters are optimizable."""
        super().set_optimizable(optimizable)
        self.edge_model.set_optimizable(optimizable)
        self.node_model.set_optimizable(optimizable)

    def run(self, xin: jnp.ndarray, parent_val: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the model: y = PCE_edge(x) @ parent_val + PCE_node(x)."""
        edge_val = self.edge_model.run(xin)
        node_val = self.node_model.run(xin)
        correction = jnp.einsum("sop,sp->so", edge_val, parent_val)
        return correction + node_val


# --- Initializer Functions ---


def init_linear_params(key: jax.Array, d_in: int, d_out: int) -> LinearParams:
    """Initialize parameters for a LinearModel."""
    w_key, b_key = jax.random.split(key)
    weight = jax.random.normal(w_key, (d_out, d_in))
    bias = jax.random.normal(b_key, (d_out,))
    return LinearParams(weight, bias)


def init_linear_model(key: jax.Array, d_in: int, d_out: int) -> LinearModel:
    """Initialize a complete LinearModel.

    This function creates the parameters and instantiates the LinearModel
    class, making it discoverable by the application.

    Args:
        key: A JAX random key.
        d_in: The dimension of the input features.
        d_out: The dimension of the output features.

    Returns
    -------
        An instance of LinearModel.
    """
    params = init_linear_params(key, d_in, d_out)
    return LinearModel(params)


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


def init_mlp_model(
    key: jax.Array,
    layer_sizes: list[int],
    activation: Callable[[jnp.ndarray], jnp.ndarray] = jnn.gelu,
) -> MLPModel:
    """Initialize a complete MLPModel.

    This is a wrapper that creates the parameters and instantiates the
    MLPModel class, making it discoverable by the app.

    Args:
        key: A JAX random key.
        layer_sizes: A list defining the network structure,
                      e.g., [d_in, hidden1, d_out].
        activation: The activation function for hidden layers.

    Returns
    -------
        An instance of MLPModel.
    """
    params = init_mlp_params(key, layer_sizes)
    return MLPModel(params, activation=activation)


def init_mlp_enhancement_model(
    key: jax.Array,
    layer_sizes: list[int],
    activation: Callable[[jnp.ndarray], jnp.ndarray] = jnn.gelu,
) -> MLPEnhancementModel:
    """Initialize a complete MLPEnhancementModel."""
    mlp_params = init_mlp_params(key, layer_sizes)
    mlp_model = MLPModel(mlp_params, activation)
    return MLPEnhancementModel(mlp_model)


def init_pce_model(
    key: jax.Array,
    d_in: int,
    d_out: int,
    degree: int,
    poly_type: str = "hermite",
) -> PCEModel:
    """Initialize a PCEModel."""
    multi_indices = _compute_multi_indices(d_in, degree)
    num_basis_terms = multi_indices.shape[0]
    pce_coeffs = init_linear_params(key, num_basis_terms, d_out)
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
    edge_model = init_pce_model_2d(
        edge_key, d_in, d_out, d_parent, degree, poly_type
    )
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
