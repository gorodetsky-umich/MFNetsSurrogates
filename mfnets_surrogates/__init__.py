"""
MFNetsSurrogates: A package for Multi-Fidelity Surrogate Networks.

This package provides tools for building, training, and evaluating multi-fidelity
surrogate models using computational graphs and JAX.
"""

# Import the core components from the JAX backend to make them
# accessible directly from the top-level package. This creates a clean
# and stable user-facing API.
from .net_jax import (
    # Specific model implementations
    LinearModel,
    LinearModel2D,
    LinearParams,
    LinearScaleShiftModel,
    # Core graph class
    MFNetJax,
    MLPEnhancementModel,
    MLPModel,
    MLPParams,
    # Base model class and parameter structures
    Model,
    init_linear2d_params,
    # Helper functions for initialization and graph creation
    init_linear_params,
    init_linear_scale_shift_model,
    init_mlp_enhancement_model,
    init_mlp_params,
    make_graph_2gen,
    # Loss functions
    mse_loss_graph,
    resid_loss_graph,
)

# Define the public API of the package.
# This is a list of names that will be imported when a user does
# `from mfnets_surrogates import *`. It also helps static analysis tools.
__all__ = [
    "MFNetJax",
    "Model",
    "LinearParams",
    "LinearModel",
    "LinearModel2D",
    "LinearScaleShiftModel",
    "MLPModel",
    "mse_loss_graph",
    "resid_loss_graph",
    "init_linear_params",
    "init_linear2d_params",
    "init_linear_scale_shift_model",
    "init_mlp_params",
    "make_graph_2gen",
]
