"""
MFNetsSurrogates: A package for Multi-Fidelity Surrogate Networks.

This package provides tools for building, training, and evaluating multi-fidelity
surrogate models using computational graphs and JAX.
"""

# Import the core components from the JAX backend to make them
# accessible directly from the top-level package. This creates a clean
# and stable user-facing API.
from .net_jax import (
    # Basic models
    LinearModel,
    LinearModel2D,
    LinearParams,
    LinearScaleShiftModel,
    # Core graph class
    MFNetJax,
    MLPEnhancementModel,
    # MLP models
    MLPModel,
    MLPParams,
    # Base classes and parameter structures
    Model,
    # PCE models
    PCEModel,
    PCEnhancementModel,
    build_poly_basis,
    init_linear2d_params,
    # Initializer functions
    init_linear_params,
    init_linear_scale_shift_model,
    init_mlp_enhancement_model,
    init_mlp_params,
    init_pc_enhancement_model,
    init_pce_model,
    # Graph helpers
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
    "MLPParams",
    "MLPModel",
    "MLPEnhancementModel",
    "PCEModel",
    "PCEnhancementModel",
    "build_poly_basis",
    "mse_loss_graph",
    "resid_loss_graph",
    "init_linear_params",
    "init_linear2d_params",
    "init_linear_scale_shift_model",
    "init_mlp_params",
    "init_mlp_enhancement_model",
    "init_pce_model",
    "init_pc_enhancement_model",
    "make_graph_2gen",
]
