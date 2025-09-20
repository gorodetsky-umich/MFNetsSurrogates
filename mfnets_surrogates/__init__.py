"""A JAX-based library for multi-fidelity surrogate modeling."""

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
    PCEAdditiveModel,
    # PCE models
    PCEModel,
    PCEModel2D,
    PCEScaleShiftModel,
    build_poly_basis,
    init_linear2d_params,
    # Initializer functions
    init_linear_params,
    init_linear_scale_shift_model,
    init_mlp_enhancement_model,
    init_mlp_params,
    init_pc_additive_model,
    init_pce_model,
    init_pce_model_2d,
    init_pce_scale_shift_model,
    # Graph helpers
    make_graph_2gen,
    # Loss functions
    mse_loss_graph,
    resid_loss_graph,
)

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
    "PCEModel2D",
    "PCEAdditiveModel",
    "PCEScaleShiftModel",
    "build_poly_basis",
    "mse_loss_graph",
    "resid_loss_graph",
    "init_linear_params",
    "init_linear2d_params",
    "init_linear_scale_shift_model",
    "init_mlp_params",
    "init_mlp_enhancement_model",
    "init_pce_model",
    "init_pce_model_2d",
    "init_pc_additive_model",
    "init_pce_scale_shift_model",
    "make_graph_2gen",
]
