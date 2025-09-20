# Welcome to MFNets-Surrogates

**MFNets-Surrogates** is a JAX-native library for building, training, and analyzing multi-fidelity surrogate models using flexible, differentiable graph structures.

This documentation provides a comprehensive guide to the library, including a tutorial on how to get started and a full API reference.

## Key Features

-   **JAX Core**: The entire library is built with JAX, utilizing `jax.jit` for compilation and `jax.grad` for automatic differentiation.
-   **End-to-End Differentiable**: The `MFNetJax` class is registered as a JAX PyTree, making the graph structure transparent to JAX's transformations.
-   **Flexible Graph Structures**: Uses NetworkX to define arbitrary directed acyclic graphs.
-   **Composable Models**: Includes a suite of built-in models like `LinearModel`, `MLPModel`, and `PCEModel`.

To get started, check out the **Tutorial**.
