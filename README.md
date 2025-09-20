# **Multi-Fidelity Surrogate Networks (MFNets)**

A JAX-native library for building, training, and analyzing multi-fidelity surrogate models using flexible, differentiable graph structures.

## **Introduction**

**MFNets-Surrogates** provides a powerful framework for fusing multiple sources of information. The core idea is to represent the relationships between different data fidelities (e.g., low vs. high-resolution simulations, or cheap vs. expensive experiments) as a **directed acyclic graph (DAG)**. The entire graph of models is end-to-end differentiable, allowing for efficient, gradient-based training of all model parameters simultaneously.  
This library is built entirely on **JAX**, ensuring high performance on modern hardware accelerators like GPUs and TPUs.

## **Key Features**

* **🚀 High-Performance JAX Core**: The entire library is built with JAX, leveraging jax.jit for compilation and jax.grad for optimization.  
* **🌳 End-to-End Differentiable**: The MFNetJax class is registered as a JAX **PyTree**, making the entire graph structure transparent to JAX's transformations.  
* **🕸️ Flexible Graph Structures**: Use the popular **NetworkX** library to define arbitrary directed acyclic graphs, giving you full control over the model architecture.  
* **🧩 Composable Models**: Includes a suite of built-in models that can be used as nodes in the graph, including LinearModel, MLPModel, and PCEModel (Polynomial Chaos Expansion). The powerful PCEScaleShiftModel is ideal for hierarchical relationships.  
* **🛠️ Modern Tooling**: Uses **Optax** for optimization, **Ruff** for linting and formatting, and **Pytest** for a comprehensive testing suite.

## **Installation**

You can install the package directly using pip. For a standard installation, run:  
pip install .

For development, it is recommended to install the package in editable mode with all development and testing dependencies:  
pip install \-e ".\[dev\]"

## **Quick Start Example**

Here is a complete example of how to define, train, and evaluate a simple two-fidelity hierarchical model (1 \-\> 2).  
import jax  
import jax.numpy as jnp  
import networkx as nx  
import optax  
from jax import tree\_util

from mfnets\_surrogates import (  
    MFNetJax,  
    LinearModel,  
    LinearScaleShiftModel,  
    init\_linear\_params,  
    init\_linear\_scale\_shift\_model,  
    mse\_loss\_graph,  
)

def main():  
    """A complete example of building, training, and running an MFNet."""  
    key \= jax.random.PRNGKey(0)  
    d\_in, d1\_out, d2\_out \= 2, 2, 3  
    key, true\_key, train\_key, data\_key \= jax.random.split(key, 4\)

    \# 1\. Define a "true" model and generate some training data  
    true\_m1 \= LinearModel(init\_linear\_params(true\_key, d\_in, d1\_out))  
    true\_m2 \= init\_linear\_scale\_shift\_model(true\_key, d\_in, d1\_out, d2\_out)  
    true\_graph \= nx.DiGraph(\[(1, 2)\])  
    true\_graph.add\_node(1, func=true\_m1)  
    true\_graph.add\_node(2, func=true\_m2)  
    true\_mfnet \= MFNetJax(true\_graph)

    x\_train \= jax.random.normal(data\_key, (100, d\_in))  
    y\_train \= true\_mfnet.run((1, 2), x\_train)

    \# 2\. Create the MFNetJax model to be trained  
    train\_m1 \= LinearModel(init\_linear\_params(train\_key, d\_in, d1\_out))  
    train\_m2 \= init\_linear\_scale\_shift\_model(train\_key, d\_in, d1\_out, d2\_out)  
    train\_graph\_struct \= nx.DiGraph(\[(1, 2)\])  
    train\_graph\_struct.add\_node(1, func=train\_m1)  
    train\_graph\_struct.add\_node(2, func=train\_m2)  
    mfnet\_to\_train \= MFNetJax(train\_graph\_struct)

    \# 3\. Set up the training loop with Optax  
    params, treedef \= tree\_util.tree\_flatten(mfnet\_to\_train)  
    optimizer \= optax.adam(learning\_rate=1e-3)  
    opt\_state \= optimizer.init(params)

    def loss\_fn(p, x, y):  
        """Loss function that un-flattens parameters inside."""  
        model \= treedef.unflatten(p)  
        return mse\_loss\_graph(model, nodes=(1, 2), x=x, y=y)

    @jax.jit  
    def step(p, opt\_s, x, y):  
        """A single JIT-compiled training step."""  
        loss\_val, grads \= jax.value\_and\_grad(loss\_fn)(p, x, y)  
        updates, opt\_s \= optimizer.update(grads, opt\_s)  
        p \= optax.apply\_updates(p, updates)  
        return p, opt\_s, loss\_val

    print("--- Starting Training \---")  
    initial\_loss \= loss\_fn(params, x\_train, y\_train)  
    print(f"Initial Loss: {initial\_loss:.4f}")

    for i in range(2000):  
        params, opt\_state, loss \= step(params, opt\_state, x\_train, y\_train)  
        if (i \+ 1\) % 500 \== 0:  
            print(f"  Step {i+1}, Loss: {loss:.4f}")

    \# 4\. Make predictions with the final trained model  
    mfnet\_fitted \= treedef.unflatten(params)  
    x\_test \= jax.random.normal(data\_key, (10, d\_in))  
    predictions \= mfnet\_fitted.run((1, 2), x\_test)

    print("\\n--- Predictions from Node 2 (Highest Fidelity) \---")  
    print(predictions\[1\])

if \_\_name\_\_ \== "\_\_main\_\_":  
    main()

## **Development & Usage**

The Makefile provides several convenient commands for development:

* make install-dev: Installs the package in editable mode with all development dependencies.  
* make lint: Formats the code with Ruff and automatically fixes linting errors.  
* make test: Runs the complete test suite using pytest.  
* make run-example: Runs one of the example scripts in the examples/ directory.

Use the help command to see all available options:  
make help

## **Citation**

If you use this code in your research, please cite the original paper:

* Gorodetsky, Alex A., John D. Jakeman, and Gianluca Geraci. "MFNets: data efficient all-at-once learning of multifidelity surrogates as directed networks of information sources." *Computational Mechanics* 68.4 (2021): 741-758.

## **License**

This project is licensed under the **MIT License**.

