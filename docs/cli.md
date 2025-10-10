# Command-Line Interface (CLI)

The mfnets-surrogates package includes a powerful command-line tool, mfnets-run, that allows you to define, train, and run an entire multi-fidelity experiment from a single configuration file without writing any Python code.

## Installation

To use the CLI, you must install the package with the [cli] extra:

```
pip install "mfnets-surrogates[cli]"
```

## Usage

The main command is run, which takes a path to a YAML configuration file.

```
mfnets-run run --config /path/to/your/config.yml
```

### Listing Available Models

To see a list of all models you can use in your configuration file, run the list-models command:

```
mfnets-run list-models
```

This will output a table of available model types and the parameters you can configure for each.

## Configuration File (`config.yml`)

The YAML configuration file is the core of the CLI. It defines the graph structure, the models at each node, the training parameters, and the data to be used.

Here is an example with explanations:
```
# The graph structure. 'nodes' is a list of all node IDs, and 'edges'
# defines the connections between them.
graph:
  nodes: [1, 2, 3]
  edges:
    - [1, 2]
    - [2, 3]

# Maps each node ID to a model definition. The 'type' must match a name
# from `mfnets-run list-models`. 'params' is a dictionary of arguments
# passed to the model's initializer.
models:
  1: # Low-fidelity model
    type: "PCEModel"
    params:
      degree: 2
      poly_type: "hermite"
  2: # Mid-fidelity enhancement model
    type: "PCEScaleShiftModel"
    params:
      degree: 2
  3: # High-fidelity enhancement model
    type: "PCEScaleShiftModel"
    params:
      degree: 3

# Parameters for the training process.
training:
  learning_rate: 0.005
  num_steps: 8000

# A list of datasets to use for training or prediction.
datasets:
  # This dataset will be used for training the entire graph.
  - name: "training_set"
    type: "training"
    # Path to an .npz file. It must contain arrays named 'x_train_NODE' and
    # 'y_train_NODE' for each node listed below.
    data_path: "examples/cli_tool/training_data.npz"
    nodes: [1, 2, 3]

  # This dataset defines a prediction task.
  - name: "prediction_grid"
    type: "prediction"
    # Path to an .npz file containing an 'x_predict' array.
    data_path: "examples/cli_tool/prediction_inputs.npz"
    nodes: [3] # Generate predictions for node 3.
    # The path where the results will be saved as an .npz file.
    output_path: "results/predictions.npz"
```

## Auto Mode Configuration

When `mode` is set to `"auto"`, `mfnets-run` executes the two-stage `AutoMFNet` pipeline. This mode requires additional configuration fields to define the structure learning process and the models used in each stage.

Here is a complete example of an `auto_config.yml` file:

```yaml
# Set mode to 'auto' to enable the two-stage structure learning pipeline.
mode: auto

# Alpha controls the weight of the acyclicity penalty in Stage 1.
# Higher values encourage a more strictly acyclic graph.
alpha: 1.0

# Beta controls the weight of the L1 sparsity penalty in Stage 1.
# Higher values encourage a sparser graph (fewer edges).
beta: 1.0

# Threshold for pruning edges from the learned adjacency matrix in Stage 2.
# Edges with |W_ij| <= threshold are removed.
threshold: 0.1

# Optional: Node ID to force as the sink node (no outgoing edges) in Stage 1.
# If None, the highest fidelity node with training data will be selected as sink.
sink_node: 3 # Node 3 is highest fidelity, explicitly set as sink.

# Definitions for base models (δ_j) used in Stage 1 of structure learning.
# Keys are node IDs. These models are typically simple (e.g., LinearModel, PCEModel).
base_models:
  1: # Base model for node 1
    type: "PCEModel"
    params:
      degree: 3
      poly_type: hermite
  2: # Base model for node 2
    type: "MLPModel"
    params: {} # Empty params dictionary for default MLP settings
  3: # Base model for node 3
    type: "MLPModel"
    params: {}

# Template for leaf models created in Stage 2 (nodes with no parents in the DAG).
# These are typically more expressive than base models.
leaf_model:
  type: "PCEModel"
  params:
    degree: 3
    poly_type: hermite

# Template for edge/enhancement models created in Stage 2 (nodes with parents).
# These models take both primary input `x` and parent outputs as input.
edge_model:
  type: "MLPModel"
  params: {}

# The 'graph' and 'models' sections here define the *initial configuration*
# or expected structure. In 'auto' mode, 'models' are ignored during Stage 1
# but the 'graph.nodes' are used to define the nodes participating in
# structure learning.
graph:
  nodes: [1, 2, 3] # Nodes included in the structure learning
  edges: [[1, 2], [2, 3]] # Initial guess (can be empty or full)

# In 'auto' mode, this section is used to initialize the MFNetJax model
# *after* structure learning, with the discovered graph and more complex models.
# However, it's often more convenient to let `leaf_model` and `edge_model`
# define the Stage 2 models, and omit this section.
models: {} # Can be empty if leaf_model/edge_model are fully specified

# Training parameters for Stage 2 (parameter learning) of AutoMFNet.
training:
  learning_rate: 0.001
  num_steps: 5000

# Datasets are defined the same way as in 'fixed' mode.
datasets:
  - name: "train_data"
    type: "training"
    data_path: "examples/cli_tool/training_data.npz"
    nodes: [1, 2, 3] # Nodes for which training data is available

  - name: "predict_hf"
    type: "prediction"
    data_path: "examples/cli_tool/prediction_inputs.npz"
    nodes: [3]
    output_path: "results/auto_hf_predictions.npz"
```
