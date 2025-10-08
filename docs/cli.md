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
