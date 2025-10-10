from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from typer.testing import CliRunner

from mfnets_surrogates.cli import app

runner = CliRunner()


@pytest.fixture
def cli_config_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for CLI configs and data."""
    config_dir = tmp_path / "cli_test_data"
    config_dir.mkdir()
    return config_dir


@pytest.fixture
def dummy_training_data_npz(cli_config_dir: Path) -> Path:
    """Create a dummy NPZ file for training data."""
    data_path = cli_config_dir / "training_data.npz"
    np.savez(
        data_path,
        x_train_1=np.random.rand(10, 2),
        y_train_1=np.random.rand(10, 1),
        x_train_2=np.random.rand(10, 2),
        y_train_2=np.random.rand(10, 1),
        x_train_3=np.random.rand(10, 2),
        y_train_3=np.random.rand(10, 1),
    )
    return data_path


@pytest.fixture
def dummy_prediction_data_npz(cli_config_dir: Path) -> Path:
    """Create a dummy NPZ file for prediction inputs."""
    data_path = cli_config_dir / "prediction_inputs.npz"
    np.savez(data_path, x_predict=np.random.rand(5, 2))
    return data_path


@pytest.fixture
def minimal_auto_config_path(
    cli_config_dir: Path, dummy_training_data_npz: Path
) -> Path:
    """Create a minimal, valid auto mode config file for testing."""
    config_content = f"""
mode: auto
alpha: 1.0
beta: 1.0
threshold: 0.1
sink_node: 3

base_models:
  1: {{type: "LinearModel", params: {{}}}}
  2: {{type: "LinearModel", params: {{}}}}
  3: {{type: "LinearModel", params: {{}}}}

leaf_model: {{type: "LinearModel", params: {{}}}}
edge_model: {{type: "LinearModel", params: {{}}}}

graph:
  nodes: [1, 2, 3]
  edges: []

models: {{}}

training:
  learning_rate: 0.001
  num_steps: 2

datasets:
  - name: "train_data"
    type: "training"
    data_path: "{dummy_training_data_npz}"
    nodes: [1, 2, 3]
"""
    config_path = cli_config_dir / "auto_config.yml"
    config_path.write_text(config_content)
    return config_path


def test_cli_run_invalid_config_path():
    """Test running the CLI with an invalid config path."""
    result = runner.invoke(app, ["run", "--config", "non_existent.yml"])
    assert result.exit_code != 0
    # Typer resolves the path, so we check for the resolved path in the error.
    expected_error_part = (
        f"File not found at '{Path('non_existent.yml').resolve()}'"
    )
    assert expected_error_part in result.stdout


def test_cli_run_auto_mode_missing_fields(cli_config_dir: Path):
    """Test auto mode with missing required fields."""
    # This config is valid for Pydantic, but will fail the manual check
    # for auto-mode fields because base_models, etc., are missing.
    config_content = """
mode: auto
alpha: 1.0
beta: 1.0
threshold: 0.1
graph:
  nodes: []
  edges: []
models: {}
training:
  learning_rate: 0.001
  num_steps: 1
datasets: []
"""
    config_path = cli_config_dir / "missing_fields.yml"
    config_path.write_text(config_content)

    result = runner.invoke(app, ["run", "--config", str(config_path)])
    assert result.exit_code != 0
    assert "Missing required fields for auto mode." in result.stdout


def test_cli_load_training_data_missing_keys(
    cli_config_dir: Path, dummy_prediction_data_npz: Path
):
    """Test _load_training_data with a NPZ missing expected keys."""
    # This config is valid, so parsing will succeed, but data loading will fail.
    config_content = f"""
mode: fixed
graph:
  nodes: [1]
  edges: []
models:
  1:
    type: "LinearModel"
    params: {{}}
training:
  learning_rate: 0.001
  num_steps: 1
datasets:
  - name: "bad_data"
    type: "training"
    data_path: "{dummy_prediction_data_npz}" # This NPZ has only x_predict
    nodes: [1]
"""
    config_path = cli_config_dir / "missing_data_keys.yml"
    config_path.write_text(config_content)

    result = runner.invoke(app, ["run", "--config", str(config_path)])
    assert result.exit_code != 0
    assert "Error: x_train_1 or y_train_1 not found" in result.stdout


def test_cli_load_training_data_node_not_in_graph_warning(
    cli_config_dir: Path, dummy_training_data_npz: Path
):
    """Test _load_training_data warns if node data is not in graph['nodes']."""
    # This config is valid, so parsing will succeed, but data loading will warn.
    config_content = f"""
mode: fixed
graph:
  nodes: [1, 2] # Node 3 is in NPZ but not here
  edges: []
models:
  1: {{type: "LinearModel", params: {{}}}}
  2: {{type: "LinearModel", params: {{}}}}
training:
  learning_rate: 0.001
  num_steps: 1
datasets:
  - name: "partial_data"
    type: "training"
    data_path: "{dummy_training_data_npz}"
    nodes: [1, 2, 3] # This lists node 3
"""
    config_path = cli_config_dir / "partial_nodes_config.yml"
    config_path.write_text(config_content)

    result = runner.invoke(app, ["run", "--config", str(config_path)])
    # The warning is printed, but the run continues and fails later because
    # `_build_mfnet_from_config` can't find a model for a node that isn't
    # in the 'models' section. We just check for the warning for now.
    assert "Warning: Training data found for node ID 3" in result.stdout


@patch("mfnets_surrogates.structure.AutoMFNet")  # Corrected patch target
def test_cli_run_auto_mode_sink_node_inference(
    mock_auto_mfnet_cls, minimal_auto_config_path: Path
):
    """Test auto mode infers sink_node correctly when not specified."""
    # Modify config to remove sink_node explicitly
    config_content = minimal_auto_config_path.read_text()
    config_content = config_content.replace("sink_node: 3", "")
    minimal_auto_config_path.write_text(config_content)

    runner.invoke(app, ["run", "--config", str(minimal_auto_config_path)])

    # Assert that AutoMFNet was instantiated with sink_node=3 (highest
    # fidelity)
    mock_auto_mfnet_cls.assert_called_once_with(
        sink_node=3, alpha=1.0, beta=1.0
    )
