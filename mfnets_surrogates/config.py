"""Pydantic models for the CLI configuration file."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class ModelParams(BaseModel):
    """Defines the type and parameters for a single node's model."""

    type: Literal["PCEModel", "MLPModel", "PCEScaleShiftModel"]
    params: dict[str, Any] = Field(default_factory=dict)


class TrainingParams(BaseModel):
    """Defines parameters for the training process."""

    learning_rate: float = 0.001
    num_steps: int = 5000


class DataSet(BaseModel):
    """Defines a set of input/output files for a task."""

    name: str = Field(description="Unique name for this dataset.")
    type: Literal["training", "prediction"]
    data_path: str = Field(description="Path to the .npz data file.")
    # The node in the graph this data corresponds to.
    # For training, this is a list of all nodes with data in the file.
    # For prediction, this is the single node to get predictions from.
    nodes: list[int | str]
    output_path: str | None = None


class Config(BaseModel):
    """The root model for the YAML configuration file."""

    # ------------------------------------------------------------------
    # Two-stage Auto-MFNet specific fields
    # ------------------------------------------------------------------

    mode: Literal["fixed", "auto"] = Field(
        default="fixed",
        description=(
            'Run-mode selector: "fixed" executes single-stage training, while '
            '"auto" triggers the two-stage structure-learning pipeline.'
        ),
    )
    alpha: float = Field(
        1.0,
        description="Acyclicity penalty weight for structure learning "
        "(auto mode only).",
    )
    beta: float = Field(
        1.0,
        description="L1 sparsity penalty weight for structure learning "
        "(auto mode only).",
    )
    threshold: float = Field(
        0.1,
        description="Edge-pruning threshold |W_ij| ≤ τ during DAG extraction "
        "(auto mode only).",
    )

    # Optional model templates used exclusively in auto mode
    base_models: dict[int | str, ModelParams] | None = Field(
        default=None,
        description="Per-node δ-model definitions for Stage-1 (auto mode).",
    )
    leaf_model: ModelParams | None = Field(
        default=None,
        description="Factory template for leaf nodes in Stage-2 (auto mode).",
    )
    edge_model: ModelParams | None = Field(
        default=None,
        description="Factory template for edge/enhancement nodes in Stage-2 "
        "(auto mode).",
    )

    graph: dict[str, list[Any]] = Field(
        description="Defines the graph structure with nodes and edges."
    )
    models: dict[int | str, ModelParams] = Field(
        description="Maps node IDs to their model definitions."
    )
    training: TrainingParams
    datasets: list[DataSet] = Field(
        description="A list of datasets for training and prediction tasks."
    )
