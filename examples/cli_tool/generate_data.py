"""Generates dummy data for the CLI tool example."""

import os

import numpy as np


def main():
    """Generate and save training and prediction data."""
    # Create directories if they don't exist
    os.makedirs("examples/cli_tool", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # --- Generate Training Data ---
    # This simulates having data for a 3-fidelity problem
    np.random.seed(42)
    x_train_all = np.linspace(-2, 2, 100).reshape(-1, 1)

    # True functions (similar to other examples)
    y_true_1 = 0.5 * np.cos(np.pi * x_train_all)
    y_true_2 = y_true_1**2 + 0.1 * np.sin(np.pi * x_train_all)
    y_true_3 = y_true_2 + 0.1 * np.cos(2 * np.pi * x_train_all)

    # Save to a single .npz file
    training_path = "examples/cli_tool/training_data.npz"
    np.savez(
        training_path,
        x_train_1=x_train_all,
        y_train_1=y_true_1,
        x_train_2=x_train_all,
        y_train_2=y_true_2,
        x_train_3=x_train_all,
        y_train_3=y_true_3,
    )
    print(f"Saved training data to {training_path}")

    # --- Generate Prediction Inputs ---
    x_predict = np.linspace(-2.5, 2.5, 200).reshape(-1, 1)
    prediction_path = "examples/cli_tool/prediction_inputs.npz"
    np.savez(prediction_path, x_predict=x_predict)
    print(f"Saved prediction inputs to {prediction_path}")


if __name__ == "__main__":
    main()
