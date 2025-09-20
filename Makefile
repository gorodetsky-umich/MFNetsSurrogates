# Makefile for MFNetsSurrogates project

.DEFAULT_GOAL := help
.PHONY: help install install-dev lint format check check-format type-check test clean ci run-example docs

# ==============================================================================
# Installation
# ==============================================================================

install: ## Install the package in standard mode
	@echo "--> Installing the package in standard mode..."
	python -m pip install .

install-dev: ## Install the package in editable mode with all dev dependencies
	@echo "--> Installing the package in editable mode with all development dependencies..."
	python -m pip install -e ".[dev]"

# ==============================================================================
# Code Quality & Testing
# ==============================================================================

lint: format check ## Run the formatter and linter with auto-fix
	@echo "--> Linting and formatting complete."

format: ## Format code with Ruff and apply automatic fixes
	@echo "--> Formatting code with Ruff..."
	ruff format .
	@echo "--> Applying automatic fixes with Ruff..."
	ruff check . --fix

check: ## Check for linting errors with Ruff
	@echo "--> Checking for linting errors with Ruff..."
	ruff check .

check-format: ## Check for formatting issues without changing files
	@echo "--> Checking for formatting issues with Ruff..."
	ruff format . --check

type-check: ## Run static type checking with mypy
	@echo "--> Running static type checking with mypy..."
	mypy mfnets_surrogates

test: ## Run all tests with pytest
	@echo "--> Running all tests with pytest..."
	pytest

# ==============================================================================
# Documentation
# ==============================================================================
docs: ## Build and serve the documentation locally
	@echo "--> Building and serving documentation at http://127.0.0.1:8000"
	@mkdocs serve

# ==============================================================================
# Examples
# ==============================================================================
run-example: ## Run the basic training example script
	@echo "--> Running basic training example..."
	@python examples/basic_training.py

run-mlp-graph: ## Run an example showing MLP nodes with multiple graphs
	@echo "--> Running MLP multiple graph example..."
	@python examples/mlp_graph.py

run-pce-graph: ## Run an example showing PCE nodes with multiple graphs on MLP truth
	@echo "--> Running PCE graph example..."
	@python examples/pce_graphs_with_mlp_truth.py


# ==============================================================================
# CI & Cleanup
# ==============================================================================

ci: check-format check type-check test ## Run all checks for Continuous Integration
	@echo "--> CI checks passed successfully."

clean: ## Clean up build artifacts, caches, and temp files
	@echo "--> Cleaning up build artifacts and caches..."
	rm -rf build/ dist/ .eggs/ *.egg-info/ site/
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +

# ==============================================================================
# Help
# ==============================================================================

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'


