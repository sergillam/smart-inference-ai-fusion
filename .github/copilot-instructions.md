# Copilot Instructions for Smart Inference AI Fusion

## Project Overview
- Modular framework for synthetic inference experiments and controlled perturbations in AI algorithms.
- Focus: robustness, variability, and failure testing in data, labels, and model parameters.
- Main entry: `main.py` (uses `PYTHONPATH=src` for module resolution).

## Architecture & Key Components
- `src/core/`: Base classes for `Experiment`, `Model`, and `Dataset`.
- `src/datasets/`: Factories and loaders for datasets (e.g., sklearn, CSV).
- `src/inference/`: Inference logic, including:
  - `engine/`: Orchestrators (e.g., `InferenceEngine`, `LabelRunner`, `ParameterInferenceEngine`).
  - `pipeline/`: Unified pipeline for applying all inference steps.
  - `transformations/`: Modular transformations for data (`data/`), labels (`label/`), and parameters (`params/`).
- `src/models/`: Model implementations (KNN, SVM, Tree, GaussianNB, etc.).
- `src/utils/`: Reporting, metrics, enums, and type definitions (see `types.py`).
- `experiments/`: Experiment scripts organized by dataset. Each subfolder (e.g., `digits/`, `wine/`) contains scripts for different models and a `run_all.py`.
- `datasets/`: Raw data files for CSV-based experiments.
- `results/`: Output logs and results (e.g., perturbed parameter logs).

## Developer Workflows
- **Build/Run:**
  - `make run` — Runs the main experiment pipeline.
  - `make run-debug` — (Recommended) Runs with `LOG_LEVEL=DEBUG` for detailed logging (see below).
  - `make lint` — Lints code with `pylint`.
  - `make test` — Runs unit tests with `pytest`.
- **Logging:**
  - Logging is configured via a utility (suggested: `src/utils/logging_config.py`).
  - Set `LOG_LEVEL` env var (`DEBUG`, `INFO`, etc.) to control verbosity.
  - All transformations and inference steps should log actions and parameters for traceability.
- **Adding Experiments:**
  - Add new scripts in `experiments/<dataset>/`.
  - Use the provided base classes and pipeline patterns.
  - Register new transformations in the appropriate `transformations/` submodule.

## Project Conventions
- Use `PYTHONPATH=src` for all CLI invocations to ensure correct imports.
- Prefer explicit configuration objects (e.g., `DataNoiseConfig`, `ParameterNoiseConfig`, `LabelNoiseConfig`) for experiment reproducibility.
- All experiment results and logs should be written to `results/`.
- Logging and reporting should use the utilities in `src/utils/`.
- Modular design: new models, datasets, or transformations should be added as new classes/files in their respective folders.

## Examples
- See `experiments/digits/gaussian_digits.py` for a full experiment using data, label, and parameter perturbations.
- See `src/inference/transformations/data/group_outlier_injection.py` for a custom data transformation pattern.

## External Dependencies
- Uses `scikit-learn`, `numpy`, and other standard ML/data libraries (see `requirements.txt`).
- All dependencies should be installed via `pip install -r requirements.txt`.

---

For any unclear conventions or missing documentation, check `README.md` or ask for clarification from project maintainers.
