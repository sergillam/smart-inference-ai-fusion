# ===========================
# smart-inference-ai-fusion - Makefile
# ===========================
.SHELLFLAGS := -eu -o pipefail -c
SHELL := /bin/bash
.ONESHELL:

# -------- Variables --------
PKG              ?= smart_inference_ai_fusion
SRC_DIR          ?= $(PKG)
TESTS_DIR        ?= tests
VENV             ?= .venv
LOGS_DIR         ?= logs
RESULTS_DIR      ?= results

# Prioritizes python3.10, falls back to python3
PYTHON_310       := $(shell command -v python3.10 2>/dev/null || true)
PYTHON           ?= $(if $(PYTHON_310),$(PYTHON_310),python3)

# Venv binaries
PY               := $(VENV)/bin/python
PIP              := $(VENV)/bin/pip

# Allows passing arguments: make run ARGS="--dataset digits"
ARGS            ?=

# Configurable flags
LINT_PATHS       ?= $(SRC_DIR) $(TESTS_DIR)
PYLINT_ARGS      ?=
PYDOCSTYLE_ARGS  ?= --convention=google
PYTEST_ARGS      ?=

.DEFAULT_GOAL := help
.PHONY: help venv ensure-venv-py310 install install-dev uninstall run debug \
        lint style check test clean clean-pyc clean-all clean-venv \
        freeze compile-reqs compile-reqs-dev ci print-% \
        build publish publish-prod deploy check-clean check-version default \
        format check-format clean-outputs

# -------- Help (Self-documenting) --------
help: ## Shows this help message
	@echo "======================================"
	@echo "  smart-inference-ai-fusion - Make"
	@echo "======================================"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-25s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "🔹 Tip: You don't need to 'activate' the venv to use 'make' — targets use .venv automatically."
	@echo "🔹 Ex.: make run EXP=$(PKG).experiments.digits ARGS=\"--seed 42\""


# -------- Environment Setup --------
venv: ## Creates and prepares the virtualenv using $(PYTHON)
	@if [ ! -d "$(VENV)" ]; then \
		echo "🐍 Creating venv with $(PYTHON) in $(VENV)"; \
		"$(PYTHON)" -m venv "$(VENV)"; \
	fi
	@echo "✅ venv ready in $(VENV)"
	@$(PIP) --version >/dev/null

ensure-venv-py310: ## Checks if the venv is running on Python 3.10
	@v="$$( $(PY) -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' )"; \
	if [ "$$v" != "3.10" ]; then \
		echo "❌ The venv is not Python 3.10 (current: $$v). Recreate it with: python3.10 -m venv .venv"; \
		exit 2; \
	fi

# -------- Installation --------
install: venv ## Installs runtime dependencies from pyproject.toml
	@echo "📦 Installing package for RUNTIME..."
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e .
	@echo "✅ Runtime installation complete."

install-dev: venv ## Installs all dependencies for development (including dev extras)
	@echo "📦 Installing package for DEVELOPMENT..."
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e .[dev]
	@echo "✅ Development installation complete."

uninstall: venv ## Removes the installed package from the venv
	@echo "🧹 Removing installed package..."
	-$(PIP) uninstall -y $(PKG) || true
	@echo "✅ Removed."

# -------- Execution --------
run: venv ensure-venv-py310 ## Runs experiments. Use EXP=<module|package> and ARGS="<options>"
	@if [ -z "$(EXP)" ]; then \
		echo "🚀 Running ALL experiments via auto-discovery…"; \
		$(PY) -m $(PKG).experiments $(ARGS); \
	else \
		# Se EXP já começa com smart_inference_ai_fusion.experiments., usa direto; senão, monta o caminho completo
		if echo "$(EXP)" | grep -q '^smart_inference_ai_fusion\.experiments\.'; then \
			EXP_MODULE="$(EXP)"; \
		else \
			EXP_MODULE="smart_inference_ai_fusion.experiments.$(EXP)"; \
		fi; \
		echo "🚀 Running specific target EXP='$$EXP_MODULE' with ARGS='$(ARGS)'…"; \
		$(PY) scripts/run_experiment.py "$$EXP_MODULE" $(ARGS); \
	fi
	@echo "✅ Done."

debug: venv ensure-venv-py310 ## Runs the main experiment orchestrator in DEBUG mode
	@echo "🐞 Running in DEBUG mode (LOG_LEVEL=DEBUG)…"
	LOG_LEVEL=DEBUG $(PY) -m $(PKG).experiments $(ARGS)
	@echo "✅ Done (debug)."

# -------- Code Quality & Testing --------
format: install-dev ## Formats code with black and isort
	@echo "🎨 Formatting code..."
	$(PY) -m isort $(SRC_DIR) $(TESTS_DIR)
	$(PY) -m black $(SRC_DIR) $(TESTS_DIR)
	@echo "✅ Code formatted."

check-format: install-dev ## Checks code formatting without making changes
	@echo "🔍 Checking formatting (black and isort)..."
	$(PY) -m black --check $(SRC_DIR) $(TESTS_DIR)
	$(PY) -m isort --check-only $(SRC_DIR) $(TESTS_DIR)
	@echo "✅ Formatting OK."

lint: install-dev ## Lints code with pylint
	@echo "🔍 Running pylint..."
	$(PY) -m pylint $(PYLINT_ARGS) $(LINT_PATHS)

style: install-dev ## Checks docstrings (Google style)
	@echo "📝 Checking docstrings..."
	$(PY) -m pydocstyle $(PYDOCSTYLE_ARGS) $(LINT_PATHS)

check: check-format lint style ## Runs all code quality checks (format, lint, style)
	@echo "✅ All code quality checks passed."

test: install-dev ## Runs unit tests with pytest
	@echo "🧪 Running unit tests..."
	$(PY) -m pytest -E $(PYTEST_ARGS) $(TESTS_DIR)

# -------- Reproducibility --------
compile-reqs: venv ## Generates requirements.txt from pyproject.toml
	@echo "🧰 Generating requirements.txt..."
	$(PIP) install -U pip-tools
	$(VENV)/bin/pip-compile --upgrade --output-file=requirements.txt pyproject.toml
	@echo "✅ requirements.txt updated."

compile-reqs-dev: venv ## Generates requirements-dev.txt (includes [dev] extras)
	@echo "🧰 Generating requirements-dev.txt..."
	$(PIP) install -U pip-tools
	$(VENV)/bin/pip-compile --extra=dev --upgrade --output-file=requirements-dev.txt pyproject.toml
	@echo "✅ requirements-dev.txt updated."

freeze: venv ## Generates requirements-freeze.txt (a snapshot of the current venv)
	@echo "🧊 Generating requirements-freeze.txt..."
	$(PIP) freeze | sort > requirements-freeze.txt
	@echo "✅ Saved to requirements-freeze.txt"

# -------- Maintenance --------
clean: ## Removes build artifacts and Python cache files
	rm -rf build/ dist/ *.egg-info .pytest_cache .coverage htmlcov
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "🧼 Build and cache files cleaned."

clean-outputs: ## WARNING: Deletes all generated logs and results
	@echo "🔥 Deleting all contents of $(LOGS_DIR)/ and $(RESULTS_DIR)/..."
	rm -rf $(LOGS_DIR)/* $(RESULTS_DIR)/* 2>/dev/null || true
	@echo "✅ Logs and results contents have been cleared (directories preserved)."

clean-pyc: ## Removes only Python bytecode cache files
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f -name "*.py[co]" -delete
	@echo "🧼 pyc files cleaned."

clean-all: clean clean-outputs clean-pyc ## Runs all clean tasks, including logs and results
	@echo "✅ All clean tasks completed."

clean-venv: ## Removes the .venv virtual environment directory
	rm -rf $(VENV)
	@echo "🧹 venv removed."

# -------- CI/CD Pipeline --------
ci: install-dev check test run ## Runs the complete CI pipeline (quality checks, tests, and run)
	@echo "✅ CI pipeline completed successfully."

# -------- Publishing --------
build: venv ## Builds the wheel and sdist packages into ./dist
	@echo "📦 Building wheel + sdist…"
	$(PIP) install --upgrade build
	$(PY) -m build
	@echo "✅ Artifacts ready in ./dist"

publish: build ## Publishes the package to the TestPyPI repository
	@echo "🚀 Publishing to TestPyPI…"
	$(PIP) install --upgrade twine
	$(PY) -m twine upload --repository testpypi dist/*
	@echo "✅ Published to TestPyPI"

publish-prod: build ## Publishes the package to the official PyPI repository
	@echo "🚀 Publishing to PyPI…"
	$(PIP) install --upgrade twine
	$(PY) -m twine upload dist/*
	@echo "✅ Published to PyPI"

deploy: publish ## Alias for the 'publish' command

# -------- Formal Verification --------
verify-install: venv ## Install formal verification dependencies
	@echo "🔧 Installing formal verification dependencies..."
	$(PIP) install z3-solver
	@echo "✅ Formal verification dependencies installed."

verify-test: venv ## Test formal verification system
	@echo "🧪 Testing formal verification system..."
	PYTHONPATH=$(PWD) $(PY) scripts/test_formal_verification.py

verify-z3: verify-install ## Test Z3 capabilities specifically
	@echo "🧠 Testing Z3 SMT solver capabilities..."
	PYTHONPATH=$(PWD) $(PY) scripts/test_formal_verification.py

verify-list: venv ## List available formal verifiers
	@echo "📋 Listing formal verification plugins..."
	PYTHONPATH=$(PWD) $(PY) -c "\
from smart_inference_ai_fusion.verification import list_verifiers; \
verifiers = list_verifiers(); \
print('Available Formal Verifiers:'); \
[print(f'  {name}: {\"🟢 Available\" if info[\"available\"] else \"🔴 Unavailable\"}, {\"✅ Enabled\" if info[\"enabled\"] else \"❌ Disabled\"}') for name, info in verifiers.items()]; \
"

# --- Internal Utility ---
# Utility for debugging make variables (e.g., make print-PKG)
print-%:
	@echo '$*=$($*)'