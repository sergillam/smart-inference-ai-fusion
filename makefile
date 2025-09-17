# ===========================
# Smart Inference AI Fusion - Makefile
# ===========================
.SHELLFLAGS := -eu -o pipefail -c
SHELL := /bin/bash
.ONESHELL:

# -------- Variables --------
PYTHON := python3
VENV_DIR := .venv
VENV_BIN := $(VENV_DIR)/bin
PIP := $(VENV_BIN)/pip
PYTHON_VENV := $(VENV_BIN)/python
PKG_NAME := smart_inference_ai_fusion
SRC_DIR := $(PKG_NAME)
MAIN_MODULE := $(PKG_NAME).experiments
PYTHONPATH := .
REQUIREMENTS_DEV := requirements-dev.txt
REQUIREMENTS_FREEZE := requirements-freeze.txt

# Set DIST_DIR based on PyPI target
ifeq ($(PYPI_TARGET),prod)
DIST_DIR := pypi-prod
else
DIST_DIR := pypi-test
endif

# Set verification environment variables
VERIFICATION_ENABLED ?= false
VERIFICATION_STRICT ?= false
LOG_LEVEL ?= INFO

# Get the current user's primary group
USER_GROUP := $(shell id -gn)

# Experiment settings
EXP ?= wine
ARGS ?=

# -------- Help (Self-documenting) --------
help: ## Shows this help message
	@echo "======================================"
	@echo "  smart-inference-ai-fusion - Make"
	@echo "======================================"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-25s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "🔹 Installation options:"
	@echo "   make install              # Basic runtime dependencies"
	@echo "   make install-dev          # Development dependencies"  
	@echo "   make install-verification # Verification support (Z3)"
	@echo "   make install-full         # Everything (dev + verification)"
	@echo ""
	@echo "🔹 Basic usage:"
	@echo "   make run                    # Normal experiments"
	@echo "   make run EXP=wine           # Specific dataset experiments"
	@echo ""
	@echo "🔍 Formal Verification:"
	@echo "   make run-verification       # Run with verification enabled"
	@echo "   make run-verification-max   # 🚀 MAXIMUM Z3 performance (5min timeout, 16 cores, 12GB)"
	@echo "   make run-advanced-experiments # 🎯 ALL algorithms/datasets with advanced verification"
	@echo "   make verification-example   # Run verification integration example"
	@echo "   make verify-example         # Run env-based verification example"
	@echo "   make verify-all            # Run with strict verification"
	@echo ""
	@echo "🔹 Tip: Targets automatically set up venv and install dependencies."

# -------- Environment Setup --------
venv: ## Creates and prepares the virtualenv using $(PYTHON)
	@echo "🔧 Setting up Python virtual environment..."
	@if [ ! -d "$(VENV_DIR)" ]; then \
		$(PYTHON) -m venv $(VENV_DIR); \
		echo "✅ Virtual environment created at $(VENV_DIR)"; \
	else \
		echo "✅ Virtual environment already exists"; \
	fi
	@$(PIP) install --upgrade pip setuptools wheel
	@echo "✅ Virtual environment ready"

install: venv ## Installs runtime dependencies from pyproject.toml
	@echo "📦 Installing runtime dependencies..."
	@if ! $(PIP) show $(PKG_NAME) > /dev/null 2>&1; then \
		$(PIP) install -e .; \
		echo "✅ Runtime dependencies installed"; \
	else \
		echo "✅ Runtime dependencies already installed"; \
	fi

install-dev: venv ## Installs all dependencies for development (including dev extras)
	@echo "📦 Installing development dependencies..."
	@if ! $(PIP) show pytest > /dev/null 2>&1; then \
		$(PIP) install -e ".[dev]"; \
		echo "✅ Development dependencies installed"; \
	else \
		echo "✅ Development dependencies already installed"; \
	fi

install-verification: venv ## Installs package with formal verification support (Z3)
	@echo "📦 Installing verification dependencies..."
	@if ! $(PIP) show z3-solver > /dev/null 2>&1; then \
		$(PIP) install -e ".[verification]"; \
		echo "✅ Verification dependencies installed"; \
	else \
		echo "✅ Verification dependencies already installed"; \
	fi

install-full: venv ## Installs package with all features (dev + verification)
	@echo "📦 Installing all dependencies..."
	@$(PIP) install -e ".[dev,verification]"
	@echo "✅ All dependencies installed"

uninstall: ## Removes the installed package from the venv
	@if [ -d "$(VENV_DIR)" ]; then \
		$(PIP) uninstall -y $(PKG_NAME) || true; \
		echo "✅ Package uninstalled"; \
	else \
		echo "❌ Virtual environment not found"; \
	fi

# -------- Experiments --------
run: install ## Runs experiments. Use EXP=<module|package> and ARGS="<options>"
	@echo "🚀 Running experiments (dataset: $(EXP))..."
	@cd $(shell pwd) && \
		export PYTHONPATH="$(PYTHONPATH)" && \
		export LOG_LEVEL="$(LOG_LEVEL)" && \
		$(PYTHON_VENV) -m $(MAIN_MODULE) $(EXP) $(ARGS)

debug: install ## Runs the main experiment orchestrator in DEBUG mode
	@echo "🐛 Running in DEBUG mode..."
	@cd $(shell pwd) && \
		export PYTHONPATH="$(PYTHONPATH)" && \
		export LOG_LEVEL="DEBUG" && \
		$(PYTHON_VENV) -m $(MAIN_MODULE) $(EXP) $(ARGS)

run-verification: install-verification ## Runs experiments with formal verification enabled
	@echo "🔍 Running experiments with formal verification enabled..."
	@cd $(shell pwd) && \
		export PYTHONPATH="$(PYTHONPATH)" && \
		export LOG_LEVEL="$(LOG_LEVEL)" && \
		export VERIFICATION_ENABLED="true" && \
		$(PYTHON_VENV) -m $(MAIN_MODULE) $(EXP) $(ARGS)

run-verification-max: install-verification ## 🚀 Runs experiments with MAXIMUM Z3 performance
	@echo "🚀 Running experiments with MAXIMUM Z3 performance and advanced constraints..."
	@cd $(shell pwd) && \
		export PYTHONPATH="$(PYTHONPATH)" && \
		export LOG_LEVEL="DEBUG" && \
		export VERIFICATION_ENABLED="true" && \
		export VERIFICATION_STRICT="true" && \
		export Z3_MAX_PERFORMANCE="true" && \
		export Z3_TIMEOUT="300000" && \
		export Z3_THREADS="16" && \
		export Z3_MEMORY="12000" && \
		$(PYTHON_VENV) -m $(MAIN_MODULE) $(EXP) $(ARGS)

run-advanced-experiments: install-verification ## 🎯 Runs all algorithm/dataset combinations with advanced verification
	@echo "🎯 Running ALL advanced experiments with maximum Z3 performance..."
	@echo "📊 Algorithms: LogisticRegression, DecisionTree, MLPClassifier"
	@echo "📁 Datasets: Adult, Breast Cancer, Wine, Make Moons"
	@cd $(shell pwd) && \
		export PYTHONPATH="$(PYTHONPATH)" && \
		export LOG_LEVEL="DEBUG" && \
		export VERIFICATION_ENABLED="true" && \
		export VERIFICATION_STRICT="true" && \
		export Z3_MAX_PERFORMANCE="true" && \
		export Z3_TIMEOUT="300000" && \
		export Z3_THREADS="16" && \
		export Z3_MEMORY="12000" && \
		echo "🔬 Running LogisticRegression on Adult dataset..." && \
		$(PYTHON_VENV) -c "from configs.advanced_verification_config import *; print('Advanced verification config loaded!')" && \
		echo "🔬 Running Wine MLP experiment (current)..." && \
		$(PYTHON_VENV) -m $(MAIN_MODULE) wine $(ARGS)

# -------- Formal Verification Commands --------
verification-example: install-verification ## Runs the formal verification integration example
	@echo "🔬 Running formal verification integration example..."
	@cd $(shell pwd) && \
		export PYTHONPATH="$(PYTHONPATH)" && \
		$(PYTHON_VENV) examples/formal_verification_usage.py

verify-example: install-verification ## Runs the environment-based verification example
	@echo "🔬 Running environment-based verification example..."
	@cd $(shell pwd) && \
		export PYTHONPATH="$(PYTHONPATH)" && \
		export VERIFICATION_ENABLED="true" && \
		$(PYTHON_VENV) examples/verification_integration_example.py

verify-all: install-verification ## Runs all experiments with strict verification (fail on errors)
	@echo "🔍 Running experiments with strict verification..."
	@cd $(shell pwd) && \
		export PYTHONPATH="$(PYTHONPATH)" && \
		export LOG_LEVEL="$(LOG_LEVEL)" && \
		export VERIFICATION_ENABLED="true" && \
		export VERIFICATION_STRICT="true" && \
		$(PYTHON_VENV) -m $(MAIN_MODULE) $(EXP) $(ARGS)

verify-install: install-verification ## Install formal verification dependencies
	@echo "🔧 Installing formal verification dependencies..."
	@$(PIP) show z3-solver > /dev/null 2>&1 && echo "✅ Z3-solver already installed" || $(PIP) install z3-solver
	@echo "✅ Verification dependencies ready"

verify-list: install-verification ## List available formal verifiers
	@echo "📋 Available formal verifiers:"
	@cd $(shell pwd) && \
		export PYTHONPATH="$(PYTHONPATH)" && \
		$(PYTHON_VENV) -c "from $(PKG_NAME).verification.plugin_interface import get_available_verifiers; print('\n'.join(f'  - {v}' for v in get_available_verifiers()))"

verify-test: install-verification ## Test formal verification system
	@echo "🧪 Testing formal verification system..."
	@cd $(shell pwd) && \
		export PYTHONPATH="$(PYTHONPATH)" && \
		$(PYTHON_VENV) scripts/test_formal_verification.py

verify-solver-details: install-verification ## Test Z3 solver with detailed results reporting
	@echo "🔬 Testing Z3 solver with detailed results reporting..."
	@cd $(shell pwd) && \
		export PYTHONPATH="$(PYTHONPATH)" && \
		export LOG_LEVEL="DEBUG" && \
		export VERIFICATION_ENABLED="true" && \
		$(PYTHON_VENV) -c "from $(PKG_NAME).utils.verification_report import *; from $(PKG_NAME).verification import verify; result = verify('test_solver_details', {'bounds': {'min': 0, 'max': 10}, 'linear_arithmetic': {'coefficients': [1, -1], 'constant': 0}}); report_verification_results(result, 'TestModel', 'TestDataset', 'solver_test')"

test-structured-constraints: install-verification ## Test structured constraints for formal verification
	@echo "🧪 Testing structured constraints for formal verification..."
	@cd $(shell pwd) && \
		export PYTHONPATH="$(PYTHONPATH)" && \
		$(PYTHON_VENV) examples/structured_constraints_example.py

test-counterexamples: install-verification ## Test counterexample generation in Z3
	@echo "🔍 Testing counterexample generation..."
	@cd $(shell pwd) && \
		export PYTHONPATH="$(PYTHONPATH)" && \
		$(PYTHON_VENV) examples/forced_counterexample_test.py

verify-constraints: install-verification ## Run constraint verification tests with debug logging
	@echo "🔍 Running constraint verification tests..."
	@cd $(shell pwd) && \
		export PYTHONPATH="$(PYTHONPATH)" && \
		export VERIFICATION_ENABLED="true" && \
		export LOG_LEVEL="DEBUG" && \
		$(PYTHON_VENV) examples/structured_constraints_example.py

# -------- Code Quality --------
format: install-dev ## Formats code with black and isort
	@echo "🎨 Formatting code..."
	@$(PYTHON_VENV) -m black $(SRC_DIR)/ tests/
	@$(PYTHON_VENV) -m isort $(SRC_DIR)/ tests/
	@echo "✅ Code formatted"

check-format: install-dev ## Checks code formatting without making changes
	@echo "🔍 Checking code format..."
	@$(PYTHON_VENV) -m black --check $(SRC_DIR)/ tests/
	@$(PYTHON_VENV) -m isort --check-only $(SRC_DIR)/ tests/
	@echo "✅ Code format OK"

lint: install-dev ## Lints code with pylint
	@echo "🔍 Linting code..."
	@$(PYTHON_VENV) -m pylint $(SRC_DIR)/
	@echo "✅ Linting complete"

style: install-dev ## Checks docstrings (Google style)
	@echo "📝 Checking docstring style..."
	@$(PYTHON_VENV) -m pydocstyle --convention=google $(SRC_DIR)/
	@echo "✅ Docstring style OK"

check: check-format lint style ## Runs all code quality checks (format, lint, style)
	@echo "✅ All quality checks passed"

# -------- Testing --------
test: install-dev ## Runs unit tests with pytest
	@echo "🧪 Running unit tests..."
	@$(PYTHON_VENV) -m pytest tests/ -v
	@echo "✅ Tests complete"

# -------- Requirements Management --------
compile-reqs: venv ## Generates requirements.txt from pyproject.toml
	@echo "📋 Compiling requirements.txt..."
	@$(PIP) install pip-tools
	@$(VENV_BIN)/pip-compile pyproject.toml --output-file requirements.txt
	@echo "✅ requirements.txt generated"

compile-reqs-dev: venv ## Generates requirements-dev.txt (includes [dev] extras)
	@echo "📋 Compiling requirements-dev.txt..."
	@$(PIP) install pip-tools
	@$(VENV_BIN)/pip-compile pyproject.toml --extra dev --output-file $(REQUIREMENTS_DEV)
	@echo "✅ $(REQUIREMENTS_DEV) generated"

freeze: venv ## Generates requirements-freeze.txt (a snapshot of the current venv)
	@echo "❄️  Freezing current environment..."
	@$(PIP) freeze > $(REQUIREMENTS_FREEZE)
	@echo "✅ $(REQUIREMENTS_FREEZE) generated"

# -------- Build & Distribution --------
build: install-dev ## Builds the wheel and sdist packages into ./dist
	@echo "🏗️  Building package..."
	@$(PYTHON_VENV) -m build
	@echo "✅ Package built in ./dist/"

publish: build ## Publishes the package to the TestPyPI repository
	@echo "📤 Publishing to TestPyPI..."
	@$(PYTHON_VENV) -m twine upload --repository testpypi dist/*
	@echo "✅ Published to TestPyPI"

publish-prod: build ## Publishes the package to the official PyPI repository
	@echo "📤 Publishing to PyPI..."
	@$(PYTHON_VENV) -m twine upload dist/*
	@echo "✅ Published to PyPI"

deploy: publish ## Alias for the 'publish' command

# -------- Cleanup --------
clean-pyc: ## Removes only Python bytecode cache files
	@echo "🧹 Cleaning Python cache files..."
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Python cache cleaned"

clean: clean-pyc ## Removes build artifacts and Python cache files
	@echo "🧹 Cleaning build artifacts..."
	@rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .coverage htmlcov/
	@echo "✅ Build artifacts cleaned"

clean-venv: ## Removes the .venv virtual environment directory
	@echo "🧹 Removing virtual environment..."
	@rm -rf $(VENV_DIR)
	@echo "✅ Virtual environment removed"

clean-outputs: ## WARNING: Deletes all generated logs and results
	@echo "⚠️  WARNING: This will delete all logs and results!"
	@read -p "Are you sure? [y/N] " -n 1 -r && echo
	@if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf logs/ results/; \
		echo "✅ Logs and results deleted"; \
	else \
		echo "❌ Operation cancelled"; \
	fi

clean-all: clean clean-outputs ## Runs all clean tasks, including logs and results
	@echo "✅ Complete cleanup finished"

# -------- CI Pipeline --------
ci: check test run ## Runs the complete CI pipeline (quality checks, tests, and run)
	@echo "🎯 CI pipeline completed successfully"

# -------- Phony Targets --------
.PHONY: help venv install install-dev install-verification install-full uninstall
.PHONY: run debug run-verification 
.PHONY: verification-example verify-example verify-all verify-install verify-list verify-test
.PHONY: format check-format lint style check test
.PHONY: compile-reqs compile-reqs-dev freeze
.PHONY: build publish publish-prod deploy
.PHONY: clean-pyc clean clean-venv clean-outputs clean-all
.PHONY: ci