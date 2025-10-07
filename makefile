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
SOLVERS ?= auto

# Default output directories (can be overridden on the make command line)
LOGS_DIR ?= logs
RESULTS_DIR ?= results

# -------- Help (Self-documenting) --------
help: ## Shows this help message
	@echo "======================================"
	@echo "  smart-inference-ai-fusion - Make"
	@echo "======================================"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-25s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "🔹 Installation options:"
	@echo "   make install              # Complete dev environment (fresh venv)"
	@echo "   make install-quick        # Complete dev environment (keep venv)"
	@echo "   make install-dev          # Development dependencies only"
	@echo "   make install-verification # Verification support (Z3) only"
	@echo "   make install-cvc5         # CVC5 SMT solver only"
	@echo "   make clean-venv           # Clean and recreate virtual environment"
	@echo ""
	@echo "🔹 Experiment modes:"
	@echo "   make run-basic            # 🔹 Basic algorithms (no verification)"
	@echo "   make run-inference        # 🧠 Algorithms + synthetic inference"
	@echo "   make run-verification     # 🔍 Algorithms + formal verification"
	@echo "   make run-all              # 🎯 Complete pipeline (all modes)"
	@echo ""
	@echo "🔹 Solver selection:"
	@echo "   make run-z3               # � Z3 solver only"
	@echo "   make run-cvc5             # 🧮 CVC5 solver only"
	@echo "   make run-both-solvers     # ⚡ Both solvers (parallel + comparison)"
	@echo ""
	@echo "🔹 Advanced options:"
	@echo "   make debug                # 🐛 DEBUG mode with detailed logging"
	@echo "   make run-advanced         # 🚀 Maximum performance (both solvers)"
	@echo ""
	@echo "🔹 Dataset experiments (generic):"
	@echo "   make run EXP=wine             # Run wine experiments (basic mode)"
	@echo "   make run-all EXP=wine         # Run ALL wine experiments (all modes)"
	@echo "   make run-basic EXP=wine       # Wine experiments (basic only)"
	@echo "   make run-inference EXP=wine   # Wine experiments (inference only)"
	@echo "   make run-verify EXP=wine      # Wine experiments (verification only)"
	@echo ""
	@echo "🔹 Solver-specific experiments (any dataset):"
	@echo "   make run-z3 EXP=wine          # Wine with Z3 solver"
	@echo "   make run-cvc5 EXP=digits      # Digits with CVC5 solver"
	@echo "   make run-both-solvers EXP=wine # Wine with both solvers"
	@echo ""
	@echo "🔹 Advanced syntax (like you requested):"
	@echo "   make run verify EXP=wine                    # Verify all wine experiments"
	@echo "   make run verify EXP=digits SOLVERS=z3       # Digits with Z3 only"
	@echo "   make run verify EXP=wine SOLVERS=cvc5       # Wine with CVC5 only"
	@echo "   make run verify EXP=wine SOLVERS=z3,cvc5    # Wine with both solvers"
	@echo ""

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

clean-venv: ## Removes and recreates the virtual environment
	@echo "🧹 Cleaning virtual environment..."
	@if [ -d "$(VENV_DIR)" ]; then \
		rm -rf $(VENV_DIR); \
		echo "🗑️ Removed old virtual environment"; \
	fi
	@$(PYTHON) -m venv $(VENV_DIR)
	@$(PIP) install --upgrade pip setuptools wheel
	@echo "✅ Fresh virtual environment created"

install: clean-venv ## Installs complete development environment (dev + verification + CVC5 + sklearn)
	@echo "📦 Installing complete development environment..."
	@echo "🔧 Installing base package with all extras..."
	@$(PIP) install -e ".[dev,verification]"
	@echo "🧠 Installing CVC5 SMT solver..."
	@$(PIP) install "cvc5>=1.0.5"
	@echo "📊 Installing additional ML dependencies..."
	@$(PIP) install "scikit-learn>=1.3.0" "pandas>=2.0.0" "numpy>=1.24.0"
	@echo "✅ Complete development environment installed!"
	@echo ""
	@echo "🎯 Available solvers:"
	@echo "   - Z3 (via verification extras)"
	@echo "   - CVC5 (latest version)"
	@echo "🔬 ML Libraries:"
	@echo "   - scikit-learn, pandas, numpy"
	@echo "🛠️ Dev Tools:"
	@echo "   - pytest, pylint, black, mypy"

install-quick: venv ## Installs complete environment without cleaning venv first
	@echo "📦 Installing complete development environment (preserving venv)..."
	@echo "🔧 Installing base package with all extras..."
	@$(PIP) install -e ".[dev,verification]"
	@echo "🧠 Installing CVC5 SMT solver..."
	@$(PIP) install "cvc5>=1.0.5"
	@echo "📊 Installing additional ML dependencies..."
	@$(PIP) install "scikit-learn>=1.3.0" "pandas>=2.0.0" "numpy>=1.24.0"
	@echo "✅ Complete development environment installed!"

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

install-cvc5: venv ## Installs CVC5 SMT solver for advanced verification
	@echo "📦 Installing CVC5 solver..."
	@if ! $(PIP) show cvc5 > /dev/null 2>&1; then \
		$(PIP) install "cvc5>=1.0.5"; \
		echo "✅ CVC5 solver installed"; \
	else \
		echo "✅ CVC5 solver already installed"; \
	fi

install-full: venv ## Installs package with all features (dev + verification + CVC5)
	@echo "📦 Installing all dependencies..."
	@$(PIP) install -e ".[dev,verification]"
	@$(PIP) install "cvc5>=1.0.5"
	@echo "✅ All dependencies installed"

uninstall: ## Removes the installed package from the venv
	@if [ -d "$(VENV_DIR)" ]; then \
		$(PIP) uninstall -y $(PKG_NAME) || true; \
		echo "✅ Package uninstalled"; \
	else \
		echo "❌ Virtual environment not found"; \
	fi

# implementations appear later in the file and are the canonical definitions.
# -------- Debug and Advanced --------
debug: install-quick ## 🐛 Runs in DEBUG mode with detailed logging
	@echo "🐛 Running in DEBUG mode..."
	@cd $(shell pwd) && \
		export PYTHONPATH="$(PYTHONPATH)" && \
		export LOG_LEVEL="DEBUG" && \
		export VERIFICATION_DETAILED_LOG="true" && \
		$(PYTHON_VENV) -m $(MAIN_MODULE) $(EXP) $(ARGS)

run-advanced: install-quick ## 🚀 Runs with MAXIMUM performance settings
	@echo "� Running with MAXIMUM performance settings..."
	@cd $(shell pwd) && \
		export PYTHONPATH="$(PYTHONPATH)" && \
		export LOG_LEVEL="DEBUG" && \
		export VERIFICATION_MODE="all" && \
		export VERIFICATION_SOLVER="both" && \
		export VERIFICATION_PARALLEL="true" && \
		export VERIFICATION_COMPARE_SOLVERS="true" && \
		export VERIFICATION_TIMEOUT_TOTAL="600" && \
		export VERIFICATION_DETAILED_LOG="true" && \
		$(PYTHON_VENV) -m $(MAIN_MODULE) $(EXP) $(ARGS)
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

# -------- Error Handling & Robustness Testing --------
test-error-handling: install-quick ## 🛡️ Tests robust error handling system
	@echo "🛡️ Testing robust error handling system..."
	@cd $(shell pwd) && \
		export PYTHONPATH="$(PYTHONPATH)" && \
		export LOG_LEVEL="DEBUG" && \
		export VERIFICATION_MODE="verification" && \
		export VERIFICATION_SOLVER="both" && \
		export VERIFICATION_ERROR_HANDLING="true" && \
		export VERIFICATION_DETAILED_LOG="true" && \
		$(PYTHON_VENV) -c "from smart_inference_ai_fusion.verification.core.error_handling import *; print('🛡️ Error handling system loaded'); print(get_error_summary())"

test-solver-fallback: install-quick ## 🔄 Tests automatic fallback between solvers
	@echo "🔄 Testing automatic fallback between solvers..."
	@cd $(shell pwd) && \
		export PYTHONPATH="$(PYTHONPATH)" && \
		export LOG_LEVEL="DEBUG" && \
		export VERIFICATION_MODE="verification" && \
		export VERIFICATION_SOLVER="both" && \
		export VERIFICATION_FALLBACK_STRATEGY="next_solver" && \
		$(PYTHON_VENV) -c "from smart_inference_ai_fusion.verification.core.error_handling import VerificationErrorHandler, FallbackStrategy; handler = VerificationErrorHandler(FallbackStrategy.NEXT_SOLVER); print('🔄 Fallback system ready')"

test-error-recovery: install-quick ## 🔧 Tests error recovery mechanisms
	@echo "🔧 Testing error recovery mechanisms..."
	@cd $(shell pwd) && \
		export PYTHONPATH="$(PYTHONPATH)" && \
		export LOG_LEVEL="DEBUG" && \
		export VERIFICATION_MODE="verification" && \
		export VERIFICATION_SOLVER="both" && \
		$(PYTHON_VENV) -c "from smart_inference_ai_fusion.verification.core.error_handling import *; print('🔧 Error recovery system loaded'); handler = VerificationErrorHandler(); print('Recovery strategies available:', list(handler.recovery_strategies.keys()))"

test-reliability-tracking: install-quick ## 📊 Tests solver reliability tracking
	@echo "📊 Testing solver reliability tracking..."
	@cd $(shell pwd) && \
		export PYTHONPATH="$(PYTHONPATH)" && \
		export LOG_LEVEL="DEBUG" && \
		$(PYTHON_VENV) -c "from smart_inference_ai_fusion.verification.core.error_handling import global_error_handler; print('📊 Reliability tracking:'); print('Current reliability scores:', global_error_handler.solver_reliability)"

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

# clean-outputs block remains above; CI pipeline defined later. Keep only canonical definitions below.

# ============================
# � DATASET-AGNOSTIC COMMANDS
# ============================

# -------- Generic Dataset Experiments --------
run-all: install-quick ## � Runs ALL experiments for any dataset (all modes: basic + inference + verification)
	@echo "� Running ALL $(EXP) experiments (basic + inference + verification)..."
	@cd $(shell pwd) && \
		export PYTHONPATH="$(PYTHONPATH)" && \
		export LOG_LEVEL="$(LOG_LEVEL)" && \
		$(PYTHON_VENV) -m $(MAIN_MODULE) $(EXP) --mode all $(if $(findstring debug,$(MAKECMDGOALS)),--debug)

run-basic: install-quick ## 🔹 Runs experiments for any dataset (basic algorithms only)
	@echo "🔹 Running $(EXP) experiments (BASIC algorithms only)..."
	@cd $(shell pwd) && \
		export PYTHONPATH="$(PYTHONPATH)" && \
		export LOG_LEVEL="$(LOG_LEVEL)" && \
		$(PYTHON_VENV) -m $(MAIN_MODULE) $(EXP) --mode basic $(if $(findstring debug,$(MAKECMDGOALS)),--debug)

run-inference: install-quick ## 🧠 Runs experiments for any dataset (synthetic inference only)
	@echo "🧠 Running $(EXP) experiments (SYNTHETIC INFERENCE only)..."
	@cd $(shell pwd) && \
		export PYTHONPATH="$(PYTHONPATH)" && \
		export LOG_LEVEL="$(LOG_LEVEL)" && \
		$(PYTHON_VENV) -m $(MAIN_MODULE) $(EXP) --mode inference $(if $(findstring debug,$(MAKECMDGOALS)),--debug)

run-verify: install-quick ## 🔍 Runs experiments for any dataset (formal verification only)
	@echo "🔍 Running $(EXP) experiments (FORMAL VERIFICATION only)..."
	@cd $(shell pwd) && \
		export PYTHONPATH="$(PYTHONPATH)" && \
		export LOG_LEVEL="$(LOG_LEVEL)" && \
		$(PYTHON_VENV) -m $(MAIN_MODULE) $(EXP) --mode verification $(if $(findstring debug,$(MAKECMDGOALS)),--debug)

run-z3: install-quick ## ⚡ Runs experiments for any dataset with Z3 solver only
	@echo "⚡ Running $(EXP) experiments with Z3 SOLVER only..."
	@cd $(shell pwd) && \
		export PYTHONPATH="$(PYTHONPATH)" && \
		export LOG_LEVEL="$(LOG_LEVEL)" && \
		$(PYTHON_VENV) -m $(MAIN_MODULE) $(EXP) --mode verification --solvers z3 $(if $(findstring debug,$(MAKECMDGOALS)),--debug)

run-cvc5: install-quick ## 🧮 Runs experiments for any dataset with CVC5 solver only
	@echo "🧮 Running $(EXP) experiments with CVC5 SOLVER only..."
	@cd $(shell pwd) && \
		export PYTHONPATH="$(PYTHONPATH)" && \
		export LOG_LEVEL="$(LOG_LEVEL)" && \
		$(PYTHON_VENV) -m $(MAIN_MODULE) $(EXP) --mode verification --solvers cvc5 $(if $(findstring debug,$(MAKECMDGOALS)),--debug)

run-both-solvers: install-quick ## ⚡ Runs experiments for any dataset with BOTH solvers (parallel)
	@echo "⚡ Running $(EXP) experiments with BOTH solvers in parallel..."
	@cd $(shell pwd) && \
		export PYTHONPATH="$(PYTHONPATH)" && \
		export LOG_LEVEL="$(LOG_LEVEL)" && \
		$(PYTHON_VENV) -m $(MAIN_MODULE) $(EXP) --mode verification --solvers z3,cvc5 --parallel $(if $(findstring debug,$(MAKECMDGOALS)),--debug)

# ============================
# 🎯 ADVANCED SYNTAX COMMANDS
# ============================

# Comando principal: make run verify EXP=<dataset> SOLVERS=<solvers>
verify: install-quick ## 🔍 Advanced syntax: make run verify EXP=wine SOLVERS=z3,cvc5
	@echo "🔍 Advanced verification command..."
	@echo "📊 Dataset: $(EXP)"
	@echo "🔧 Solvers: $(SOLVERS)"
	@cd $(shell pwd) && \
	export PYTHONPATH="$(PYTHONPATH)" && \
	export LOG_LEVEL="$(LOG_LEVEL)" && \
	if [ "$(SOLVERS)" = "z3" ]; then \
		$(PYTHON_VENV) -m $(MAIN_MODULE) $(EXP) --mode verification --solvers z3; \
	elif [ "$(SOLVERS)" = "cvc5" ]; then \
		$(PYTHON_VENV) -m $(MAIN_MODULE) $(EXP) --mode verification --solvers cvc5; \
	elif [ "$(SOLVERS)" = "z3,cvc5" ] || [ "$(SOLVERS)" = "cvc5,z3" ]; then \
		$(PYTHON_VENV) -m $(MAIN_MODULE) $(EXP) --mode verification --solvers z3,cvc5 --parallel; \
	else \
		$(PYTHON_VENV) -m $(MAIN_MODULE) $(EXP) --mode verification; \
	fi

# Para suportar: make run verify EXP=wine
run:
	@if [ "$(filter verify,$(MAKECMDGOALS))" ]; then \
		$(MAKE) verify EXP=$(EXP) SOLVERS=$(SOLVERS); \
	else \
		echo "🚀 Running experiments (dataset: $(EXP))..."; \
		cd $(shell pwd) && \
		export PYTHONPATH="$(PYTHONPATH)" && \
		export LOG_LEVEL="$(LOG_LEVEL)" && \
		$(PYTHON_VENV) -m $(MAIN_MODULE) $(EXP); \
	fi

# Permite que 'verify' seja usado como alvo secundário
.PHONY: verify

## -------- Cleanup outputs (logs/results) --------
clean-outputs: ## WARNING: Deletes all generated logs and results (safe checks included)
	@if [ -z "$(LOGS_DIR)" ] || [ -z "$(RESULTS_DIR)" ]; then \
		echo "⚠️  LOGS_DIR or RESULTS_DIR empty. Aborting to avoid catastrophic delete."; \
		exit 1; \
	fi
	# Disallow absolute paths for safety
	@if echo "$(LOGS_DIR)" | grep -q "^/" || echo "$(RESULTS_DIR)" | grep -q "^/"; then \
		echo "⚠️  Absolute paths not allowed for LOGS_DIR/RESULTS_DIR. Aborting."; \
		exit 1; \
	fi
	# Disallow . or .. as targets
	@if [ "$(LOGS_DIR)" = "." ] || [ "$(RESULTS_DIR)" = "." ] || [ "$(LOGS_DIR)" = ".." ] || [ "$(RESULTS_DIR)" = ".." ]; then \
		echo "⚠️  LOGS_DIR/RESULTS_DIR cannot be '.' or '..' - Aborting."; \
		exit 1; \
	fi
	@echo "🔥 Deleting all contents of $(LOGS_DIR)/ and $(RESULTS_DIR)/..."
	@if [ -d "$(LOGS_DIR)" ]; then \
		find "$(LOGS_DIR)" -mindepth 1 -exec rm -rf -- {} + 2>/dev/null || true; \
	fi
	@if [ -d "$(RESULTS_DIR)" ]; then \
		find "$(RESULTS_DIR)" -mindepth 1 -exec rm -rf -- {} + 2>/dev/null || true; \
	fi
	@echo "✅ Logs and results contents have been cleared (directories preserved)."

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
