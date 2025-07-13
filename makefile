PYTHON = python
SRC_DIR = src
EXPERIMENTS_DIR = experiments
TESTS_DIR = tests

.PHONY: run lint test debug all style check help requirements default

# Allow both 'make lint main.py' and 'make lint TARGET=main.py'
ifeq ($(origin TARGET), undefined)
  ifneq ($(MAKECMDGOALS),)
    ifneq ($(filter lint style check test,$(firstword $(MAKECMDGOALS))),)
      ifneq ($(words $(MAKECMDGOALS)),1)
        TARGET := $(word 2,$(MAKECMDGOALS))
        MAKECMDGOALS := $(firstword $(MAKECMDGOALS))
      endif
    endif
  endif
endif

default: help

help:
	@echo "======================================"
	@echo "  smart-inference-ai-fusion - Make"
	@echo "======================================"
	@echo ""
	@echo "Main commands:"
	@echo "  make run                     - Run the inference framework"
	@echo "  make lint                    - Lint all source, experiments, and tests"
	@echo "  make style                   - Check Google style docstrings in all code"
	@echo "  make check                   - Lint and style check (all code)"
	@echo "  make test                    - Run ALL unit tests"
	@echo "  make debug                   - Run framework with LOG_LEVEL=DEBUG"
	@echo "  make requirements            - Install Python dependencies"
	@echo "  make all                     - Run check, test, and run (full pipeline)"
	@echo ""
	@echo "Advanced usage (directory or file):"
	@echo "  make lint path/to/file.py      - Lint a specific file"
	@echo "  make style some/dir/           - Check docstring style in a specific directory"
	@echo "  make check src/models/         - Lint & style check in a specific directory"
	@echo "  make test tests/test_utils.py  - Run a specific test file"
	@echo ""
	@echo "Or use TARGET= for full flexibility:"
	@echo "  make lint TARGET=src/utils/logging.py"
	@echo "  make style TARGET=experiments/iris/"
	@echo "  make check TARGET=src/models/"
	@echo "  make test TARGET=tests/test_utils.py"
	@echo ""
	@echo "See README for more info or open an Issue for help."

run:
	@echo "🚀 Executing Inference Framework..."
	PYTHONPATH=$(SRC_DIR) $(PYTHON) main.py
	@echo "✅ Done."

lint:
	@echo "🔍 Running pylint..."
	@if [ -z "$(TARGET)" ]; then \
		PYTHONPATH=$(SRC_DIR) pylint $(SRC_DIR) $(EXPERIMENTS_DIR) $(TESTS_DIR); \
	else \
		PYTHONPATH=$(SRC_DIR) pylint $(TARGET); \
	fi

style:
	@echo "📝 Checking docstrings (Google style)..."
	@if [ -z "$(TARGET)" ]; then \
		pydocstyle --convention=google $(SRC_DIR) $(EXPERIMENTS_DIR) $(TESTS_DIR); \
	else \
		pydocstyle --convention=google $(TARGET); \
	fi

check:
	@echo "🔍 Lint and style checks in progress..."
	$(MAKE) lint TARGET="$(TARGET)"
	$(MAKE) style TARGET="$(TARGET)"
	@echo "✅ Lint and style checks completed."

test:
	@echo "🧪 Running unit tests..."
	@if [ -z "$(TARGET)" ]; then \
		PYTHONPATH=$(SRC_DIR) pytest -E $(TESTS_DIR); \
	else \
		PYTHONPATH=$(SRC_DIR) pytest -E $(TARGET); \
	fi

debug:
	@echo "🐞 Executing in DEBUG mode (LOG_LEVEL=DEBUG)..."
	LOG_LEVEL=DEBUG PYTHONPATH=$(SRC_DIR) $(PYTHON) main.py
	@echo "✅ Done (debug)."

all:
	@echo "🔄 Running all tasks: lint, style, test, and run..."
	$(MAKE) check
	$(MAKE) test
	$(MAKE) run
	@echo "✅ All tasks completed."

requirements:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt
	@echo "✅ Dependencies installed."