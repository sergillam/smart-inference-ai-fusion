PYTHON=python
SRC_DIR=src
EXPERIMENTS_DIR=experiments
TESTS_DIR=tests

.PHONY: run lint test all

run:
	@echo "🚀 Executing Inference Framework..."
	PYTHONPATH=$(SRC_DIR) $(PYTHON) main.py
	@echo "✅ Done."

lint:
	@echo "🔍 Running pylint..."
	pylint $(SRC_DIR) $(EXPERIMENTS_DIR)

test:
	@echo "🧪 Running unit tests..."
	pytest $(TESTS_DIR)

all: lint test run
