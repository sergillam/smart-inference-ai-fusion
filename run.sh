#!/bin/bash
# run.sh - Executes the framework or auxiliary tasks
# Usage:
#   ./run.sh          → runs main.py
#   ./run.sh lint     → runs pylint
#   ./run.sh all      → runs pylint and then main.py

if [ "$1" == "lint" ]; then
    echo "🔍 Running pylint on src/ and experiments/..."
    pylint src/ experiments/
elif [ "$1" == "all" ]; then
    echo "🔍 Running pylint..."
    pylint src/ experiments/
    echo "🚀 Executing Inference Framework..."
    PYTHONPATH=src python main.py
else
    echo "🚀 Executing Inference Framework..."
    PYTHONPATH=src python main.py
fi
echo "✅ Done."