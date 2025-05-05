#!/bin/bash

# setupenv.sh: Set up and activate the TrialMesh environment (source this script!)
# Logs all output to setupenv.log

# Guard: must be sourced, not executed
(return 0 2>/dev/null) || { echo "Please source this script: source setupenv.sh"; exit 1; }

DEBUG=1  # Set to 1 for debugging 0 for not

if [[ $DEBUG -eq 0 ]]; then
    set -e
fi

LOGFILE="setupenv.log"
exec > >(tee -a "$LOGFILE") 2>&1

ENV_NAME="trialmesh"
PYTHON_VERSION="3.12"
VLLM_VERSION="0.8.5"
PYTEST_VERSION="8.3.3"

echo "=== TrialMesh environment setup started at $(date) ==="

if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install Anaconda or Miniconda first."
    return 1 2>/dev/null || exit 1
fi

# Check if environment exists
if conda info --envs | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    echo "Conda environment '$ENV_NAME' already exists."
    read -p "Do you want to delete and recreate it? [y/N]: " yn
    case "$yn" in
        [Yy]* )
            echo "Removing environment '$ENV_NAME'..."
            conda deactivate 2>/dev/null || true
            conda remove -n "$ENV_NAME" --all -y
            echo "Creating conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
            conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
            ;;
        * )
            echo "Keeping existing environment."
            ;;
    esac
else
    echo "Creating conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
fi

echo "Activating environment '$ENV_NAME'..."
conda activate "$ENV_NAME"

# Print the current environment
CUR_ENV=$(conda info --json | python -c "import sys, json; print(json.load(sys.stdin)['active_prefix_name'])")
echo "Current active conda environment: $CUR_ENV"

if [[ "$CUR_ENV" != "$ENV_NAME" ]]; then
    echo "WARNING: Expected environment '$ENV_NAME' but current is '$CUR_ENV'."
    echo "Conda activation may not have worked as expected. You may need to manually 'conda activate $ENV_NAME'."
    return 1 2>/dev/null || exit 1
fi

echo "Installing vllm==$VLLM_VERSION via pip..."
pip install "vllm==$VLLM_VERSION"
pip install "pytest==$PYTEST_VERSION"

echo "Environment setup complete. You are now in the '$ENV_NAME' environment."

echo "=== Running test suite: trialmesh/tests/test_torch.py ==="
pytest -s tests/test_torch.py

echo "=== TrialMesh environment setup and test complete at $(date) ==="