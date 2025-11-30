#!/bin/bash

# --- CONFIGURATION ---
REQUIRED_PYTHON="3.10.4"  # CHANGE THIS to your specific version
ENV_NAME=".potato_ui_env"
# ---------------------

echo "--- Setting up environment with Python $REQUIRED_PYTHON ---"

# 1. Check if pyenv is installed
if ! command -v pyenv &> /dev/null; then
    echo "Error: pyenv is not installed."
    echo "Please install it: https://github.com/pyenv/pyenv#installation"
    exit 1
fi

# 2. Install the specific Python version if missing
if ! pyenv versions | grep -q $REQUIRED_PYTHON; then
    echo "Python $REQUIRED_PYTHON not found. Installing..."
    pyenv install $REQUIRED_PYTHON
else
    echo "Python $REQUIRED_PYTHON is already installed."
fi

# 3. Set local version and create venv
# This ensures 'python' command refers to the specific version in this folder
pyenv local $REQUIRED_PYTHON

echo "Creating virtual environment '$ENV_NAME' using $(python --version)..."
python -m venv $ENV_NAME

# 4. Activate and Install
source $ENV_NAME/bin/activate

echo "--- Upgrading pip and installing requirements ---"
pip install --upgrade pip

if [ -f "potato_ui_env_reqs.txt" ]; then
    pip install -r potato_ui_env_reqs.txt
fi

echo "--- Setup Complete! ---"
echo "Run: source $ENV_NAME/bin/activate"