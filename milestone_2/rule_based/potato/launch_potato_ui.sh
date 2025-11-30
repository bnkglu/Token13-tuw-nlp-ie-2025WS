#!/bin/bash
# POTATO UI Launcher Script
# This script activates the potato_ui_env and launches the POTATO frontend

cd "$(dirname "$0")"

echo "========================================="
echo "POTATO UI Launcher"
echo "========================================="
echo ""

# Activate the environment
source ./.potato_ui_env/bin/activate

# Download Stanza English model if not already downloaded
echo "Checking Stanza models..."
python -c "import stanza; stanza.download('en', verbose=False)" 2>/dev/null || echo "Stanza models already downloaded"

echo ""
echo "Starting POTATO UI..."
echo "========================================="
echo ""
echo "Access the UI in your browser at:"
echo "  http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""
echo "========================================="
echo ""

# Launch streamlit with your datasets
streamlit run potato_frontend/frontend/app.py -- \
    -t potato_train_dataset.tsv \
    -v potato_val_dataset.tsv \
    -g ud