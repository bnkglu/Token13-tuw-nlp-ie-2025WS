#!/bin/bash
# Setup script for SemEval-2010 Task 8 preprocessing pipeline

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo ""
echo "Downloading spaCy English model (en_core_web_lg)..."
python3 -m spacy download en_core_web_lg

echo ""
echo "Setup complete!"
echo ""
echo "The preprocessing pipeline can now be run:"
echo "  python3 src/preprocess.py --split both"
