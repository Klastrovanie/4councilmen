#!/bin/bash
# 4CM Theory - Quick Run Script
# PhD Dissertation 2011 — Claude API Version 2026

echo "=================================="
echo "  4CM Theory — Claude API Version"
echo "=================================="
echo ""

# Check API key
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "ERROR: ANTHROPIC_API_KEY is not set."
    echo "Run: export ANTHROPIC_API_KEY=your_key_here"
    exit 1
fi

# Install dependencies
echo "[1/2] Installing dependencies..."
pip install anthropic numpy scikit-learn sentence-transformers matplotlib requests -q

# Run
echo "[2/2] Running multi-round demo..."
echo ""
cd "$(dirname "$0")"
python multi_round_demo_nocontext.py
