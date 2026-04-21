#!/bin/bash
# ============================================================
# PySpark Project — Conda Environment Setup
# ============================================================

ENV_NAME="pyspark-env"
PYTHON_VERSION="3.11"

echo "==> Creating conda environment: $ENV_NAME (Python $PYTHON_VERSION)"
conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y

echo "==> Activating environment"
conda activate "$ENV_NAME"

# --- Core: PySpark + Java runtime ---
echo "==> Installing PySpark and OpenJDK via conda-forge"
conda install -c conda-forge pyspark openjdk=17 -y

# --- Useful extras (install via pip for latest versions) ---
echo "==> Installing additional Python dependencies"
pip install \
    pandas \
    pyarrow \
    numpy \
    matplotlib \
    jupyter \
    pytest \
    scikit-learn

echo ""
echo "============================================"
echo "  Setup complete!"
echo "  Activate with:  conda activate $ENV_NAME"
echo "============================================"