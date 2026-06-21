#!/usr/bin/env bash
# Run from the project root: ./run.sh
# Sets PYTHONPATH so that `from app.rag.xyz import ...` works correctly.
cd "$(dirname "$0")"
PYTHONPATH=. ./venv/bin/streamlit run app/main.py "$@"
