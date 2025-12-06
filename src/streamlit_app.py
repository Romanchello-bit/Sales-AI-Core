import os
import sys
import streamlit as st

# Make sure project root is on PYTHONPATH so we can import app.py
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Importing app.py will execute the Streamlit UI defined there
import app  # noqa: F401

if __name__ == "__main__":
    # Nothing extra to run here; the UI renders on import of app.py
    pass