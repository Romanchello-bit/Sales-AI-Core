import os
import sys
import traceback
import streamlit as st

# Make sure project root is on PYTHONPATH so we can import app.py
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Importing app.py will execute the Streamlit UI defined there
try:
    import app  # noqa: F401
except Exception as e:
    st.error("Failed to load the main app module ('app.py'). See details below and fix missing dependencies or import errors.")
    st.code(traceback.format_exc(), language="python")
    st.stop()

if __name__ == "__main__":
    # Nothing extra to run here; the UI renders on import of app.py
    pass