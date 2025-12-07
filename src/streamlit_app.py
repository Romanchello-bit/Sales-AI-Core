import os
import sys
import traceback

# Ensure app.py sees that we're intentionally running the Streamlit UI
os.environ.setdefault('RUNNING_IN_STREAMLIT', '1')

# Make sure project root is on PYTHONPATH so we can import app.py
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Importing app.py will not execute the Streamlit UI at import time anymore
try:
    import app  # noqa: F401
except Exception:
    # Streamlit may not be available in this environment; show a console traceback instead of calling streamlit APIs
    print("Failed to load the main app module ('app.py'). See traceback below:")
    traceback.print_exc()
    sys.exit(1)

if __name__ == "__main__":
    # When launched directly (e.g., via `streamlit run src/streamlit_app.py`), delegate to app's runner
    try:
        app.run_streamlit_app()
    except Exception:
        print("Error while running the Streamlit app:")
        traceback.print_exc()
        raise
