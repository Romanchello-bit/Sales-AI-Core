# Sales AI Core

This repository contains the core tooling and experiments for the Sales AI platform. The primary runnable product in this repository is a set of command-line experiments and visualization plots driven by `main.py` (benchmark and analysis runner).

## Quick-start (Windows PowerShell)

# Create and activate virtual environment, install deps, run main
powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py

# Run tests
powershell
.\.venv\Scripts\Activate.ps1
python -m unittest discover -v

## Quick-start (Makefile - cross-platform with GNU make)

# Create venv and install (on Windows use Git Bash or WSL to run make)
make install

# Run main
make start

# Run Streamlit UI
make streamlit

# Run tests
make test

## Devcontainer

Open in VS Code Remote-Containers or Codespaces. The devcontainer builds a Docker image with Python 3.14 and Graphviz, creates `.venv` and installs `requirements.txt` during `postCreateCommand`.

Activate venv inside the container:

bash
source .venv/bin/activate

Start the app (inside the container):

bash
python main.py

Or run the Streamlit UI:

bash
streamlit run src/streamlit_app.py --server.enableCORS false --server.enableXsrfProtection false

## Notes
- The Graphviz Python package requires the system `dot` binary (installed in the devcontainer Dockerfile).
- Use the provided scripts in `scripts/` if you prefer PowerShell or shell helper scripts.

## Key entrypoints

- `main.py` — runs benchmark experiments (Bellman-Ford comparisons) and produces `benchmark_results.png`.
- `src/streamlit_app.py` — optional Streamlit demo/UI (not the main product by default).

## Getting started (recommended)

1. Create and activate a virtual environment (recommended):

    ```bash
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1  # PowerShell
    ```

2. Install dependencies:

    ```bash
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    ```

3. Run the main product (benchmarks and plots):

    ```bash
    python main.py
    ```

   The script will run experiments, save a `benchmark_results.png` file in the repository root, and open a matplotlib window for the plots.

## Optional: Run the Streamlit demo

If you want to run the Streamlit demo UI, use the included Streamlit app (this is optional and not the default entrypoint):

```bash
streamlit run src/streamlit_app.py
```

## Notes and troubleshooting

- Make sure you run commands with the same Python interpreter that has the dependencies installed (use the full path shown by `python -c "import sys; print(sys.executable)"` when in doubt).
- The `graphviz` Python package requires the Graphviz system binary (`dot`) to be installed and available on your PATH for rendering graph images. On Windows, install Graphviz and add its `bin` folder (e.g. `C:\Program Files\Graphviz\bin`) to your PATH.
- If you plan to use the Google Generative AI functionality, ensure you set `GOOGLE_API_KEY` or provide the API key where the Streamlit app prompts for it. The project uses `google-generativeai` (see `requirements.txt`).

## Project structure (high level)

- `main.py` — benchmark runner and plotting
- `src/streamlit_app.py` — Streamlit-based demo UI
- `algorithms.py`, `generator.py`, `experiments.py`, `graph_module.py` — core algorithm and experiment code
- `requirements.txt`, `packages.txt` — dependency lists

If you want, I can also update `requirements.txt` or add a short helper script to create and activate the environment on Windows. Would you like me to add that helper or adjust the README further?
