VENV=.venv
PYTHON=${VENV}/Scripts/python.exe
ACTIVATE=${VENV}/Scripts/Activate.ps1

.PHONY: venv install start streamlit test clean

venv:
	python -m venv $(VENV)
	@echo "Created virtual environment at $(VENV)"

install: venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	@echo "Installed requirements"

start: install
	$(PYTHON) main.py

streamlit: install
	$(PYTHON) -m streamlit run src/streamlit_app.py --server.enableCORS false --server.enableXsrfProtection false

test: install
	$(PYTHON) -m unittest discover -v

clean:
	rm -rf $(VENV)
	rm -f benchmark_results.png
	@echo "Cleaned"
