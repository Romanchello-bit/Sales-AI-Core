# PowerShell dev start script
Set-StrictMode -Version Latest
if (-Not (Test-Path -Path .venv\Scripts\Activate.ps1)) {
    python -m venv .venv
}
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
if (Test-Path requirements.txt) { python -m pip install -r requirements.txt }
python main.py

