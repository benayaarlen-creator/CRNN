$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

conda run -n env_skripsi python -m streamlit run streamlit_app.py

