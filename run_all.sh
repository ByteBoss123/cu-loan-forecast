#!/usr/bin/env bash
# Full pipeline: data prep -> origination -> credit risk -> report outputs -> figures
set -euo pipefail
cd "$(dirname "$0")"
python3 src/01_prepare.py
python3 src/02_origination.py
python3 src/03_credit_risk.py
python3 src/04_report_data.py
python3 src/05_figures.py
