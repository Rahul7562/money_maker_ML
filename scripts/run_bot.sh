#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

if [[ -f .venv/bin/activate ]]; then
	source .venv/bin/activate
elif [[ -f venv/bin/activate ]]; then
	source venv/bin/activate
else
	echo "No virtual environment found. Run ./scripts/setup_ubuntu.sh first."
	exit 1
fi

python main.py
