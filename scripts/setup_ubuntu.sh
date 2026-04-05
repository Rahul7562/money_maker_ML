#!/usr/bin/env bash
set -euo pipefail

sudo apt update
sudo apt install -y python3 python3-venv python3-pip

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env from .env.example. Edit .env if you need live keys."
fi

echo "Ubuntu setup complete."
