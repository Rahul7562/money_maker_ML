#!/usr/bin/env bash
# deploy_linux.sh - Deploy Money Maker ML Trading Bot on Ubuntu
# This script installs dependencies, sets up the virtual environment,
# and configures systemd to run the bot as a service.

set -euo pipefail

echo "=============================================="
echo "  Money Maker ML Trading Bot - Deployment"
echo "=============================================="
echo ""

# Step 1: Update system and install dependencies
echo "[1/14] Updating system packages..."
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git curl

# Step 2: Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[2/14] Project directory: $SCRIPT_DIR"
cd "$SCRIPT_DIR"

# Pick the best available Python interpreter on the VM
if command -v python3.11 >/dev/null 2>&1; then
    PYTHON_BIN="python3.11"
else
    PYTHON_BIN="python3"
fi
echo "Using interpreter: $PYTHON_BIN"

# Step 3: Create virtual environment
echo "[3/14] Creating virtual environment..."
"$PYTHON_BIN" -m venv .venv

# Step 4: Activate virtual environment
echo "[4/14] Activating virtual environment..."
source .venv/bin/activate

# Step 5: Install Python dependencies
echo "[5/14] Installing Python dependencies..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Step 6: Create state directory
echo "[6/14] Creating state directory..."
mkdir -p state/

# Create .env from tuned template for first run
if [[ ! -f .env ]]; then
    if [[ -f .env.example ]]; then
        cp .env.example .env
        echo "Created .env from .env.example (balanced profile)."
    else
        echo "Warning: .env.example not found; continuing without .env bootstrap."
    fi
fi

# Step 7: Ask for Linux username
echo ""
echo "[7/14] Enter your Linux username (this will run the bot):"
read -p "Username: " LINUX_USER

# Validate username exists
if ! id "$LINUX_USER" &>/dev/null; then
    echo "Error: User '$LINUX_USER' does not exist."
    exit 1
fi

# Step 8: Render service template with user/workdir/python path
echo "[8/14] Configuring systemd service for user '$LINUX_USER'..."
PYTHON_PATH="$SCRIPT_DIR/.venv/bin/python"
sed \
    -e "s|__LINUX_USER__|$LINUX_USER|g" \
    -e "s|__WORKDIR__|$SCRIPT_DIR|g" \
    -e "s|__PYTHON_BIN__|$PYTHON_PATH|g" \
    trading_bot.service > /tmp/trading_bot.service

if grep -q "__LINUX_USER__\|__WORKDIR__\|__PYTHON_BIN__" /tmp/trading_bot.service; then
    echo "Error: Failed to render systemd service template."
    exit 1
fi

# Step 9: Copy service file to systemd
echo "[9/14] Installing systemd service..."
sudo cp /tmp/trading_bot.service /etc/systemd/system/trading_bot.service

# Step 10: Reload systemd daemon
echo "[10/14] Reloading systemd daemon..."
sudo systemctl daemon-reload

# Step 11: Enable service to start on boot
echo "[11/14] Enabling trading_bot service..."
sudo systemctl enable trading_bot

# Step 12: Start the service
echo "[12/14] Starting trading_bot service..."
sudo systemctl start trading_bot

# Step 13: Show service status
echo "[13/14] Checking service status..."
sleep 2
sudo systemctl status trading_bot --no-pager || true

# Step 14: Completion message
echo ""
echo "=============================================="
echo "  Deployment Complete!"
echo "=============================================="
echo ""
echo "[14/14] Bot is running. Monitor with: bash monitor.sh"
echo ""
echo "Useful commands:"
echo "  - View logs:    journalctl -u trading_bot -f"
echo "  - Stop bot:     sudo systemctl stop trading_bot"
echo "  - Restart bot:  sudo systemctl restart trading_bot"
echo "  - Disable bot:  sudo systemctl disable trading_bot"
echo ""
echo "Don't forget to configure your .env file with API keys!"
echo ""
