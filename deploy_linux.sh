#!/bin/bash
# deploy_linux.sh - Deploy Money Maker ML Trading Bot on Ubuntu
# This script installs dependencies, sets up the virtual environment,
# and configures systemd to run the bot as a service.

set -e

echo "=============================================="
echo "  Money Maker ML Trading Bot - Deployment"
echo "=============================================="
echo ""

# Step 1: Update system and install dependencies
echo "[1/14] Updating system packages..."
sudo apt update && sudo apt install -y python3.11 python3.11-venv python3-pip git

# Step 2: Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[2/14] Project directory: $SCRIPT_DIR"
cd "$SCRIPT_DIR"

# Step 3: Create virtual environment
echo "[3/14] Creating Python 3.11 virtual environment..."
python3.11 -m venv venv

# Step 4: Activate virtual environment
echo "[4/14] Activating virtual environment..."
source venv/bin/activate

# Step 5: Install Python dependencies
echo "[5/14] Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Step 6: Create state directory
echo "[6/14] Creating state directory..."
mkdir -p state/

# Step 7: Ask for Linux username
echo ""
echo "[7/14] Enter your Linux username (this will run the bot):"
read -p "Username: " LINUX_USER

# Validate username exists
if ! id "$LINUX_USER" &>/dev/null; then
    echo "Error: User '$LINUX_USER' does not exist."
    exit 1
fi

# Step 8: Replace %i with username in service file
echo "[8/14] Configuring systemd service for user '$LINUX_USER'..."
sed "s/%i/$LINUX_USER/g" trading_bot.service > /tmp/trading_bot.service

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
