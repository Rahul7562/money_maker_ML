#!/bin/bash
# deploy_lowmem.sh - Optimized deployment for 1GB RAM / 1 CPU Linux VMs
# This script installs with memory-optimized settings for constrained environments.

set -e

echo "=============================================="
echo "  Money Maker ML - Low Memory Deployment"
echo "  (Optimized for 1GB RAM / 1 CPU)"
echo "=============================================="
echo ""

# Step 1: System prep
echo "[1/12] Installing system packages..."
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3-pip git

# Step 2: Create swap (essential for 1GB RAM)
echo "[2/12] Setting up swap space (1GB)..."
if [ ! -f /swapfile ]; then
    sudo fallocate -l 1G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
    echo "Swap created and enabled."
else
    echo "Swap already exists."
fi

# Step 3: Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[3/12] Project directory: $SCRIPT_DIR"
cd "$SCRIPT_DIR"

# Step 4: Create virtual environment
echo "[4/12] Creating Python 3.11 virtual environment..."
python3.11 -m venv venv

# Step 5: Activate venv
echo "[5/12] Activating virtual environment..."
source venv/bin/activate

# Step 6: Install base deps (without PyTorch initially)
echo "[6/12] Installing Python dependencies (no ML)..."
pip install --upgrade pip
pip install pandas python-binance python-dotenv numpy aiohttp requests scikit-learn

# Step 7: Optional PyTorch (only if >1.5GB RAM available)
TOTAL_MEM=$(free -m | awk '/^Mem:/{print $2}')
echo "[7/12] Detected RAM: ${TOTAL_MEM}MB"

if [ "$TOTAL_MEM" -gt 1500 ]; then
    echo "Installing PyTorch CPU-only for ML features..."
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    ML_AVAILABLE="true"
else
    echo "Skipping PyTorch (insufficient RAM). ML features disabled."
    ML_AVAILABLE="false"
fi

# Step 8: Create state directory
echo "[8/12] Creating state directory..."
mkdir -p state/

# Step 9: Create optimized .env if not exists
if [ ! -f .env ]; then
    echo "[9/12] Creating optimized .env for low-memory operation..."
    cat > .env << EOF
# Binance credentials (required for LIVE mode only)
BINANCE_API_KEY=
BINANCE_API_SECRET=

# Core mode
PAPER_TRADING=true
MARKET_MODE=SPOT
TRADE_INTERVAL=1h

# Low-memory optimizations
MAX_SYMBOLS_ANALYZED=15
ML_ENABLED=false
SIMULATION_ENABLED=false
WFO_ENABLED=true
WFO_MAX_SYMBOLS=2
WFO_MAX_PARAMETER_SETS=8
CORRELATION_LOOKBACK=30
CANDLES_LIMIT=200

# Risk settings
MAX_OPEN_POSITIONS=3
MAX_DAILY_DRAWDOWN_PERCENT=5.0
MIN_SIGNAL_SCORE=0.58
EOF
    echo "Created .env with low-memory settings."
else
    echo "[9/12] .env already exists, skipping."
fi

# Step 10: Get username and configure service
echo ""
echo "[10/12] Enter your Linux username (this will run the bot):"
read -p "Username: " LINUX_USER

if ! id "$LINUX_USER" &>/dev/null; then
    echo "Error: User '$LINUX_USER' does not exist."
    exit 1
fi

# Step 11: Configure systemd service with memory limits
echo "[11/12] Configuring systemd service with memory limits..."
cat > /tmp/trading_bot.service << EOF
[Unit]
Description=Money Maker ML Trading Bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$LINUX_USER
WorkingDirectory=$SCRIPT_DIR
ExecStart=$SCRIPT_DIR/venv/bin/python main.py
Restart=always
RestartSec=30
StandardOutput=journal
StandardError=journal
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONOPTIMIZE=1

# Memory limits for low-RAM VMs
MemoryMax=700M
MemoryHigh=500M

# CPU limits (nice to other processes)
CPUWeight=50

[Install]
WantedBy=multi-user.target
EOF

sudo cp /tmp/trading_bot.service /etc/systemd/system/trading_bot.service
sudo systemctl daemon-reload
sudo systemctl enable trading_bot

# Step 12: Start service
echo "[12/12] Starting trading_bot service..."
sudo systemctl start trading_bot
sleep 3
sudo systemctl status trading_bot --no-pager || true

echo ""
echo "=============================================="
echo "  Deployment Complete! (Low Memory Mode)"
echo "=============================================="
echo ""
echo "Useful commands:"
echo "  - View logs:    journalctl -u trading_bot -f"
echo "  - Stop bot:     sudo systemctl stop trading_bot"
echo "  - Restart bot:  sudo systemctl restart trading_bot"
echo "  - Check memory: free -h"
echo "  - Health check: curl http://localhost:8080/health"
echo ""
echo "IMPORTANT for 1GB RAM VMs:"
echo "  - ML features are DISABLED (need 2GB+ RAM)"
echo "  - Simulation is DISABLED to save memory"
echo "  - Max 15 symbols scanned (reduce in .env if needed)"
echo ""
