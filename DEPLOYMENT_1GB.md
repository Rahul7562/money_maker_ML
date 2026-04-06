# 🚀 Deployment Guide: 1GB RAM / 1 CPU Linux VM

## Quick Start (Recommended Method)

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/money_maker_ML.git
cd money_maker_ML

# 2. Run the low-memory deployment script
chmod +x deploy_lowmem.sh
./deploy_lowmem.sh
```

That's it! The script handles everything automatically.

---

## Manual Deployment Steps

### Step 1: System Preparation

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.11
sudo apt install -y python3.11 python3.11-venv python3-pip git
```

### Step 2: Create Swap (CRITICAL for 1GB RAM)

```bash
# Create 1GB swap file
sudo fallocate -l 1G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make persistent across reboots
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Verify swap
free -h
```

### Step 3: Clone and Setup Project

```bash
# Navigate to home or preferred directory
cd ~

# Clone repository
git clone https://github.com/yourusername/money_maker_ML.git
cd money_maker_ML

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies (without PyTorch for low memory)
pip install --upgrade pip
pip install pandas python-binance python-dotenv numpy aiohttp requests scikit-learn
```

### Step 4: Configure Environment

```bash
# Create .env file
cp .env.example .env

# Edit with nano or vim
nano .env
```

**Recommended .env for 1GB RAM:**
```env
# Binance (leave blank for paper trading)
BINANCE_API_KEY=
BINANCE_API_SECRET=

# Core mode
PAPER_TRADING=true
MARKET_MODE=SPOT
TRADE_INTERVAL=1h

# LOW MEMORY OPTIMIZATIONS
MAX_SYMBOLS_ANALYZED=15
ML_ENABLED=false
SIMULATION_ENABLED=false
WFO_ENABLED=true
WFO_MAX_SYMBOLS=2
WFO_MAX_PARAMETER_SETS=8
CANDLES_LIMIT=200

# Risk settings
MAX_OPEN_POSITIONS=3
MAX_DAILY_DRAWDOWN_PERCENT=5.0
```

### Step 5: Test Run

```bash
# Activate venv if not active
source venv/bin/activate

# Quick test run (Ctrl+C to stop after ~30 seconds)
python main.py
```

### Step 6: Setup Systemd Service

```bash
# Create service file
sudo nano /etc/systemd/system/trading_bot.service
```

Paste this content (replace `YOUR_USERNAME` and path):
```ini
[Unit]
Description=Money Maker ML Trading Bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=YOUR_USERNAME
WorkingDirectory=/home/YOUR_USERNAME/money_maker_ML
ExecStart=/home/YOUR_USERNAME/money_maker_ML/venv/bin/python main.py
Restart=always
RestartSec=30
StandardOutput=journal
StandardError=journal
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONOPTIMIZE=1

# Memory limits for 1GB VMs
MemoryMax=700M
MemoryHigh=500M
CPUWeight=50

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable trading_bot
sudo systemctl start trading_bot

# Check status
sudo systemctl status trading_bot
```

---

## Memory Optimization Tips

| Setting | Low Memory Value | Description |
|---------|------------------|-------------|
| `MAX_SYMBOLS_ANALYZED` | 10-15 | Fewer symbols = less memory |
| `ML_ENABLED` | false | PyTorch uses ~400MB RAM |
| `SIMULATION_ENABLED` | false | Saves ~50MB per cycle |
| `WFO_MAX_SYMBOLS` | 2 | Limits tuning scope |
| `CANDLES_LIMIT` | 200 | Smaller history window |
| `CORRELATION_LOOKBACK` | 30 | Smaller correlation matrix |

**Estimated Memory Usage:**
- Base bot (no ML/simulation): ~200-350MB
- With Walk-Forward Tuning: ~300-450MB
- With ML enabled: ~600-800MB (NOT recommended for 1GB VM)

---

## Monitoring Commands

```bash
# View live logs
journalctl -u trading_bot -f

# Check bot status
sudo systemctl status trading_bot

# Check memory usage
free -h
htop

# Health check endpoint
curl http://localhost:8080/health

# View last 100 log lines
journalctl -u trading_bot -n 100

# Check bot.log file
tail -f bot.log
```

---

## Troubleshooting

### Bot Keeps Crashing / OOM Killed
```bash
# Check if OOM killed it
dmesg | grep -i "killed process"

# Solution: Reduce memory usage
nano .env
# Set: MAX_SYMBOLS_ANALYZED=10
# Set: ML_ENABLED=false
# Set: SIMULATION_ENABLED=false

sudo systemctl restart trading_bot
```

### Bot Won't Start
```bash
# Check logs for errors
journalctl -u trading_bot -n 50

# Test manually
cd ~/money_maker_ML
source venv/bin/activate
python main.py
```

### API Connection Issues
```bash
# Check network
ping api.binance.com

# Test API (in Python)
source venv/bin/activate
python -c "from binance.client import Client; c=Client('',''); print(c.get_server_time())"
```

### High CPU Usage
```bash
# This is normal during:
# - Walk-forward tuning (every 12 cycles)
# - First startup
# - Market data fetching

# If persistent, reduce:
nano .env
# Set: WFO_ENABLED=false
# Or: WFO_REOPTIMIZE_EVERY_CYCLES=24
```

---

## Useful Scripts

### Restart Bot
```bash
sudo systemctl restart trading_bot
```

### Stop Bot
```bash
sudo systemctl stop trading_bot
```

### Update Code
```bash
cd ~/money_maker_ML
git pull
sudo systemctl restart trading_bot
```

### Clear State (Start Fresh)
```bash
sudo systemctl stop trading_bot
rm -rf state/*
rm bot.log
sudo systemctl start trading_bot
```

### Backup State
```bash
tar -czvf bot_backup_$(date +%Y%m%d).tar.gz state/ .env bot.log
```

---

## Cloud Provider Notes

### DigitalOcean
- Use $6/month droplet (1GB RAM, 1 vCPU)
- Enable swap during creation or use script

### AWS Lightsail  
- Use $5/month instance (1GB RAM)
- Add swap file manually

### Vultr
- Use $5/month plan (1GB RAM)
- Works well with the low-memory config

### Oracle Cloud (Free Tier)
- Use Always Free ARM instance
- Note: ARM architecture - same setup works

---

## Security Recommendations

1. **Never commit .env file** (already in .gitignore)
2. **Use read-only API keys** for paper trading
3. **Enable IP whitelist** on Binance API for live trading
4. **Keep system updated**: `sudo apt update && sudo apt upgrade`
5. **Use firewall**: `sudo ufw enable && sudo ufw allow 22 && sudo ufw allow 8080`
