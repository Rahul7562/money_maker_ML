#!/usr/bin/env bash
# monitor.sh - Monitor the Money Maker ML Trading Bot
# Shows service status, recent logs, health check, and uptime

set -euo pipefail

echo "=============================================="
echo "  Money Maker ML Trading Bot - Monitor"
echo "=============================================="
echo ""

# Step 1: Service status
echo "=== Service Status ==="
sudo systemctl status trading_bot --no-pager
echo ""

# Step 2: Recent log lines
echo "=== Last 50 log lines ==="
journalctl -u trading_bot -n 50 --no-pager
echo ""

# Step 3: Health check
echo "=== Health Check ==="
if command -v python3 >/dev/null 2>&1; then
	curl -s http://localhost:8080/health | python3 -m json.tool 2>/dev/null || echo "Health check unavailable (bot may be starting up)"
else
	curl -s http://localhost:8080/health || echo "Health check unavailable (bot may be starting up)"
fi
echo ""

# Step 4: Bot running since
echo "=== Bot running since ==="
systemctl show trading_bot --property=ActiveEnterTimestamp
echo ""
