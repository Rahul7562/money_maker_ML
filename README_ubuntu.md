# Ubuntu VM Quick Start

## 1) Prepare project
```bash
cd /path/to/money_maker_ML
chmod +x scripts/setup_ubuntu.sh scripts/run_bot.sh
./scripts/setup_ubuntu.sh
```

## 2) Run paper bot
```bash
./scripts/run_bot.sh
```

## 3) Optional: run in background overnight
```bash
source .venv/bin/activate
nohup python main.py > bot_stdout.log 2>&1 &
```

## 4) Check logs
```bash
tail -f bot.log
```

## 5) Stop background bot
```bash
pkill -f "python main.py"
```

## Notes
- Paper + Spot mode does not require API keys.
- Live trading requires BINANCE_API_KEY and BINANCE_API_SECRET in `.env`.
- Keep `.env` private; it is ignored by `.gitignore`.

## Paper Readiness Report
- The bot now writes readiness reports automatically during cycles.
- Files:
	- `state/paper_readiness_report.json`
	- `state/paper_readiness_report.md`
- Useful knobs in `.env`:
	- `PAPER_GRADUATION_MIN_TRADES`
	- `PAPER_GRADUATION_MIN_WIN_RATE`
	- `PAPER_GRADUATION_MIN_PROFIT_FACTOR`
	- `PAPER_GRADUATION_MIN_TOTAL_PNL_USDT`
	- `PAPER_GRADUATION_MAX_DRAWDOWN`
	- `WEEKLY_REPORT_WINDOW_DAYS`
	- `WEEKLY_REPORT_EVERY_HOURS`

## Recommended Profile (Balanced)
- The Linux deploy flow now bootstraps `.env` from `.env.example` if missing.
- `.env.example` is tuned for a balanced VPS profile (good risk-adjusted throughput).
- Before live mode, review at least: `PAPER_TRADING`, `MAX_TRADE_PERCENT`, `MAX_OPEN_POSITIONS`, `MAX_DAILY_DRAWDOWN_PERCENT`, and API keys.

## Quick Profile Overrides
For stronger hardware (4GB+ RAM), you can increase:
```bash
MAX_SYMBOLS_ANALYZED=35
MAX_OPEN_POSITIONS=4
SIMULATION_ENABLED=true
```

For 1GB RAM VMs, keep:
```bash
MAX_SYMBOLS_ANALYZED=15
MAX_OPEN_POSITIONS=3
ML_ENABLED=false
SIMULATION_ENABLED=false
```
