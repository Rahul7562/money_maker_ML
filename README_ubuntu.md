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
