# Money Maker ML — Autonomous Crypto Trading Bot

An intelligent cryptocurrency trading bot that combines technical analysis, machine learning (LSTM+Attention), and multi-agent architecture for automated trading on Binance.

## Architecture

```
                          ┌─────────────────────────────────────────────────────────────┐
                          │                         User                                │
                          └───────────────────────────┬─────────────────────────────────┘
                                                      │
                                                      ▼
                          ┌─────────────────────────────────────────────────────────────┐
                          │                   main.py (Orchestrator)                    │
                          │         Coordinates all agents, runs trading loop           │
                          └───────────────────────────┬─────────────────────────────────┘
                                                      │
          ┌───────────────────────────────────────────┼───────────────────────────────────────────┐
          │                                           │                                           │
          ▼                                           ▼                                           ▼
┌─────────────────────┐               ┌─────────────────────────┐               ┌─────────────────────┐
│     DataAgent       │               │     AnalysisAgent       │               │      MLAgent        │
│  ─────────────────  │               │  ─────────────────────  │               │  ─────────────────  │
│  Binance API        │               │  RSI / MACD / VWAP      │               │  PyTorch LSTM       │
│  Async fetching     │               │  Bollinger Bands        │               │  + Attention        │
│  Data validation    │               │  EMA crossover          │               │  Price prediction   │
└─────────────────────┘               └─────────────────────────┘               └─────────────────────┘
          │                                           │                                           │
          │                                           │                                           │
          ▼                                           ▼                                           ▼
┌─────────────────────┐               ┌─────────────────────────┐               ┌─────────────────────┐
│    RegimeAgent      │               │    SentimentAgent       │               │  CorrelationAgent   │
│  ─────────────────  │               │  ─────────────────────  │               │  ─────────────────  │
│  ADX trend detect   │               │  Fear & Greed API       │               │  Pearson filter     │
│  BULL / BEAR /      │               │  Score modifier         │               │  Avoid correlated   │
│  SIDEWAYS           │               │  Extreme detection      │               │  positions          │
└─────────────────────┘               └─────────────────────────┘               └─────────────────────┘
          │                                           │                                           │
          └───────────────────────────────────────────┼───────────────────────────────────────────┘
                                                      │
                                                      ▼
                          ┌─────────────────────────────────────────────────────────────┐
                          │                       RiskAgent                             │
                          │  ─────────────────────────────────────────────────────────  │
                          │  Kelly criterion sizing    │    ATR-based stops             │
                          │  Cooldown enforcement      │    Position limits             │
                          └───────────────────────────┬─────────────────────────────────┘
                                                      │
                                                      ▼
                          ┌─────────────────────────────────────────────────────────────┐
                          │                    ExecutionAgent                           │
                          │  ─────────────────────────────────────────────────────────  │
                          │  Paper trading (simulated) │  Slippage simulation           │
                          │  Live trading (Binance)    │  LIMIT order support           │
                          └───────────────────────────┬─────────────────────────────────┘
                                                      │
          ┌───────────────────────────────────────────┼───────────────────────────────────────────┐
          │                                           │                                           │
          ▼                                           ▼                                           ▼
┌─────────────────────┐               ┌─────────────────────────┐               ┌─────────────────────┐
│  PortfolioAgent     │               │   PerformanceAgent      │               │    TuningAgent      │
│  ─────────────────  │               │  ─────────────────────  │               │  ─────────────────  │
│  State persistence  │               │  CSV logging            │               │  Walk-Forward       │
│  JSON portfolio     │               │  Win rate tracking      │               │  Optimization       │
│  Position tracking  │               │  Cooldown management    │               │  Auto-tune params   │
└─────────────────────┘               └─────────────────────────┘               └─────────────────────┘
```

## Prerequisites

- **Ubuntu 24 LTS** (or compatible Linux distribution)
- **Python 3.11** (required for PyTorch compatibility)
- **2GB RAM minimum** (4GB recommended for ML training)
- **Binance account** with API key and secret (for live trading)
- **Internet connection** for market data and Fear & Greed API

## Quick Setup

```bash
# Clone the repository
git clone https://github.com/Rahul7562/money_maker_ML.git

# Navigate to project directory
cd money_maker_ML

# Run the deployment script
bash deploy_linux.sh
```

The deployment script will:
1. Install Python 3.11 and dependencies
2. Create a virtual environment
3. Install all required packages
4. Set up systemd service for auto-start
5. Start the bot

## .env Configuration

Create a `.env` file in the project root with your settings. Copy from `.env.example`:

```bash
cp .env.example .env
nano .env
```

### Configuration Variables

| Variable | Default | Description |
|----------|---------|-------------|
| **Binance API** | | |
| `BINANCE_API_KEY` | *(empty)* | Your Binance API key (required for live trading) |
| `BINANCE_API_SECRET` | *(empty)* | Your Binance API secret (required for live trading) |
| **Trading Mode** | | |
| `PAPER_TRADING` | `true` | Enable paper trading mode (simulated trades) |
| `MARKET_MODE` | `SPOT` | Market type: `SPOT` or `FUTURES` |
| `TRADE_INTERVAL` | `1h` | Candle interval: `1m`, `5m`, `15m`, `1h`, `4h`, `1d` |
| `QUOTE_ASSET` | `USDT` | Quote currency for trading pairs |
| **Paper Trading** | | |
| `PAPER_STARTING_USDT` | `500` | Starting balance for paper trading |
| `PAPER_ORDER_TYPE` | `MARKET` | Order type: `MARKET` or `LIMIT` |
| `SLIPPAGE_STD` | `0.0003` | Slippage standard deviation (0.03%) |
| **Universe Selection** | | |
| `MAX_SYMBOLS_ANALYZED` | `35` | Maximum symbols to analyze per cycle |
| `MIN_24H_QUOTE_VOLUME` | `20000000` | Minimum 24h volume in USDT |
| `CANDLES_LIMIT` | `250` | Number of candles to fetch per symbol |
| **Risk Management** | | |
| `MAX_TRADE_PERCENT` | `0.14` | Maximum trade size (14% of balance) |
| `MIN_TRADE_PERCENT` | `0.04` | Minimum trade size (4% of balance) |
| `MIN_BALANCE_USDT` | `20` | Minimum balance to continue trading |
| `STOP_LOSS_PERCENT` | `0.018` | Default stop loss (1.8%) |
| `TAKE_PROFIT_PERCENT` | `0.045` | Default take profit (4.5%) |
| `TRAILING_STOP_MULTIPLIER` | `1.2` | Trailing stop multiplier |
| `MAX_OPEN_POSITIONS` | `4` | Maximum concurrent positions |
| `MAX_DAILY_DRAWDOWN_PERCENT` | `5.0` | Daily drawdown limit (halts new trades) |
| **Futures (Optional)** | | |
| `FUTURES_ENABLE_SHORTS` | `true` | Allow short positions in futures mode |
| `FUTURES_DEFAULT_LEVERAGE` | `2` | Default leverage for futures |
| `FUTURES_MAX_LEVERAGE` | `3` | Maximum allowed leverage |
| `FUTURES_MAINT_MARGIN_BUFFER` | `0.35` | Maintenance margin safety buffer |
| **Technical Analysis** | | |
| `RSI_PERIOD` | `14` | RSI calculation period |
| `RSI_OVERSOLD` | `32` | RSI oversold threshold |
| `RSI_OVERBOUGHT` | `68` | RSI overbought threshold |
| `MACD_FAST` | `12` | MACD fast EMA period |
| `MACD_SLOW` | `26` | MACD slow EMA period |
| `MACD_SIGNAL` | `9` | MACD signal line period |
| `EMA_FAST` | `20` | Fast EMA period for trend |
| `EMA_SLOW` | `50` | Slow EMA period for trend |
| `ATR_PERIOD` | `14` | ATR period for volatility |
| `MIN_SIGNAL_SCORE` | `0.56` | Minimum signal score to trade |
| `MIN_CONFIDENCE` | `0.60` | Minimum confidence threshold |
| **Regime Detection** | | |
| `REGIME_ADX_PERIOD` | `14` | ADX period for regime detection |
| `REGIME_TREND_ADX_THRESHOLD` | `24` | ADX threshold for trending market |
| `REGIME_SIDEWAYS_ATR_THRESHOLD` | `0.015` | ATR threshold for sideways detection |
| `ALLOW_SIDEWAYS_TRADES` | `true` | Allow trades in sideways markets |
| `SIDEWAYS_MIN_SCORE` | `0.68` | Higher score required for sideways trades |
| **Walk-Forward Optimization** | | |
| `WFO_ENABLED` | `true` | Enable walk-forward optimization |
| `WFO_REOPTIMIZE_EVERY_CYCLES` | `8` | Cycles between re-optimization |
| `WFO_MAX_SYMBOLS` | `4` | Max symbols to tune per cycle |
| `WFO_MAX_PARAMETER_SETS` | `16` | Parameter combinations to test |
| `WFO_TRAIN_CANDLES` | `160` | Training window size |
| `WFO_TEST_CANDLES` | `60` | Test window size |
| `WFO_MIN_WIN_RATE` | `0.45` | Minimum win rate for valid params |
| **Machine Learning** | | |
| `ML_ENABLED` | `true` | Enable ML predictions |
| `ML_WEIGHT` | `0.4` | ML influence on final score (0-1) |
| `ML_MIN_CONFIDENCE` | `0.60` | Minimum ML confidence to trade |
| `ML_SEQUENCE_LENGTH` | `60` | Input sequence length for LSTM |
| `ML_RETRAIN_EVERY_N_TUNING_CYCLES` | `3` | Retrain ML every N WFO cycles |
| `ML_EARLY_STOPPING_PATIENCE` | `10` | Early stopping epochs |
| **Sentiment Analysis** | | |
| `SENTIMENT_ENABLED` | `true` | Enable Fear & Greed sentiment |
| `SENTIMENT_EXTREME_FEAR_THRESHOLD` | `20` | Extreme fear threshold (boost buys) |
| `SENTIMENT_EXTREME_GREED_THRESHOLD` | `80` | Extreme greed threshold (boost sells) |
| `SENTIMENT_CACHE_HOURS` | `4` | Hours to cache sentiment data |
| **Correlation Filter** | | |
| `CORRELATION_FILTER_ENABLED` | `true` | Enable correlation filtering |
| `CORRELATION_THRESHOLD` | `0.85` | Max correlation between positions |
| `CORRELATION_LOOKBACK` | `50` | Candles for correlation calculation |
| **Additional Risk** | | |
| `ATR_STOP_MULTIPLIER` | `1.5` | ATR multiplier for dynamic stops |
| `SYMBOL_COOLDOWN_HOURS` | `24` | Cooldown after consecutive losses |
| `KELLY_ENABLED` | `true` | Use Kelly criterion for sizing |
| `KELLY_MIN_TRADES` | `20` | Minimum trades before Kelly applies |
| `MIN_SCORE_FOR_SLOT_2_PLUS` | `0.63` | Higher score for 2nd+ positions |
| **Multi-Timeframe** | | |
| `MTF_ENABLED` | `false` | Enable multi-timeframe analysis |
| `MTF_HIGHER_INTERVAL` | `4h` | Higher timeframe to check |
| **Simulation** | | |
| `SIMULATION_ENABLED` | `true` | Run internal simulation each cycle |
| `SIMULATION_CANDLES` | `350` | Candles for simulation backtest |
| `SIMULATION_WARMUP` | `80` | Warmup candles before signals |
| `SIMULATION_FEE_RATE` | `0.001` | Simulated trading fee (0.1%) |
| **Logging** | | |
| `LOG_FILE` | `bot.log` | Log file path |
| `LOG_ROTATION_MB` | `50` | Max log file size before rotation |
| `LOG_MAX_BACKUPS` | `5` | Number of log backups to keep |
| `STATE_DIR` | `state/` | Directory for persistent state |
| **Notifications** | | |
| `TELEGRAM_ENABLED` | `false` | Enable Telegram notifications |
| `TELEGRAM_BOT_TOKEN` | *(empty)* | Telegram bot token |
| `TELEGRAM_CHAT_ID` | *(empty)* | Telegram chat ID for messages |
| `HEALTH_CHECK_PORT` | `8080` | HTTP health check port |

## Running Modes

### Paper Trading (Default, Recommended)

Paper trading simulates trades without risking real money. **Start here!**

```bash
# In .env:
PAPER_TRADING=true
PAPER_STARTING_USDT=500
```

This mode:
- Simulates realistic slippage (+/- 0.15% max)
- Tracks portfolio in `state/portfolio.json`
- Supports both MARKET and LIMIT orders
- Records all trades for analysis

### Live SPOT Trading

Trade real cryptocurrency on Binance Spot markets.

```bash
# In .env:
PAPER_TRADING=false
MARKET_MODE=SPOT
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret
```

⚠️ **Important**: Start with small amounts. The bot places real orders.

### Live FUTURES Trading

Trade on Binance Futures with leverage (higher risk).

```bash
# In .env:
PAPER_TRADING=false
MARKET_MODE=FUTURES
FUTURES_DEFAULT_LEVERAGE=2
FUTURES_MAX_LEVERAGE=3
FUTURES_ENABLE_SHORTS=true
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret
```

⚠️ **Warning**: Futures trading with leverage can result in rapid losses. Only use if you understand the risks.

## Monitoring

### Quick Status Check

```bash
bash monitor.sh
```

This shows:
- Service status (running/stopped)
- Last 50 log lines
- Health check response
- Bot uptime

### Live Log Stream

```bash
journalctl -f -u trading_bot
```

### Health Check Endpoint

```bash
curl http://localhost:8080/health | python3 -m json.tool
```

Response:
```json
{
    "status": "running",
    "cycle": 42,
    "equity": 523.45,
    "positions": 2,
    "last_cycle_utc": "2024-01-15T10:30:00+00:00",
    "uptime_hours": 24.5
}
```

### Service Control

```bash
# Check status
sudo systemctl status trading_bot

# Stop the bot
sudo systemctl stop trading_bot

# Restart the bot
sudo systemctl restart trading_bot

# Disable auto-start
sudo systemctl disable trading_bot

# Re-enable auto-start
sudo systemctl enable trading_bot
```

## ML Model

The bot uses a **PyTorch LSTM + Self-Attention** neural network for price prediction.

### Architecture

- **Input**: 60 candles × 19 features (OHLCV + indicators)
- **LSTM**: 2 layers, 128 hidden units, 0.3 dropout
- **Attention**: Multi-head self-attention (4 heads)
- **Output**: 3-class classification (UP/DOWN/NEUTRAL)

### Training

- **Automatic**: Trains on first run if no model exists
- **Retrains**: Every N WFO cycles (configurable)
- **Early stopping**: Prevents overfitting
- **CPU-only**: No GPU required

### Model Files

- **Model weights**: `state/lstm_model.pt`
- **Metadata**: `state/ml_metadata.json`

### Manual Retraining

Delete the model file to force retraining on next cycle:
```bash
rm state/lstm_model.pt
sudo systemctl restart trading_bot
```

## State Files

All persistent state is stored in the `state/` directory:

| File | Purpose |
|------|---------|
| `portfolio.json` | Current portfolio state (cash, positions, PnL) |
| `lstm_model.pt` | Trained ML model weights |
| `ml_metadata.json` | ML training metadata |
| `performance_log.csv` | Trade history for analysis |
| `cooldowns.json` | Symbol cooldown tracking |
| `wfo_params.json` | Optimized strategy parameters |

## Troubleshooting

### Bot won't start
```bash
# Check logs for errors
journalctl -u trading_bot -n 100 --no-pager

# Check if port 8080 is in use
ss -tlnp | grep 8080

# Verify Python environment
/home/YOUR_USER/money_maker_ML/venv/bin/python --version
```

### No trades executing
- Check `MIN_SIGNAL_SCORE` - may be too high
- Check `MAX_OPEN_POSITIONS` - may be at limit
- Check daily drawdown guard in logs
- Verify sufficient balance (`MIN_BALANCE_USDT`)

### ML model issues
```bash
# Force retrain
rm state/lstm_model.pt
rm state/ml_metadata.json
sudo systemctl restart trading_bot
```

### API errors
- Verify API keys are correct in `.env`
- Check Binance API restrictions
- Ensure IP is whitelisted (if enabled)

## Risk Warning

⚠️ **IMPORTANT: READ BEFORE USING**

This bot trades real money when `PAPER_TRADING=False`.

- **Past performance does not guarantee future results**
- **Cryptocurrency markets are highly volatile**
- **Never trade more than you can afford to lose**
- **Start with paper trading for minimum 30 days**
- **Test thoroughly before using real funds**
- **The developers are not responsible for any financial losses**

Trading cryptocurrencies involves substantial risk of loss and is not suitable for all investors. You should carefully consider whether trading is suitable for you in light of your circumstances, knowledge, and financial resources.

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## Support

- **Issues**: [GitHub Issues](https://github.com/Rahul7562/money_maker_ML/issues)
- **Documentation**: This README
