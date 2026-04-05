# config.py - Central config shared by all agents.
import os
from dotenv import load_dotenv

load_dotenv()


def _as_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# --- Binance ---
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")

# --- Trading Universe ---
QUOTE_ASSET = os.getenv("QUOTE_ASSET", "USDT")
TRADE_INTERVAL = os.getenv("TRADE_INTERVAL", "1h")
CANDLES_LIMIT = int(os.getenv("CANDLES_LIMIT", "250"))
MAX_SYMBOLS_ANALYZED = int(os.getenv("MAX_SYMBOLS_ANALYZED", "35"))
MIN_24H_QUOTE_VOLUME = float(os.getenv("MIN_24H_QUOTE_VOLUME", "20000000"))

# Tokens with these substrings are excluded from scanning to avoid
# leveraged and non-standard products.
EXCLUDED_SYMBOL_KEYWORDS = (
    "UP",
    "DOWN",
    "BULL",
    "BEAR",
    "EUR",
    "BRL",
    "TRY",
)

# --- Risk ---
MAX_TRADE_PERCENT = float(os.getenv("MAX_TRADE_PERCENT", "0.14"))
MIN_TRADE_PERCENT = float(os.getenv("MIN_TRADE_PERCENT", "0.04"))
MIN_BALANCE_USDT = float(os.getenv("MIN_BALANCE_USDT", "20"))
STOP_LOSS_PERCENT = float(os.getenv("STOP_LOSS_PERCENT", "0.018"))
TAKE_PROFIT_PERCENT = float(os.getenv("TAKE_PROFIT_PERCENT", "0.045"))
TRAILING_STOP_MULTIPLIER = float(os.getenv("TRAILING_STOP_MULTIPLIER", "1.2"))
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.60"))

# --- Analysis ---
RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
RSI_OVERSOLD = float(os.getenv("RSI_OVERSOLD", "32"))
RSI_OVERBOUGHT = float(os.getenv("RSI_OVERBOUGHT", "68"))
MACD_FAST = int(os.getenv("MACD_FAST", "12"))
MACD_SLOW = int(os.getenv("MACD_SLOW", "26"))
MACD_SIGNAL = int(os.getenv("MACD_SIGNAL", "9"))
EMA_FAST = int(os.getenv("EMA_FAST", "20"))
EMA_SLOW = int(os.getenv("EMA_SLOW", "50"))
ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))
MIN_SIGNAL_SCORE = float(os.getenv("MIN_SIGNAL_SCORE", "0.56"))

# --- Internal Simulation ---
SIMULATION_ENABLED = _as_bool("SIMULATION_ENABLED", True)
SIMULATION_CANDLES = int(os.getenv("SIMULATION_CANDLES", "350"))
SIMULATION_WARMUP = int(os.getenv("SIMULATION_WARMUP", "80"))
SIMULATION_FEE_RATE = float(os.getenv("SIMULATION_FEE_RATE", "0.001"))

# --- Mode ---
PAPER_TRADING = _as_bool("PAPER_TRADING", True)
PAPER_STARTING_USDT = float(os.getenv("PAPER_STARTING_USDT", "500"))

# --- Logging ---
LOG_FILE = os.getenv("LOG_FILE", "bot.log")
