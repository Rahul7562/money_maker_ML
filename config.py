# config.py - Central config shared by all agents.
import os
from dotenv import load_dotenv

load_dotenv()


def _as_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_int_list(name: str, default_csv: str) -> list[int]:
    raw = os.getenv(name, default_csv)
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            values.append(int(token))
        except ValueError:
            continue
    return values


def _as_float_list(name: str, default_csv: str) -> list[float]:
    raw = os.getenv(name, default_csv)
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            values.append(float(token))
        except ValueError:
            continue
    return values


# --- Binance ---
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")

# --- Trading Universe ---
MARKET_MODE = os.getenv("MARKET_MODE", "SPOT").strip().upper()  # SPOT | FUTURES
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
MAX_OPEN_POSITIONS = int(os.getenv("MAX_OPEN_POSITIONS", "4"))
MAX_DAILY_DRAWDOWN_PERCENT = float(os.getenv("MAX_DAILY_DRAWDOWN_PERCENT", "5.0"))

# --- Futures (optional) ---
FUTURES_ENABLE_SHORTS = _as_bool("FUTURES_ENABLE_SHORTS", True)
FUTURES_DEFAULT_LEVERAGE = int(os.getenv("FUTURES_DEFAULT_LEVERAGE", "2"))
FUTURES_MAX_LEVERAGE = int(os.getenv("FUTURES_MAX_LEVERAGE", "3"))
FUTURES_MAINT_MARGIN_BUFFER = float(os.getenv("FUTURES_MAINT_MARGIN_BUFFER", "0.35"))

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

# --- Regime Filter ---
REGIME_ADX_PERIOD = int(os.getenv("REGIME_ADX_PERIOD", "14"))
REGIME_TREND_ADX_THRESHOLD = float(os.getenv("REGIME_TREND_ADX_THRESHOLD", "24"))
REGIME_SIDEWAYS_ATR_THRESHOLD = float(os.getenv("REGIME_SIDEWAYS_ATR_THRESHOLD", "0.015"))
ALLOW_SIDEWAYS_TRADES = _as_bool("ALLOW_SIDEWAYS_TRADES", False)

# --- Walk-Forward Tuning ---
WFO_ENABLED = _as_bool("WFO_ENABLED", True)
WFO_REOPTIMIZE_EVERY_CYCLES = int(os.getenv("WFO_REOPTIMIZE_EVERY_CYCLES", "8"))
WFO_MAX_SYMBOLS = int(os.getenv("WFO_MAX_SYMBOLS", "4"))
WFO_MAX_PARAMETER_SETS = int(os.getenv("WFO_MAX_PARAMETER_SETS", "16"))
WFO_SIGNAL_STEP = int(os.getenv("WFO_SIGNAL_STEP", "3"))
WFO_TIME_BUDGET_SECONDS = float(os.getenv("WFO_TIME_BUDGET_SECONDS", "8"))
WFO_TRAIN_CANDLES = int(os.getenv("WFO_TRAIN_CANDLES", "160"))
WFO_TEST_CANDLES = int(os.getenv("WFO_TEST_CANDLES", "60"))
WFO_STEP_CANDLES = int(os.getenv("WFO_STEP_CANDLES", "30"))
WFO_MIN_WIN_RATE = float(os.getenv("WFO_MIN_WIN_RATE", "0.45"))
WFO_MIN_AVG_TEST_RETURN = float(os.getenv("WFO_MIN_AVG_TEST_RETURN", "0.0"))

WFO_EMA_FAST_CANDIDATES = _as_int_list("WFO_EMA_FAST_CANDIDATES", "12,20,26")
WFO_EMA_SLOW_CANDIDATES = _as_int_list("WFO_EMA_SLOW_CANDIDATES", "40,50,70")
WFO_MIN_SCORE_CANDIDATES = _as_float_list("WFO_MIN_SCORE_CANDIDATES", "0.54,0.58,0.62")
WFO_RSI_OVERSOLD_CANDIDATES = _as_float_list("WFO_RSI_OVERSOLD_CANDIDATES", "28,32,36")
WFO_RSI_OVERBOUGHT_CANDIDATES = _as_float_list("WFO_RSI_OVERBOUGHT_CANDIDATES", "64,68,72")

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
