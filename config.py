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
    "USDC",
    "BUSD",
    "TUSD",
    "FDUSD",
    "DAI",
    "USDP",
    "USDD",
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
ALLOW_SIDEWAYS_TRADES = _as_bool("ALLOW_SIDEWAYS_TRADES", True)

# --- Walk-Forward Tuning ---
WFO_ENABLED = _as_bool("WFO_ENABLED", True)
WFO_REOPTIMIZE_EVERY_CYCLES = int(os.getenv("WFO_REOPTIMIZE_EVERY_CYCLES", "12"))
WFO_MAX_SYMBOLS = int(os.getenv("WFO_MAX_SYMBOLS", "3"))
WFO_MAX_PARAMETER_SETS = int(os.getenv("WFO_MAX_PARAMETER_SETS", "12"))
WFO_SIGNAL_STEP = int(os.getenv("WFO_SIGNAL_STEP", "5"))
WFO_TIME_BUDGET_SECONDS = float(os.getenv("WFO_TIME_BUDGET_SECONDS", "6"))
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
LOG_ROTATION_MB = int(os.getenv("LOG_ROTATION_MB", "50"))
LOG_MAX_BACKUPS = int(os.getenv("LOG_MAX_BACKUPS", "5"))

# --- State Persistence ---
STATE_DIR = os.getenv("STATE_DIR", "state/")

# --- ML (PyTorch LSTM) ---
ML_ENABLED = _as_bool("ML_ENABLED", True)
ML_WEIGHT = float(os.getenv("ML_WEIGHT", "0.4"))
ML_MIN_CONFIDENCE = float(os.getenv("ML_MIN_CONFIDENCE", "0.60"))
ML_SEQUENCE_LENGTH = int(os.getenv("ML_SEQUENCE_LENGTH", "30"))
ML_RETRAIN_EVERY_N_TUNING_CYCLES = int(os.getenv("ML_RETRAIN_EVERY_N_TUNING_CYCLES", "3"))
ML_EARLY_STOPPING_PATIENCE = int(os.getenv("ML_EARLY_STOPPING_PATIENCE", "10"))

# --- Sentiment ---
SENTIMENT_ENABLED = _as_bool("SENTIMENT_ENABLED", True)
SENTIMENT_EXTREME_FEAR_THRESHOLD = int(os.getenv("SENTIMENT_EXTREME_FEAR_THRESHOLD", "20"))
SENTIMENT_EXTREME_GREED_THRESHOLD = int(os.getenv("SENTIMENT_EXTREME_GREED_THRESHOLD", "80"))
SENTIMENT_CACHE_HOURS = int(os.getenv("SENTIMENT_CACHE_HOURS", "4"))

# --- Correlation ---
CORRELATION_FILTER_ENABLED = _as_bool("CORRELATION_FILTER_ENABLED", True)
CORRELATION_THRESHOLD = float(os.getenv("CORRELATION_THRESHOLD", "0.85"))
CORRELATION_LOOKBACK = int(os.getenv("CORRELATION_LOOKBACK", "50"))

# --- Risk (Additional) ---
ATR_STOP_MULTIPLIER = float(os.getenv("ATR_STOP_MULTIPLIER", "1.5"))
SYMBOL_COOLDOWN_HOURS = int(os.getenv("SYMBOL_COOLDOWN_HOURS", "24"))
KELLY_ENABLED = _as_bool("KELLY_ENABLED", True)
KELLY_MIN_TRADES = int(os.getenv("KELLY_MIN_TRADES", "20"))

# --- Analysis (Additional) ---
MTF_ENABLED = _as_bool("MTF_ENABLED", False)
MTF_HIGHER_INTERVAL = os.getenv("MTF_HIGHER_INTERVAL", "4h")

# --- Multi-trade ---
MIN_SCORE_FOR_SLOT_2_PLUS = float(os.getenv("MIN_SCORE_FOR_SLOT_2_PLUS", "0.63"))

# --- Sideways ---
SIDEWAYS_MIN_SCORE = float(os.getenv("SIDEWAYS_MIN_SCORE", "0.68"))

# --- Execution ---
PAPER_ORDER_TYPE = os.getenv("PAPER_ORDER_TYPE", "MARKET")
SLIPPAGE_STD = float(os.getenv("SLIPPAGE_STD", "0.0003"))

# --- Notifications ---
TELEGRAM_ENABLED = _as_bool("TELEGRAM_ENABLED", False)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
HEALTH_CHECK_PORT = int(os.getenv("HEALTH_CHECK_PORT", "8080"))
