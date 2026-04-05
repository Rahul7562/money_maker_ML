# agents/data_agent.py
# Responsibility: Fetch market data and account balances from Binance.

import logging
from typing import Dict, List

import pandas as pd
from binance.client import Client

from config import (
    BINANCE_API_KEY,
    BINANCE_API_SECRET,
    CANDLES_LIMIT,
    EXCLUDED_SYMBOL_KEYWORDS,
    MAX_SYMBOLS_ANALYZED,
    MIN_24H_QUOTE_VOLUME,
    QUOTE_ASSET,
    TRADE_INTERVAL,
)

logger = logging.getLogger("DataAgent")

EXCLUDED_BASE_ASSETS = {
    "USDC",
    "USDP",
    "FDUSD",
    "TUSD",
    "BUSD",
    "DAI",
    "USDE",
    "PYUSD",
    "XAUT",
}


class DataAgent:
    """Fetches market data and builds a robust multi-coin trading universe."""

    def __init__(self):
        self.client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
        logger.info(
            "DataAgent ready -> interval=%s quote=%s max_symbols=%s",
            TRADE_INTERVAL,
            QUOTE_ASSET,
            MAX_SYMBOLS_ANALYZED,
        )

    def _to_float(self, value, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def get_tradeable_symbols(self) -> List[str]:
        """Return top liquid spot symbols (e.g. BTCUSDT, ETHUSDT)."""
        try:
            exchange_info = self.client.get_exchange_info()
            tickers = {item["symbol"]: item for item in self.client.get_ticker()}
        except Exception as exc:
            logger.error("Failed to fetch exchange universe: %s", exc)
            raise

        symbols_with_volume = []
        for meta in exchange_info.get("symbols", []):
            symbol = meta.get("symbol", "")
            base_asset = meta.get("baseAsset", "")

            if meta.get("status") != "TRADING":
                continue
            if meta.get("quoteAsset") != QUOTE_ASSET:
                continue
            if not meta.get("isSpotTradingAllowed", False):
                continue
            if not symbol.endswith(QUOTE_ASSET):
                continue
            if base_asset in EXCLUDED_BASE_ASSETS:
                continue
            if len(base_asset) < 3:
                continue
            if "USD" in base_asset:
                continue
            if symbol.startswith("1000"):
                continue
            if any(base_asset.endswith(k) or k in base_asset for k in EXCLUDED_SYMBOL_KEYWORDS):
                continue

            quote_volume = self._to_float(tickers.get(symbol, {}).get("quoteVolume"))
            if quote_volume < MIN_24H_QUOTE_VOLUME:
                continue

            symbols_with_volume.append((symbol, quote_volume))

        symbols_with_volume.sort(key=lambda item: item[1], reverse=True)
        selected = [item[0] for item in symbols_with_volume[:MAX_SYMBOLS_ANALYZED]]

        if not selected:
            raise RuntimeError("No tradeable symbols matched filters; relax volume or exclusions")

        logger.info(
            "Selected %s symbols for scanning (top by 24h quote volume)",
            len(selected),
        )
        return selected

    def get_candles(self, symbol: str, limit: int | None = None) -> pd.DataFrame:
        """Fetch recent OHLCV candles for one symbol."""
        fetch_limit = limit or CANDLES_LIMIT
        try:
            raw = self.client.get_klines(
                symbol=symbol,
                interval=TRADE_INTERVAL,
                limit=fetch_limit,
            )
            if not raw:
                raise RuntimeError(f"No candles returned for {symbol}")

            df = pd.DataFrame(
                raw,
                columns=[
                    "open_time",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "close_time",
                    "quote_volume",
                    "trades",
                    "taker_buy_base",
                    "taker_buy_quote",
                    "ignore",
                ],
            )
            df = df[["open_time", "open", "high", "low", "close", "volume"]].copy()
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            df.dropna(inplace=True)
            df.set_index("open_time", inplace=True)
            return df
        except Exception as exc:
            logger.warning("Failed fetching candles for %s: %s", symbol, exc)
            raise

    def get_batch_candles(self, symbols: List[str], limit: int | None = None) -> Dict[str, pd.DataFrame]:
        """Fetch candles for many symbols, skipping temporary API failures."""
        market_data: Dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            try:
                market_data[symbol] = self.get_candles(symbol=symbol, limit=limit)
            except Exception:
                continue

        if not market_data:
            raise RuntimeError("No symbol candles were fetched successfully")

        logger.info("Fetched candles for %s/%s symbols", len(market_data), len(symbols))
        return market_data

    def get_balance(self, asset: str = QUOTE_ASSET) -> float:
        """Get free balance for an asset from Binance account."""
        try:
            info = self.client.get_asset_balance(asset=asset)
            balance = self._to_float(info.get("free") if info else 0.0)
            logger.info("Live balance %s: %.4f", asset, balance)
            return balance
        except Exception as exc:
            logger.error("Failed to fetch live balance for %s: %s", asset, exc)
            raise

    def get_current_price(self, symbol: str) -> float:
        """Get current market price for a symbol."""
        ticker = self.client.get_symbol_ticker(symbol=symbol)
        return self._to_float(ticker.get("price"))
