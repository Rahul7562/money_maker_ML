# agents/data_agent.py
# Responsibility: Fetch market data and account balances from Binance.
# TASK 4: Async fetching with aiohttp + data quality checks

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

import aiohttp
import pandas as pd
from binance.client import Client

from config import (
    BINANCE_API_KEY,
    BINANCE_API_SECRET,
    CANDLES_LIMIT,
    EXCLUDED_SYMBOL_KEYWORDS,
    MARKET_MODE,
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

# Binance API endpoints
BINANCE_SPOT_KLINES = "https://api.binance.com/api/v3/klines"
BINANCE_FUTURES_KLINES = "https://fapi.binance.com/fapi/v1/klines"


class DataAgent:
    """Fetches market data and builds a robust multi-coin trading universe."""

    def __init__(self):
        self.client = Client(
            BINANCE_API_KEY,
            BINANCE_API_SECRET,
            requests_params={"timeout": 12},
        )
        self.market_mode = MARKET_MODE
        self._session: Optional[aiohttp.ClientSession] = None
        
        logger.info(
            "DataAgent ready -> mode=%s interval=%s quote=%s max_symbols=%s",
            self.market_mode,
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
            if self.market_mode == "FUTURES":
                exchange_info = self.client.futures_exchange_info()
                tickers = {item["symbol"]: item for item in self.client.futures_ticker()}
            else:
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
            if self.market_mode == "SPOT" and not meta.get("isSpotTradingAllowed", False):
                continue
            if self.market_mode == "FUTURES" and meta.get("contractType") not in {"PERPETUAL", None}:
                continue
            if not symbol.endswith(QUOTE_ASSET):
                continue
            # FIX 2: Filter where base_asset == QUOTE_ASSET (e.g. USDTUSDT, USDCUSDT)
            if base_asset == QUOTE_ASSET:
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

    async def _fetch_candles_async(
        self, 
        session: aiohttp.ClientSession, 
        symbol: str, 
        limit: int
    ) -> Optional[pd.DataFrame]:
        """
        TASK 4: Fetch candles for one symbol asynchronously using aiohttp.
        """
        url = BINANCE_FUTURES_KLINES if self.market_mode == "FUTURES" else BINANCE_SPOT_KLINES
        params = {
            "symbol": symbol,
            "interval": TRADE_INTERVAL,
            "limit": limit,
        }
        
        try:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as response:
                if response.status != 200:
                    logger.warning("HTTP %d fetching %s", response.status, symbol)
                    return None
                
                raw = await response.json()
                
                if not raw:
                    logger.warning("No candles returned for %s", symbol)
                    return None
                
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
                
                # TASK 4: Drop future-timestamped candles
                now_utc = datetime.now(timezone.utc)
                df = df[df["open_time"] <= now_utc]
                
                df.dropna(inplace=True)
                df.set_index("open_time", inplace=True)
                
                # TASK 4: Data quality checks
                if not self._validate_data_quality(df, symbol, limit):
                    return None
                
                return df
                
        except asyncio.TimeoutError:
            logger.warning("Timeout fetching candles for %s", symbol)
            return None
        except Exception as exc:
            logger.warning("Failed fetching candles for %s: %s", symbol, exc)
            return None

    def _validate_data_quality(self, df: pd.DataFrame, symbol: str, expected_limit: int) -> bool:
        """
        TASK 4: Data quality checks.
        Reject if:
        - >5% NaN values
        - >10% zero-volume candles
        - <90% expected candle count
        """
        if df.empty:
            logger.warning("Data quality [%s]: Empty DataFrame", symbol)
            return False
        
        # Check candle count (must have at least 90% of expected)
        min_candles = int(expected_limit * 0.9)
        if len(df) < min_candles:
            logger.warning(
                "Data quality [%s]: Insufficient candles (%d < %d expected)",
                symbol, len(df), min_candles
            )
            return False
        
        # Check NaN ratio (max 5%)
        total_cells = df.shape[0] * df.shape[1]
        nan_count = df.isna().sum().sum()
        nan_ratio = nan_count / total_cells if total_cells > 0 else 0
        if nan_ratio > 0.05:
            logger.warning(
                "Data quality [%s]: Too many NaN values (%.1f%% > 5%%)",
                symbol, nan_ratio * 100
            )
            return False
        
        # Check zero-volume ratio (max 10%)
        zero_volume_count = (df["volume"] == 0).sum()
        zero_volume_ratio = zero_volume_count / len(df)
        if zero_volume_ratio > 0.10:
            logger.warning(
                "Data quality [%s]: Too many zero-volume candles (%.1f%% > 10%%)",
                symbol, zero_volume_ratio * 100
            )
            return False
        
        return True

    async def _get_batch_candles_async(
        self, 
        symbols: List[str], 
        limit: int
    ) -> Dict[str, pd.DataFrame]:
        """
        TASK 4: Fetch candles for multiple symbols concurrently.
        """
        async with aiohttp.ClientSession() as session:
            tasks = [
                self._fetch_candles_async(session, symbol, limit)
                for symbol in symbols
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        market_data: Dict[str, pd.DataFrame] = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, pd.DataFrame) and not result.empty:
                market_data[symbol] = result
            elif isinstance(result, Exception):
                logger.debug("Exception fetching %s: %s", symbol, result)
        
        return market_data

    def get_candles(self, symbol: str, limit: int | None = None) -> pd.DataFrame:
        """Fetch recent OHLCV candles for one symbol (synchronous fallback)."""
        fetch_limit = limit or CANDLES_LIMIT
        try:
            if self.market_mode == "FUTURES":
                raw = self.client.futures_klines(
                    symbol=symbol,
                    interval=TRADE_INTERVAL,
                    limit=fetch_limit,
                )
            else:
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
            
            # Drop future-timestamped candles
            now_utc = datetime.now(timezone.utc)
            df = df[df["open_time"] <= now_utc]
            
            df.dropna(inplace=True)
            df.set_index("open_time", inplace=True)
            return df
        except Exception as exc:
            logger.warning("Failed fetching candles for %s: %s", symbol, exc)
            raise

    def get_batch_candles(self, symbols: List[str], limit: int | None = None) -> Dict[str, pd.DataFrame]:
        """
        TASK 4: Fetch candles for many symbols using async/aiohttp.
        Falls back to sequential if async fails.
        """
        fetch_limit = limit or CANDLES_LIMIT
        
        try:
            # Use asyncio to fetch concurrently
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                market_data = loop.run_until_complete(
                    self._get_batch_candles_async(symbols, fetch_limit)
                )
            finally:
                loop.close()
            
            if market_data:
                logger.info("Fetched candles for %s/%s symbols (async)", len(market_data), len(symbols))
                return market_data
                
        except Exception as exc:
            logger.warning("Async fetch failed, falling back to sequential: %s", exc)
        
        # Fallback to sequential fetching
        market_data: Dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            try:
                market_data[symbol] = self.get_candles(symbol=symbol, limit=fetch_limit)
            except Exception:
                continue

        if not market_data:
            raise RuntimeError("No symbol candles were fetched successfully")

        logger.info("Fetched candles for %s/%s symbols (sequential)", len(market_data), len(symbols))
        return market_data

    def get_balance(self, asset: str = QUOTE_ASSET) -> float:
        """Get free balance for an asset from Binance account."""
        try:
            if self.market_mode == "FUTURES":
                balances = self.client.futures_account_balance()
                row = next((b for b in balances if b.get("asset") == asset), None)
                balance = self._to_float((row or {}).get("availableBalance", 0.0))
            else:
                info = self.client.get_asset_balance(asset=asset)
                balance = self._to_float(info.get("free") if info else 0.0)
            logger.info("Live balance %s: %.4f", asset, balance)
            return balance
        except Exception as exc:
            logger.error("Failed to fetch live balance for %s: %s", asset, exc)
            raise

    def get_current_price(self, symbol: str) -> float:
        """Get current market price for a symbol."""
        if self.market_mode == "FUTURES":
            ticker = self.client.futures_symbol_ticker(symbol=symbol)
        else:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
        return self._to_float(ticker.get("price"))

    def get_position_qty(self, symbol: str) -> float:
        """Get signed position quantity for live futures; 0 for spot."""
        if self.market_mode != "FUTURES":
            return 0.0

        rows = self.client.futures_position_information(symbol=symbol)
        if not rows:
            return 0.0
        return self._to_float(rows[0].get("positionAmt", 0.0))
