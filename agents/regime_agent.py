# agents/regime_agent.py
# Responsibility: Detect bull/bear/sideways regimes and filter actions by regime.

from dataclasses import dataclass

import pandas as pd

from config import (
    ALLOW_SIDEWAYS_TRADES,
    FUTURES_ENABLE_SHORTS,
    MARKET_MODE,
    REGIME_ADX_PERIOD,
    REGIME_SIDEWAYS_ATR_THRESHOLD,
    REGIME_TREND_ADX_THRESHOLD,
)


@dataclass
class RegimeState:
    symbol: str
    regime: str
    adx: float
    plus_di: float
    minus_di: float
    atr_pct: float


class RegimeAgent:
    """Classifies market regime for each symbol and validates signal-regime alignment."""

    def __init__(self):
        self.mode = MARKET_MODE
        self.allow_sideways = ALLOW_SIDEWAYS_TRADES
        self.enable_futures_shorts = FUTURES_ENABLE_SHORTS

    def classify(self, symbol: str, df: pd.DataFrame) -> RegimeState:
        high = df["high"]
        low = df["low"]
        close = df["close"]

        up_move = high.diff()
        down_move = -low.diff()

        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

        tr = pd.concat(
            [
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr = tr.rolling(REGIME_ADX_PERIOD).mean().bfill()
        plus_di = 100 * (plus_dm.rolling(REGIME_ADX_PERIOD).mean() / atr.replace(0, 1e-12))
        minus_di = 100 * (minus_dm.rolling(REGIME_ADX_PERIOD).mean() / atr.replace(0, 1e-12))

        dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-12)) * 100
        adx = dx.rolling(REGIME_ADX_PERIOD).mean().bfill()
        atr_pct = (atr / close.replace(0, 1e-12)).fillna(0.0)

        latest_adx = float(adx.iloc[-1])
        latest_plus = float(plus_di.iloc[-1])
        latest_minus = float(minus_di.iloc[-1])
        latest_atr_pct = float(atr_pct.iloc[-1])

        if latest_adx >= REGIME_TREND_ADX_THRESHOLD:
            if latest_plus >= latest_minus:
                regime = "BULL"
            else:
                regime = "BEAR"
        else:
            regime = "SIDEWAYS"

        # Extremely low volatility usually behaves like range-bound market.
        if latest_atr_pct <= REGIME_SIDEWAYS_ATR_THRESHOLD:
            regime = "SIDEWAYS"

        return RegimeState(
            symbol=symbol,
            regime=regime,
            adx=round(latest_adx, 3),
            plus_di=round(latest_plus, 3),
            minus_di=round(latest_minus, 3),
            atr_pct=round(latest_atr_pct, 6),
        )

    def allows_action(self, regime: str, action: str) -> bool:
        if action == "HOLD":
            return True

        if regime == "SIDEWAYS":
            return self.allow_sideways

        if regime == "BULL":
            # Allow BUY in bull regime, also allow SELL to close existing positions
            if action == "BUY":
                return True
            # Allow SELL for closing positions (important for SPOT to exit longs)
            if action == "SELL":
                return True
            return False

        if regime == "BEAR":
            # Allow SELL/SHORT in bear regime for futures
            if action == "SELL" and self.mode == "FUTURES" and self.enable_futures_shorts:
                return True
            # Allow SELL for closing long positions in SPOT mode
            if action == "SELL" and self.mode == "SPOT":
                return True
            return False

        return False
