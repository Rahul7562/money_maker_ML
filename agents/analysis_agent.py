# agents/analysis_agent.py
# Responsibility: Convert OHLCV data into ranked multi-coin signals.

import logging
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from config import (
    ATR_PERIOD,
    EMA_FAST,
    EMA_SLOW,
    MACD_FAST,
    MACD_SIGNAL,
    MACD_SLOW,
    MIN_SIGNAL_SCORE,
    RSI_OVERBOUGHT,
    RSI_OVERSOLD,
    RSI_PERIOD,
)

logger = logging.getLogger("AnalysisAgent")


@dataclass
class Signal:
    symbol: str
    action: str
    confidence: float
    score: float
    reason: str
    price: float
    rsi: float
    macd: float
    macd_signal: float
    atr_pct: float


class AnalysisAgent:
    """Scores every symbol and returns robust ranked opportunities."""

    def _clamp(self, value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()

        close = data["close"]
        high = data["high"]
        low = data["low"]

        delta = close.diff()
        gain = delta.clip(lower=0).rolling(RSI_PERIOD).mean()
        loss = (-delta.clip(upper=0)).rolling(RSI_PERIOD).mean()
        rs = gain / loss.replace(0, 1e-12)
        data["rsi"] = (100 - (100 / (1 + rs))).clip(lower=0, upper=100).fillna(50.0)

        ema_fast = close.ewm(span=MACD_FAST, adjust=False).mean()
        ema_slow = close.ewm(span=MACD_SLOW, adjust=False).mean()
        data["macd"] = ema_fast - ema_slow
        data["macd_signal"] = data["macd"].ewm(span=MACD_SIGNAL, adjust=False).mean()

        data["ema_fast"] = close.ewm(span=EMA_FAST, adjust=False).mean()
        data["ema_slow"] = close.ewm(span=EMA_SLOW, adjust=False).mean()

        tr = pd.concat(
            [
                (high - low),
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        data["atr"] = tr.rolling(ATR_PERIOD).mean().bfill()
        data["atr_pct"] = (data["atr"] / close).replace([pd.NA, pd.NaT], 0.0)

        data["ret_3"] = close.pct_change(3).fillna(0.0)
        data["vol_ma"] = data["volume"].rolling(20).mean().bfill()
        return data.dropna()

    def analyze_symbol(self, symbol: str, df: pd.DataFrame) -> Signal:
        """Analyze one symbol and return BUY/SELL/HOLD with confidence score."""
        required = max(RSI_PERIOD, MACD_SLOW, EMA_SLOW, ATR_PERIOD) + 5
        if len(df) < required:
            latest_price = float(df["close"].iloc[-1])
            return Signal(
                symbol=symbol,
                action="HOLD",
                confidence=0.0,
                score=0.0,
                reason="Insufficient history",
                price=latest_price,
                rsi=50.0,
                macd=0.0,
                macd_signal=0.0,
                atr_pct=0.0,
            )

        data = self._add_indicators(df)
        latest = data.iloc[-1]
        prev = data.iloc[-2]

        trend_up = 1.0 if latest["ema_fast"] > latest["ema_slow"] else 0.0
        trend_down = 1.0 - trend_up

        macd_up = 1.0 if latest["macd"] > latest["macd_signal"] else 0.0
        macd_down = 1.0 - macd_up

        momentum_up = self._clamp(float(latest["ret_3"]) * 20.0, 0.0, 1.0)
        momentum_down = self._clamp(float(-latest["ret_3"]) * 20.0, 0.0, 1.0)

        rsi = float(latest["rsi"])
        rsi_buy = self._clamp((RSI_OVERBOUGHT - rsi) / max(RSI_OVERBOUGHT - RSI_OVERSOLD, 1.0), 0.0, 1.0)
        rsi_sell = self._clamp((rsi - RSI_OVERSOLD) / max(RSI_OVERBOUGHT - RSI_OVERSOLD, 1.0), 0.0, 1.0)

        volume_boost = 1.0 if latest["volume"] > latest["vol_ma"] * 1.1 else 0.0
        atr_pct = float(latest["atr_pct"])

        volatility_penalty = self._clamp((atr_pct - 0.035) * 8.0, 0.0, 0.3)
        crossover_boost_buy = 0.1 if prev["macd"] <= prev["macd_signal"] and macd_up else 0.0
        crossover_boost_sell = 0.1 if prev["macd"] >= prev["macd_signal"] and macd_down else 0.0

        buy_score = (
            (0.34 * trend_up)
            + (0.28 * macd_up)
            + (0.18 * momentum_up)
            + (0.15 * rsi_buy)
            + (0.05 * volume_boost)
            + crossover_boost_buy
            - volatility_penalty
        )

        sell_score = (
            (0.34 * trend_down)
            + (0.28 * macd_down)
            + (0.18 * momentum_down)
            + (0.15 * rsi_sell)
            + (0.05 * volume_boost)
            + crossover_boost_sell
            - volatility_penalty
        )

        buy_score = self._clamp(buy_score, 0.0, 1.0)
        sell_score = self._clamp(sell_score, 0.0, 1.0)

        if buy_score >= MIN_SIGNAL_SCORE and buy_score > sell_score + 0.04:
            action = "BUY"
            score = buy_score
            reason = (
                f"Trend+momentum bullish (score={buy_score:.2f}, RSI={rsi:.1f}, "
                f"ATR={atr_pct*100:.2f}%)"
            )
        elif sell_score >= MIN_SIGNAL_SCORE and sell_score > buy_score + 0.04:
            action = "SELL"
            score = sell_score
            reason = (
                f"Trend+momentum bearish (score={sell_score:.2f}, RSI={rsi:.1f}, "
                f"ATR={atr_pct*100:.2f}%)"
            )
        else:
            action = "HOLD"
            score = max(buy_score, sell_score)
            reason = f"No edge over threshold (buy={buy_score:.2f}, sell={sell_score:.2f})"

        confidence = round(self._clamp(score, 0.0, 1.0), 3)
        return Signal(
            symbol=symbol,
            action=action,
            confidence=confidence,
            score=round(score, 3),
            reason=reason,
            price=float(latest["close"]),
            rsi=round(rsi, 2),
            macd=round(float(latest["macd"]), 6),
            macd_signal=round(float(latest["macd_signal"]), 6),
            atr_pct=round(atr_pct, 6),
        )

    def analyze_market(self, market_data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """Analyze all symbols and rank by score descending."""
        signals: List[Signal] = []
        for symbol, df in market_data.items():
            try:
                signals.append(self.analyze_symbol(symbol=symbol, df=df))
            except Exception as exc:
                logger.warning("Analysis failed for %s: %s", symbol, exc)
                continue

        signals.sort(key=lambda s: s.score, reverse=True)
        return signals

    def select_best_signal(self, signals: List[Signal]) -> Signal | None:
        """Pick the strongest actionable signal across the whole market."""
        actionable = [s for s in signals if s.action in {"BUY", "SELL"}]
        if not actionable:
            return None
        return max(actionable, key=lambda s: s.score)
