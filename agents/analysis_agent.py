# agents/analysis_agent.py
# Responsibility: Convert OHLCV data into ranked multi-coin signals.

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from config import (
    ATR_PERIOD,
    EMA_FAST,
    EMA_SLOW,
    MACD_FAST,
    MACD_SIGNAL,
    MACD_SLOW,
    MIN_SIGNAL_SCORE,
    MTF_ENABLED,
    MTF_HIGHER_INTERVAL,
    RSI_OVERBOUGHT,
    RSI_OVERSOLD,
    RSI_PERIOD,
)

logger = logging.getLogger("AnalysisAgent")


@dataclass
class StrategyParams:
    """Parameters for technical analysis strategy."""
    rsi_period: int = RSI_PERIOD
    ema_fast: int = EMA_FAST
    ema_slow: int = EMA_SLOW
    rsi_oversold: float = RSI_OVERSOLD
    rsi_overbought: float = RSI_OVERBOUGHT
    min_signal_score: float = MIN_SIGNAL_SCORE
    macd_fast: int = MACD_FAST
    macd_slow: int = MACD_SLOW
    macd_signal: int = MACD_SIGNAL
    atr_period: int = ATR_PERIOD


@dataclass
class Signal:
    """Trading signal with full analysis details."""
    symbol: str
    action: str
    confidence: float
    score: float
    buy_score: float
    sell_score: float
    reason: str
    price: float
    rsi: float
    macd: float
    macd_signal: float
    atr_pct: float


class AnalysisAgent:
    """Scores every symbol and returns robust ranked opportunities."""

    def __init__(self):
        self.default_params = StrategyParams()

    def _clamp(self, value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def _add_indicators(self, df: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        """
        Add all technical indicators to the DataFrame.
        
        Includes: RSI, MACD, EMAs, ATR, Bollinger Bands, VWAP, volume ratio.
        """
        data = df.copy()

        close = data["close"]
        high = data["high"]
        low = data["low"]
        volume = data["volume"]

        # RSI
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(params.rsi_period).mean()
        loss = (-delta.clip(upper=0)).rolling(params.rsi_period).mean()
        rs = gain / loss.replace(0, 1e-12)
        data["rsi"] = (100 - (100 / (1 + rs))).clip(lower=0, upper=100).fillna(50.0)

        # MACD
        ema_fast = close.ewm(span=params.macd_fast, adjust=False).mean()
        ema_slow = close.ewm(span=params.macd_slow, adjust=False).mean()
        data["macd"] = ema_fast - ema_slow
        data["macd_signal"] = data["macd"].ewm(span=params.macd_signal, adjust=False).mean()

        # EMAs for trend
        data["ema_fast"] = close.ewm(span=params.ema_fast, adjust=False).mean()
        data["ema_slow"] = close.ewm(span=params.ema_slow, adjust=False).mean()

        # ATR
        tr = pd.concat(
            [
                (high - low),
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        data["atr"] = tr.rolling(params.atr_period).mean().bfill()
        data["atr_pct"] = (data["atr"] / close).replace([pd.NA, pd.NaT], 0.0)

        # Momentum
        data["ret_3"] = close.pct_change(3).fillna(0.0)
        data["vol_ma"] = volume.rolling(20).mean().bfill()
        
        # Bollinger Bands (20-period, 2 std dev)
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        data["bb_mid"] = bb_mid.bfill()
        data["bb_upper"] = (bb_mid + 2 * bb_std).bfill()
        data["bb_lower"] = (bb_mid - 2 * bb_std).bfill()
        
        # VWAP: (high+low+close)/3 weighted by volume, rolling 20-period
        typical_price = (high + low + close) / 3
        data["vwap"] = (typical_price * volume).rolling(20).sum() / volume.rolling(20).sum()
        data["vwap"] = data["vwap"].bfill()
        
        # Volume ratio (current / 20-period average)
        data["volume_ratio"] = (volume / data["vol_ma"].replace(0, 1)).fillna(1.0)
        
        return data.dropna()

    def analyze_symbol(
        self, 
        symbol: str, 
        df: pd.DataFrame, 
        params: StrategyParams | None = None,
        performance_agent: Optional[Any] = None,
    ) -> Signal:
        """
        Analyze one symbol and return BUY/SELL/HOLD with confidence score.
        
        Includes: VWAP, Bollinger Bands, volume spikes, candle patterns.
        
        Args:
            symbol: Trading pair symbol
            df: OHLCV DataFrame
            params: Optional strategy parameters
            performance_agent: Optional PerformanceAgent for win rate blending
            
        Returns:
            Signal with action, score, and analysis details
        """
        active_params = params or self.default_params
        required = max(
            active_params.macd_slow,
            active_params.ema_slow,
            active_params.atr_period,
            active_params.rsi_period,
        ) + 5
        if len(df) < required:
            latest_price = float(df["close"].iloc[-1])
            return Signal(
                symbol=symbol,
                action="HOLD",
                confidence=0.0,
                score=0.0,
                buy_score=0.0,
                sell_score=0.0,
                reason="Insufficient history",
                price=latest_price,
                rsi=50.0,
                macd=0.0,
                macd_signal=0.0,
                atr_pct=0.0,
            )

        data = self._add_indicators(df, active_params)
        latest = data.iloc[-1]
        prev = data.iloc[-2]

        # === BASE INDICATORS ===
        trend_up = 1.0 if latest["ema_fast"] > latest["ema_slow"] else 0.0
        trend_down = 1.0 - trend_up

        macd_up = 1.0 if latest["macd"] > latest["macd_signal"] else 0.0
        macd_down = 1.0 - macd_up

        momentum_up = self._clamp(float(latest["ret_3"]) * 20.0, 0.0, 1.0)
        momentum_down = self._clamp(float(-latest["ret_3"]) * 20.0, 0.0, 1.0)

        rsi = float(latest["rsi"])
        rsi_buy = self._clamp(
            (active_params.rsi_overbought - rsi)
            / max(active_params.rsi_overbought - active_params.rsi_oversold, 1.0),
            0.0,
            1.0,
        )
        rsi_sell = self._clamp(
            (rsi - active_params.rsi_oversold)
            / max(active_params.rsi_overbought - active_params.rsi_oversold, 1.0),
            0.0,
            1.0,
        )

        volume_boost = 1.0 if latest["volume"] > latest["vol_ma"] * 1.1 else 0.0
        atr_pct = float(latest["atr_pct"])

        volatility_penalty = self._clamp((atr_pct - 0.035) * 8.0, 0.0, 0.3)
        crossover_boost_buy = 0.1 if prev["macd"] <= prev["macd_signal"] and macd_up else 0.0
        crossover_boost_sell = 0.1 if prev["macd"] >= prev["macd_signal"] and macd_down else 0.0

        # === NEW: VWAP signals ===
        price = float(latest["close"])
        vwap = float(latest["vwap"])
        vwap_buy = 0.05 if price > vwap else 0.0
        vwap_sell = 0.05 if price < vwap else 0.0
        
        # === NEW: Bollinger Band signals ===
        bb_upper = float(latest["bb_upper"])
        bb_lower = float(latest["bb_lower"])
        bb_mid = float(latest["bb_mid"])
        
        bb_buy = 0.0
        bb_sell = 0.0
        
        # At/below lower band = buy signal
        if price <= bb_lower:
            bb_buy = 0.05
            # Breakout below lower band (extra score)
            if price < bb_lower * 0.99:
                bb_buy = 0.08
        
        # At/above upper band = sell signal
        if price >= bb_upper:
            bb_sell = 0.05
            # Breakout above upper band (extra score)
            if price > bb_upper * 1.01:
                bb_sell = 0.08
        
        # === NEW: Volume spike/drought ===
        volume_ratio = float(latest.get("volume_ratio", 1.0))
        volume_multiplier = 1.0
        if volume_ratio > 2.0:  # Volume spike
            volume_multiplier = 1.10
        elif volume_ratio < 0.5:  # Volume drought
            volume_multiplier = 0.90
        
        # === NEW: Candle patterns ===
        candle_buy = 0.0
        candle_sell = 0.0
        
        curr_open = float(latest["open"])
        curr_close = float(latest["close"])
        curr_high = float(latest["high"])
        curr_low = float(latest["low"])
        prev_open = float(prev["open"])
        prev_close = float(prev["close"])
        
        body = abs(curr_close - curr_open)
        upper_wick = curr_high - max(curr_open, curr_close)
        lower_wick = min(curr_open, curr_close) - curr_low
        
        # Bullish engulfing
        if curr_close > prev_open and curr_open < prev_close and curr_close > curr_open:
            candle_buy += 0.04
        
        # Bearish engulfing
        if curr_close < prev_open and curr_open > prev_close and curr_close < curr_open:
            candle_sell += 0.04
        
        # Doji (indecision)
        doji_multiplier = 1.0
        if body < 0.001 * curr_close:
            doji_multiplier = 0.92  # Reduce confidence on doji
        
        # Hammer (bullish reversal)
        if lower_wick > 2 * body and upper_wick < body and body > 0:
            candle_buy += 0.03
        
        # Shooting star (bearish reversal)
        if upper_wick > 2 * body and lower_wick < body and body > 0:
            candle_sell += 0.03

        # === COMPUTE SCORES ===
        buy_score = (
            (0.34 * trend_up)
            + (0.28 * macd_up)
            + (0.18 * momentum_up)
            + (0.15 * rsi_buy)
            + (0.05 * volume_boost)
            + crossover_boost_buy
            + vwap_buy
            + bb_buy
            + candle_buy
            - volatility_penalty
        )

        sell_score = (
            (0.34 * trend_down)
            + (0.28 * macd_down)
            + (0.18 * momentum_down)
            + (0.15 * rsi_sell)
            + (0.05 * volume_boost)
            + crossover_boost_sell
            + vwap_sell
            + bb_sell
            + candle_sell
            - volatility_penalty
        )

        # Apply modifiers
        buy_score = buy_score * volume_multiplier * doji_multiplier
        sell_score = sell_score * volume_multiplier * doji_multiplier

        # === NEW: Performance-based score adjustment ===
        if performance_agent is not None:
            try:
                win_rate = performance_agent.get_symbol_win_rate(symbol)
                trade_count = performance_agent.get_trade_count()
                
                if win_rate > 0.6:
                    buy_score *= 1.05
                    sell_score *= 1.05
                elif win_rate < 0.4 and trade_count > 10:
                    buy_score *= 0.92
                    sell_score *= 0.92
            except Exception:
                pass  # Skip if performance agent unavailable

        buy_score = self._clamp(buy_score, 0.0, 1.0)
        sell_score = self._clamp(sell_score, 0.0, 1.0)

        if buy_score >= active_params.min_signal_score and buy_score > sell_score + 0.04:
            action = "BUY"
            score = buy_score
            reason = (
                f"Trend+momentum bullish (score={buy_score:.2f}, RSI={rsi:.1f}, "
                f"ATR={atr_pct*100:.2f}%)"
            )
        elif sell_score >= active_params.min_signal_score and sell_score > buy_score + 0.04:
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
            buy_score=round(buy_score, 3),
            sell_score=round(sell_score, 3),
            reason=reason,
            price=float(latest["close"]),
            rsi=round(rsi, 2),
            macd=round(float(latest["macd"]), 6),
            macd_signal=round(float(latest["macd_signal"]), 6),
            atr_pct=round(atr_pct, 6),
        )

    def analyze_market(
        self,
        market_data: Dict[str, pd.DataFrame],
        params_map: Dict[str, StrategyParams] | None = None,
        performance_agent: Optional[Any] = None,
    ) -> List[Signal]:
        """
        Analyze all symbols and rank by score descending.
        
        Args:
            market_data: Dict mapping symbols to OHLCV DataFrames
            params_map: Optional dict mapping symbols to custom params
            performance_agent: Optional PerformanceAgent for win rate blending
            
        Returns:
            List of Signals sorted by score descending
        """
        signals: List[Signal] = []
        for symbol, df in market_data.items():
            try:
                one_params = (params_map or {}).get(symbol)
                signals.append(self.analyze_symbol(
                    symbol=symbol, 
                    df=df, 
                    params=one_params,
                    performance_agent=performance_agent,
                ))
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
