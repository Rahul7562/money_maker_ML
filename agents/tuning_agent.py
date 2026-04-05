# agents/tuning_agent.py
# Responsibility: Walk-forward parameter tuning with rolling train/test windows.

from dataclasses import dataclass
from itertools import product
from statistics import mean
import time
from typing import Dict, List

import pandas as pd

from agents.analysis_agent import AnalysisAgent, StrategyParams
from agents.regime_agent import RegimeAgent
from config import (
    MARKET_MODE,
    SIMULATION_FEE_RATE,
    WFO_EMA_FAST_CANDIDATES,
    WFO_EMA_SLOW_CANDIDATES,
    WFO_ENABLED,
    WFO_MAX_SYMBOLS,
    WFO_MAX_PARAMETER_SETS,
    WFO_MIN_AVG_TEST_RETURN,
    WFO_MIN_SCORE_CANDIDATES,
    WFO_MIN_WIN_RATE,
    WFO_REOPTIMIZE_EVERY_CYCLES,
    WFO_SIGNAL_STEP,
    WFO_RSI_OVERBOUGHT_CANDIDATES,
    WFO_RSI_OVERSOLD_CANDIDATES,
    WFO_STEP_CANDLES,
    WFO_TEST_CANDLES,
    WFO_TIME_BUDGET_SECONDS,
    WFO_TRAIN_CANDLES,
)


@dataclass
class TuneQuality:
    avg_test_return: float
    avg_test_drawdown: float
    win_rate: float
    windows: int


class TuningAgent:
    """Performs periodic walk-forward optimization and caches tuned params."""

    def __init__(self, analysis_agent: AnalysisAgent, regime_agent: RegimeAgent):
        self.analysis_agent = analysis_agent
        self.regime_agent = regime_agent
        self.enabled = WFO_ENABLED
        self.mode = MARKET_MODE
        self._cached_params: Dict[str, StrategyParams] = {}
        self._cached_quality: Dict[str, TuneQuality] = {}

        self._candidate_params = self._build_candidate_grid()

    def _build_candidate_grid(self) -> List[StrategyParams]:
        params = []
        for ema_fast, ema_slow, min_score, rsi_low, rsi_high in product(
            WFO_EMA_FAST_CANDIDATES,
            WFO_EMA_SLOW_CANDIDATES,
            WFO_MIN_SCORE_CANDIDATES,
            WFO_RSI_OVERSOLD_CANDIDATES,
            WFO_RSI_OVERBOUGHT_CANDIDATES,
        ):
            if ema_fast >= ema_slow:
                continue
            if rsi_low >= rsi_high:
                continue
            params.append(
                StrategyParams(
                    ema_fast=ema_fast,
                    ema_slow=ema_slow,
                    min_signal_score=min_score,
                    rsi_oversold=rsi_low,
                    rsi_overbought=rsi_high,
                )
            )

        if not params:
            return [self.analysis_agent.default_params]

        default = self.analysis_agent.default_params
        params.sort(
            key=lambda p: (
                abs(p.ema_fast - default.ema_fast)
                + abs(p.ema_slow - default.ema_slow)
                + abs(p.rsi_oversold - default.rsi_oversold)
                + abs(p.rsi_overbought - default.rsi_overbought)
                + abs(p.min_signal_score - default.min_signal_score) * 100.0
            )
        )
        return params[: max(1, WFO_MAX_PARAMETER_SETS)]

    def _max_drawdown(self, equity_curve: List[float]) -> float:
        peak = equity_curve[0]
        max_dd = 0.0
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)
        return max_dd

    def _simulate_segment(self, symbol: str, df: pd.DataFrame, params: StrategyParams) -> Dict[str, float]:
        equity = 1.0
        position = 0  # 1=long, -1=short, 0=flat
        trades = 0
        equity_curve = [equity]

        start_idx = max(params.ema_slow, 30)
        step = max(1, WFO_SIGNAL_STEP)
        for idx in range(start_idx, len(df), step):
            window = df.iloc[: idx + 1]
            signal = self.analysis_agent.analyze_symbol(symbol=symbol, df=window, params=params)
            regime = self.regime_agent.classify(symbol=symbol, df=window)

            action = signal.action if self.regime_agent.allows_action(regime.regime, signal.action) else "HOLD"
            prev_position = position

            if self.mode == "FUTURES":
                if action == "BUY":
                    position = 1
                elif action == "SELL":
                    position = -1
            else:
                if action == "BUY":
                    position = 1
                elif action == "SELL":
                    position = 0

            if position != prev_position:
                trades += 1
                equity *= (1.0 - SIMULATION_FEE_RATE)

            prev_idx = max(0, idx - step)
            if prev_idx == idx:
                continue
            ret = float(df["close"].iloc[idx] / df["close"].iloc[prev_idx] - 1.0)
            equity *= (1.0 + (position * ret))
            equity_curve.append(equity)

        total_return = equity - 1.0
        drawdown = self._max_drawdown(equity_curve)
        return {
            "return": total_return,
            "drawdown": drawdown,
            "trades": float(trades),
        }

    def _tune_symbol(self, symbol: str, df: pd.DataFrame) -> tuple[StrategyParams, TuneQuality]:
        started_at = time.time()
        required = WFO_TRAIN_CANDLES + WFO_TEST_CANDLES + 20
        if len(df) < required:
            default_params = self.analysis_agent.default_params
            quality = TuneQuality(avg_test_return=0.0, avg_test_drawdown=0.0, win_rate=0.0, windows=0)
            return default_params, quality

        selected_params = self.analysis_agent.default_params
        test_returns: List[float] = []
        test_drawdowns: List[float] = []

        start = 0
        while start + WFO_TRAIN_CANDLES + WFO_TEST_CANDLES <= len(df):
            train = df.iloc[start : start + WFO_TRAIN_CANDLES]
            test = df.iloc[start + WFO_TRAIN_CANDLES : start + WFO_TRAIN_CANDLES + WFO_TEST_CANDLES]

            best_score = float("-inf")
            best_params = self.analysis_agent.default_params

            for candidate in self._candidate_params:
                if (time.time() - started_at) > WFO_TIME_BUDGET_SECONDS:
                    break
                train_stats = self._simulate_segment(symbol=symbol, df=train, params=candidate)
                objective = (
                    train_stats["return"]
                    - (0.70 * train_stats["drawdown"])
                    + min(train_stats["trades"], 25.0) * 0.002
                )
                if objective > best_score:
                    best_score = objective
                    best_params = candidate

            selected_params = best_params
            test_stats = self._simulate_segment(symbol=symbol, df=test, params=best_params)
            test_returns.append(test_stats["return"])
            test_drawdowns.append(test_stats["drawdown"])

            start += WFO_STEP_CANDLES

            if (time.time() - started_at) > WFO_TIME_BUDGET_SECONDS:
                break

        if not test_returns:
            quality = TuneQuality(avg_test_return=0.0, avg_test_drawdown=0.0, win_rate=0.0, windows=0)
            return self.analysis_agent.default_params, quality

        wins = [x for x in test_returns if x > 0]
        quality = TuneQuality(
            avg_test_return=mean(test_returns),
            avg_test_drawdown=mean(test_drawdowns) if test_drawdowns else 0.0,
            win_rate=(len(wins) / len(test_returns)),
            windows=len(test_returns),
        )

        if quality.avg_test_return < WFO_MIN_AVG_TEST_RETURN or quality.win_rate < WFO_MIN_WIN_RATE:
            return self.analysis_agent.default_params, quality

        return selected_params, quality

    def maybe_tune_market(self, market_data: Dict[str, pd.DataFrame], cycle_count: int):
        """Run WFO periodically and return cached tuned params and quality map."""
        if not self.enabled:
            return {}, {}

        should_reoptimize = (
            cycle_count == 1
            or not self._cached_params
            or (cycle_count % max(1, WFO_REOPTIMIZE_EVERY_CYCLES) == 0)
        )

        if should_reoptimize:
            tuned: Dict[str, StrategyParams] = {}
            quality: Dict[str, TuneQuality] = {}

            for symbol, df in list(market_data.items())[:WFO_MAX_SYMBOLS]:
                best_params, best_quality = self._tune_symbol(symbol=symbol, df=df)
                tuned[symbol] = best_params
                quality[symbol] = best_quality

            self._cached_params = tuned
            self._cached_quality = quality

        return self._cached_params, self._cached_quality
