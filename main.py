# main.py
# Coordinates all agents and runs a robust multi-coin trading loop.

import logging
import time
from datetime import datetime

from agents.analysis_agent import AnalysisAgent
from agents.data_agent import DataAgent
from agents.execution_agent import ExecutionAgent
from agents.regime_agent import RegimeAgent
from agents.risk_agent import RiskAgent
from agents.tuning_agent import TuningAgent
from config import (
    LOG_FILE,
    MARKET_MODE,
    MAX_DAILY_DRAWDOWN_PERCENT,
    MAX_OPEN_POSITIONS,
    PAPER_TRADING,
    QUOTE_ASSET,
    SIMULATION_CANDLES,
    SIMULATION_ENABLED,
    SIMULATION_FEE_RATE,
    SIMULATION_WARMUP,
    TRADE_INTERVAL,
    WFO_ENABLED,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
logger = logging.getLogger("Orchestrator")


class InternalSimulator:
    """Simple walk-forward simulation used as a cycle health check."""

    def __init__(self, analysis_agent: AnalysisAgent, regime_agent: RegimeAgent):
        self.analysis_agent = analysis_agent
        self.regime_agent = regime_agent

    def _max_drawdown_pct(self, equity_curve):
        peak = equity_curve[0]
        max_dd = 0.0
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)
        return max_dd * 100.0

    def simulate_symbol(self, symbol: str, df, params=None):
        candles = df.tail(SIMULATION_CANDLES)
        if len(candles) <= SIMULATION_WARMUP + 5:
            return None

        equity = 1000.0
        position = 0  # 1=long, -1=short, 0=flat
        trades = 0
        equity_curve = [equity]

        # Re-evaluate every few candles to keep simulation responsive.
        step = 3
        for i in range(SIMULATION_WARMUP, len(candles), step):
            window = candles.iloc[: i + 1]
            signal = self.analysis_agent.analyze_symbol(symbol=symbol, df=window, params=params)
            regime = self.regime_agent.classify(symbol=symbol, df=window)
            action = signal.action if self.regime_agent.allows_action(regime.regime, signal.action) else "HOLD"

            prev_pos = position
            if MARKET_MODE == "FUTURES":
                if action == "BUY":
                    position = 1
                elif action == "SELL":
                    position = -1
            else:
                if action == "BUY":
                    position = 1
                elif action == "SELL":
                    position = 0

            if prev_pos != position:
                trades += 1
                equity *= (1.0 - SIMULATION_FEE_RATE)

            step_ret = float(window["close"].iloc[-1] / window["close"].iloc[-2] - 1.0)
            equity *= (1.0 + (position * step_ret))
            equity_curve.append(equity)

        ret_pct = ((equity - 1000.0) / 1000.0) * 100.0
        return {
            "symbol": symbol,
            "return_pct": ret_pct,
            "max_drawdown_pct": self._max_drawdown_pct(equity_curve),
            "trades": trades,
        }

    def simulate_market(self, market_data, params_map=None):
        results = []
        for symbol, df in list(market_data.items())[:8]:
            one = self.simulate_symbol(symbol, df, params=(params_map or {}).get(symbol))
            if one:
                results.append(one)

        if not results:
            return None

        results.sort(key=lambda x: x["return_pct"], reverse=True)
        avg_return = sum(x["return_pct"] for x in results) / len(results)
        avg_drawdown = sum(x["max_drawdown_pct"] for x in results) / len(results)
        winners = [x for x in results if x["return_pct"] > 0]

        return {
            "avg_return_pct": avg_return,
            "avg_drawdown_pct": avg_drawdown,
            "win_rate_pct": (len(winners) / len(results)) * 100,
            "best": results[0],
            "worst": results[-1],
            "tested": len(results),
        }


class Orchestrator:
    """Multi-agent loop: universe scan -> ranking -> risk -> execution."""

    def __init__(self):
        logger.info("=" * 72)
        logger.info("Trading Bot Starting")
        logger.info("Mode: %s", "PAPER" if PAPER_TRADING else "LIVE")
        logger.info("Market mode: %s | Interval: %s | Quote asset: %s", MARKET_MODE, TRADE_INTERVAL, QUOTE_ASSET)
        logger.info("=" * 72)

        self.data_agent = DataAgent()
        self.analysis_agent = AnalysisAgent()
        self.regime_agent = RegimeAgent()
        self.risk_agent = RiskAgent()
        self.execution_agent = ExecutionAgent(
            binance_client=self.data_agent.client if not PAPER_TRADING else None
        )
        self.tuning_agent = TuningAgent(self.analysis_agent, self.regime_agent)
        self.simulator = InternalSimulator(self.analysis_agent, self.regime_agent)
        self.cycle_count = 0

    def _get_balance(self) -> float:
        if PAPER_TRADING:
            return self.execution_agent.get_available_cash()
        return self.data_agent.get_balance(QUOTE_ASSET)

    def _get_position_qty(self, symbol: str) -> float:
        if PAPER_TRADING:
            return self.execution_agent.get_position_qty(symbol)

        if MARKET_MODE == "FUTURES":
            try:
                return self.data_agent.get_position_qty(symbol)
            except Exception:
                return 0.0

        base_asset = symbol.replace(QUOTE_ASSET, "")
        try:
            return self.data_agent.get_balance(base_asset)
        except Exception:
            return 0.0

    def run_cycle(self):
        self.cycle_count += 1
        logger.info("\n%s", "-" * 70)
        logger.info("Cycle #%s | %s", self.cycle_count, datetime.utcnow().isoformat())

        try:
            symbols = self.data_agent.get_tradeable_symbols()
            market_data = self.data_agent.get_batch_candles(symbols)
            marks = {
                sym: float(df["close"].iloc[-1]) for sym, df in market_data.items()
            }

            if PAPER_TRADING:
                exits = self.execution_agent.apply_risk_exits(price_map=marks)
                if exits:
                    logger.info("Auto risk exits triggered: %s", len(exits))

            params_map, quality_map = self.tuning_agent.maybe_tune_market(
                market_data=market_data,
                cycle_count=self.cycle_count,
            )
            signals = self.analysis_agent.analyze_market(market_data, params_map=params_map)

            regimes = {
                symbol: self.regime_agent.classify(symbol=symbol, df=df)
                for symbol, df in market_data.items()
            }
            bull_count = len([1 for r in regimes.values() if r.regime == "BULL"])
            bear_count = len([1 for r in regimes.values() if r.regime == "BEAR"])
            side_count = len([1 for r in regimes.values() if r.regime == "SIDEWAYS"])
            logger.info(
                "Regime distribution | bull=%s bear=%s sideways=%s",
                bull_count,
                bear_count,
                side_count,
            )

            if WFO_ENABLED and quality_map:
                quality_values = list(quality_map.values())
                avg_wfo_ret = sum(q.avg_test_return for q in quality_values) / len(quality_values)
                avg_wfo_win = sum(q.win_rate for q in quality_values) / len(quality_values)
                logger.info(
                    "WFO quality | tuned=%s symbols | avg_test_ret=%.3f | avg_win_rate=%.2f",
                    len(quality_values),
                    avg_wfo_ret,
                    avg_wfo_win,
                )

            top_preview = [
                f"{s.symbol}:{s.action}:{s.score:.2f}" for s in signals[:5]
            ]
            logger.info("Top ranked signals: %s", " | ".join(top_preview) if top_preview else "none")

            balance = self._get_balance()
            selected_signal = None
            selected_decision = None

            allow_new_positions = True
            if PAPER_TRADING:
                pre_portfolio = self.execution_agent.get_portfolio_summary(price_map=marks)
                if pre_portfolio["day_pnl_pct"] <= -abs(MAX_DAILY_DRAWDOWN_PERCENT):
                    allow_new_positions = False
                    logger.warning(
                        "Daily drawdown guard active (%.2f%% <= -%.2f%%): only position reductions allowed",
                        pre_portfolio["day_pnl_pct"],
                        abs(MAX_DAILY_DRAWDOWN_PERCENT),
                    )

            for candidate in signals:
                if candidate.action not in {"BUY", "SELL"}:
                    continue

                candidate_regime = regimes.get(candidate.symbol)
                if candidate_regime and not self.regime_agent.allows_action(candidate_regime.regime, candidate.action):
                    continue

                position_qty = self._get_position_qty(candidate.symbol)
                decision = self.risk_agent.evaluate(
                    signal=candidate,
                    balance_usdt=balance,
                    current_price=candidate.price,
                    position_qty=position_qty,
                )

                if not decision.approved:
                    continue

                if PAPER_TRADING and decision.position_intent.startswith("OPEN"):
                    if self.execution_agent.get_open_positions_count() >= MAX_OPEN_POSITIONS:
                        continue
                    if not allow_new_positions:
                        continue

                if decision.approved and decision.trade_amount_usdt > 0:
                    selected_signal = candidate
                    selected_decision = decision
                    break

            if not selected_signal or not selected_decision:
                logger.info("No actionable signal found in this cycle")
            else:
                result = self.execution_agent.execute(selected_signal, selected_decision)
                logger.info(
                    "Execution result | %s | %s | %s",
                    selected_signal.symbol,
                    result.get("status"),
                    result.get("reason", "ok"),
                )

            if PAPER_TRADING:
                portfolio = self.execution_agent.get_portfolio_summary(price_map=marks)
                logger.info(
                    "Portfolio | mode=%s | cash=%.2f %s | used_margin=%.2f | positions=%.2f | unrealized=%.2f | equity=%.2f | day=%.2f%% | realized=%.2f | trades=%s",
                    portfolio["mode"],
                    portfolio["cash"],
                    portfolio["quote_asset"],
                    portfolio["used_margin"],
                    portfolio["positions_value"],
                    portfolio["unrealized_pnl"],
                    portfolio["total_equity"],
                    portfolio["day_pnl_pct"],
                    portfolio["realized_pnl"],
                    portfolio["trade_count"],
                )

            if SIMULATION_ENABLED:
                sim = self.simulator.simulate_market(market_data, params_map=params_map)
                if sim:
                    logger.info(
                        "Simulation | tested=%s | avg_ret=%.2f%% | win_rate=%.1f%% | avg_dd=%.2f%% | best=%s %.2f%% | worst=%s %.2f%%",
                        sim["tested"],
                        sim["avg_return_pct"],
                        sim["win_rate_pct"],
                        sim["avg_drawdown_pct"],
                        sim["best"]["symbol"],
                        sim["best"]["return_pct"],
                        sim["worst"]["symbol"],
                        sim["worst"]["return_pct"],
                    )

        except Exception as exc:
            logger.error("Cycle failed: %s", exc, exc_info=True)

    def start(self):
        interval_map = {
            "1m": 1,
            "5m": 5,
            "15m": 15,
            "1h": 60,
            "4h": 240,
            "1d": 1440,
        }
        minutes = interval_map.get(TRADE_INTERVAL, 60)
        interval_seconds = minutes * 60
        logger.info("Scheduler active: every %s minutes", minutes)

        self.run_cycle()
        next_run_ts = time.time() + interval_seconds

        while True:
            now_ts = time.time()
            if now_ts >= next_run_ts:
                self.run_cycle()
                next_run_ts = now_ts + interval_seconds
            sleep_for = max(1, int(next_run_ts - now_ts))
            time.sleep(min(sleep_for, 15))


if __name__ == "__main__":
    Orchestrator().start()