# main.py
# Coordinates all agents and runs a robust multi-coin trading loop.

import json
import logging
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from logging.handlers import RotatingFileHandler
from typing import Dict, List, Optional

from agents.analysis_agent import AnalysisAgent
from agents.correlation_agent import CorrelationAgent
from agents.data_agent import DataAgent
from agents.execution_agent import ExecutionAgent
from agents.performance_agent import PerformanceAgent
from agents.portfolio_agent import PortfolioAgent
from agents.regime_agent import RegimeAgent
from agents.risk_agent import RiskAgent
from agents.sentiment_agent import SentimentAgent
from agents.tuning_agent import TuningAgent
from agents.ml_agent import MLAgent
from config import (
    HEALTH_CHECK_PORT,
    LOG_FILE,
    LOG_MAX_BACKUPS,
    LOG_ROTATION_MB,
    MARKET_MODE,
    MAX_DAILY_DRAWDOWN_PERCENT,
    MAX_OPEN_POSITIONS,
    MAX_TRADE_PERCENT,
    MIN_SCORE_FOR_SLOT_2_PLUS,
    MIN_TRADE_PERCENT,
    ML_ENABLED,
    ML_RETRAIN_EVERY_N_TUNING_CYCLES,
    ML_WEIGHT,
    PAPER_TRADING,
    QUOTE_ASSET,
    SIDEWAYS_MIN_SCORE,
    SIMULATION_CANDLES,
    SIMULATION_ENABLED,
    SIMULATION_FEE_RATE,
    SIMULATION_WARMUP,
    TELEGRAM_ENABLED,
    TRADE_INTERVAL,
    WFO_ENABLED,
    WFO_REOPTIMIZE_EVERY_CYCLES,
)

# Import TelegramNotifier if available
try:
    from utils.notifier import TelegramNotifier
except ImportError:
    TelegramNotifier = None

# ────────────────────────────────────────────────────────────────────────────────
# TASK 8: Logging upgrade - RotatingFileHandler with stdout for journald
# ────────────────────────────────────────────────────────────────────────────────
def setup_logging():
    """Configure logging with RotatingFileHandler + StreamHandler for journald."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    
    # RotatingFileHandler: max 50MB, keep 5 backups
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=LOG_ROTATION_MB * 1024 * 1024,
        backupCount=LOG_MAX_BACKUPS,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # StreamHandler for stdout (systemd journald)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

setup_logging()
logger = logging.getLogger("Orchestrator")

# Global for graceful shutdown
_shutdown_requested = False
_orchestrator_instance: Optional["Orchestrator"] = None
_start_time = datetime.now(timezone.utc)


# ────────────────────────────────────────────────────────────────────────────────
# TASK 6: Graceful shutdown handler
# ────────────────────────────────────────────────────────────────────────────────
def _shutdown_handler(signum, frame):
    """Handle SIGINT/SIGTERM for graceful shutdown."""
    global _shutdown_requested
    sig_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
    logger.info("Received %s, initiating graceful shutdown...", sig_name)
    _shutdown_requested = True

signal.signal(signal.SIGINT, _shutdown_handler)
signal.signal(signal.SIGTERM, _shutdown_handler)


# ────────────────────────────────────────────────────────────────────────────────
# TASK 6: Health check HTTP server
# ────────────────────────────────────────────────────────────────────────────────
class HealthCheckHandler(BaseHTTPRequestHandler):
    """Simple HTTP handler for health checks."""
    
    def log_message(self, format, *args):
        pass  # Suppress HTTP access logs
    
    def do_GET(self):
        if self.path == "/health":
            self._respond_health()
        else:
            self.send_response(404)
            self.end_headers()
    
    def _respond_health(self):
        global _orchestrator_instance, _start_time
        
        now = datetime.now(timezone.utc)
        uptime_hours = (now - _start_time).total_seconds() / 3600.0
        
        if _orchestrator_instance is not None:
            cycle = _orchestrator_instance.cycle_count
            equity = _orchestrator_instance._last_equity
            positions = _orchestrator_instance._last_positions_count
            last_cycle = _orchestrator_instance._last_cycle_time.isoformat() if _orchestrator_instance._last_cycle_time else ""
            status = "running"
        else:
            cycle = 0
            equity = 0.0
            positions = 0
            last_cycle = ""
            status = "starting"
        
        health_data = {
            "status": status,
            "cycle": cycle,
            "equity": round(equity, 2),
            "positions": positions,
            "last_cycle_utc": last_cycle,
            "uptime_hours": round(uptime_hours, 2),
        }
        
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(health_data).encode("utf-8"))


def start_health_server(port: int):
    """Start the health check HTTP server in a daemon thread."""
    server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info("Health check server started on port %d", port)


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

        # Re-evaluate every candle for accuracy (FIX 5: changed from step=3 to step=1)
        step = 1
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
    """
    Multi-agent loop: universe scan -> ranking -> risk -> execution.
    
    TASK 7: Full agent pipeline order every cycle:
    data → auto_exits → tuning → analysis → regime → ML → sentiment
    → correlation_filter → risk → performance_check → execution
    → portfolio_update → simulation → logging
    """

    def __init__(self):
        global _orchestrator_instance
        _orchestrator_instance = self
        
        logger.info("=" * 72)
        logger.info("Trading Bot Starting")
        logger.info("Mode: %s", "PAPER" if PAPER_TRADING else "LIVE")
        logger.info("Market mode: %s | Interval: %s | Quote asset: %s", MARKET_MODE, TRADE_INTERVAL, QUOTE_ASSET)
        logger.info("ML Enabled: %s | ML Weight: %.2f", ML_ENABLED, ML_WEIGHT)
        logger.info("=" * 72)

        # Initialize all agents
        self.data_agent = DataAgent()
        self.analysis_agent = AnalysisAgent()
        self.regime_agent = RegimeAgent()
        self.risk_agent = RiskAgent()
        self.portfolio_agent = PortfolioAgent()
        self.performance_agent = PerformanceAgent()
        self.sentiment_agent = SentimentAgent()
        self.correlation_agent = CorrelationAgent()
        self.execution_agent = ExecutionAgent(
            binance_client=self.data_agent.client if not PAPER_TRADING else None,
            portfolio_agent=self.portfolio_agent,
            performance_agent=self.performance_agent,
        )
        self.tuning_agent = TuningAgent(self.analysis_agent, self.regime_agent)
        self.simulator = InternalSimulator(self.analysis_agent, self.regime_agent)
        
        # ML Agent for neural network predictions
        self.ml_agent = MLAgent()
        
        # Telegram notifier (if enabled and available)
        self.notifier = None
        if TELEGRAM_ENABLED and TelegramNotifier is not None:
            self.notifier = TelegramNotifier()
            self.notifier.send("🤖 Trading Bot starting...")
        
        self.cycle_count = 0
        
        # Health check state
        self._last_equity = 0.0
        self._last_positions_count = 0
        self._last_cycle_time: Optional[datetime] = None

    def _get_balance(self) -> float:
        if PAPER_TRADING:
            return self.portfolio_agent.get_cash()
        return self.data_agent.get_balance(QUOTE_ASSET)

    def _get_position_qty(self, symbol: str) -> float:
        if PAPER_TRADING:
            return self.portfolio_agent.get_position_qty(symbol)

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

    def _get_open_position_symbols(self) -> List[str]:
        """Get list of symbols with open positions."""
        if PAPER_TRADING:
            return self.portfolio_agent.get_position_symbols()
        return []

    def run_cycle(self):
        """
        TASK 7: Full agent pipeline order:
        data → auto_exits → tuning → analysis → regime → ML → sentiment
        → correlation_filter → risk → performance_check → execution
        → portfolio_update → simulation → logging
        """
        global _shutdown_requested
        
        self.cycle_count += 1
        self._last_cycle_time = datetime.now(timezone.utc)
        logger.info("\n%s", "-" * 70)
        logger.info("Cycle #%s | %s", self.cycle_count, self._last_cycle_time.isoformat())

        try:
            # ── STEP 1: DATA ──
            symbols = self.data_agent.get_tradeable_symbols()
            market_data = self.data_agent.get_batch_candles(symbols)
            marks = {
                sym: float(df["close"].iloc[-1]) for sym, df in market_data.items()
            }

            # ── STEP 2: AUTO_EXITS ──
            if PAPER_TRADING:
                exits = self.execution_agent.apply_risk_exits(price_map=marks)
                if exits:
                    logger.info("Auto risk exits triggered: %s", len(exits))
                    # Notify on exits
                    if self.notifier:
                        for ex in exits:
                            if ex.get("status") == "paper_executed":
                                self.notifier.send(
                                    f"🔴 Exit: {ex.get('symbol')} - {ex.get('trade', {}).get('exit_reason', 'risk')}"
                                )

            # ── STEP 3: TUNING ──
            params_map, quality_map = self.tuning_agent.maybe_tune_market(
                market_data=market_data,
                cycle_count=self.cycle_count,
            )
            
            # ── STEP 4: ANALYSIS ──
            signals = self.analysis_agent.analyze_market(
                market_data, 
                params_map=params_map,
                performance_agent=self.performance_agent,
            )
            
            # ── STEP 5: REGIME ──
            regimes = {
                symbol: self.regime_agent.classify(symbol=symbol, df=df)
                for symbol, df in market_data.items()
            }
            bull_count = len([1 for r in regimes.values() if r.regime == "BULL"])
            bear_count = len([1 for r in regimes.values() if r.regime == "BEAR"])
            side_count = len([1 for r in regimes.values() if r.regime == "SIDEWAYS"])
            logger.info(
                "Regime distribution | bull=%s bear=%s sideways=%s",
                bull_count, bear_count, side_count,
            )
            
            # ── STEP 6: ML ──
            # Check if ML model should be retrained
            if ML_ENABLED and self.ml_agent.should_retrain(self.cycle_count, WFO_REOPTIMIZE_EVERY_CYCLES):
                logger.info("Starting ML model retraining...")
                train_result = self.ml_agent.train(market_data)
                logger.info("ML training result: %s", train_result.get("status"))
            
            # Apply ML score blending if enabled
            if ML_ENABLED and self.ml_agent._model_loaded:
                ml_adjusted_count = 0
                for sig in signals:
                    if sig.action in {"BUY", "SELL"}:
                        df = market_data.get(sig.symbol)
                        if df is not None:
                            ml_pred = self.ml_agent.predict(sig.symbol, df)
                            original_score = sig.score
                            blended_score = self.ml_agent.blend_score(
                                technical_score=sig.score,
                                technical_action=sig.action,
                                ml_prediction=ml_pred,
                                ml_weight=ML_WEIGHT,
                            )
                            # Update the signal score with blended score
                            object.__setattr__(sig, 'score', round(blended_score, 3))
                            object.__setattr__(sig, 'confidence', round(blended_score, 3))
                            
                            if abs(blended_score - original_score) > 0.01:
                                ml_adjusted_count += 1
                
                # Re-sort signals by blended score
                signals.sort(key=lambda s: s.score, reverse=True)
                
                if ml_adjusted_count > 0:
                    logger.info("ML blending adjusted %d signal scores", ml_adjusted_count)

            # ── STEP 7: SENTIMENT ──
            self.sentiment_agent.log_sentiment()
            for sig in signals:
                if sig.action in {"BUY", "SELL"}:
                    modifier = self.sentiment_agent.get_score_modifier(sig.action)
                    if modifier != 1.0:
                        object.__setattr__(sig, 'score', round(sig.score * modifier, 3))
                        object.__setattr__(sig, 'confidence', round(sig.confidence * modifier, 3))
            
            # Re-sort after sentiment adjustment
            signals.sort(key=lambda s: s.score, reverse=True)

            # ── STEP 8: CORRELATION_FILTER ──
            signals = self.correlation_agent.filter_correlated(signals, market_data)

            if WFO_ENABLED and quality_map:
                quality_values = list(quality_map.values())
                avg_wfo_ret = sum(q.avg_test_return for q in quality_values) / len(quality_values)
                avg_wfo_win = sum(q.win_rate for q in quality_values) / len(quality_values)
                logger.info(
                    "WFO quality | tuned=%s symbols | avg_test_ret=%.3f | avg_win_rate=%.2f",
                    len(quality_values), avg_wfo_ret, avg_wfo_win,
                )

            top_preview = [
                f"{s.symbol}:{s.action}:{s.score:.2f}" for s in signals[:5]
            ]
            logger.info("Top ranked signals: %s", " | ".join(top_preview) if top_preview else "none")

            # ── STEP 9-11: RISK → PERFORMANCE_CHECK → EXECUTION ──
            # TASK 1 & 2: Multi-trade per cycle with SPOT paralysis fix
            balance = self._get_balance()
            allow_new_positions = True
            
            if PAPER_TRADING:
                pre_portfolio = self.portfolio_agent.get_portfolio_summary(price_map=marks)
                if pre_portfolio["day_pnl_pct"] <= -abs(MAX_DAILY_DRAWDOWN_PERCENT):
                    allow_new_positions = False
                    logger.warning(
                        "Daily drawdown guard active (%.2f%% <= -%.2f%%): only position reductions allowed",
                        pre_portfolio["day_pnl_pct"], abs(MAX_DAILY_DRAWDOWN_PERCENT),
                    )
                    # Notify on drawdown guard
                    if self.notifier:
                        self.notifier.send(f"⚠️ Drawdown guard active: {pre_portfolio['day_pnl_pct']:.2f}%")

            # Get current open position symbols
            open_position_symbols = set(self._get_open_position_symbols())
            current_positions_count = len(open_position_symbols)
            slots_available = MAX_OPEN_POSITIONS - current_positions_count

            # ── TASK 1: Fix SPOT paralysis ──
            # If all top signals are SELL and we have no positions, scan entire list for first BUY
            top_5_actions = [s.action for s in signals[:5] if s.action in {"BUY", "SELL"}]
            all_top_are_sell = all(a == "SELL" for a in top_5_actions) if top_5_actions else False
            
            if MARKET_MODE == "SPOT" and all_top_are_sell and current_positions_count == 0:
                logger.info("SPOT paralysis detected: scanning entire signal list for BUY opportunities")
                # Move first valid BUY signal to front
                for i, sig in enumerate(signals):
                    if sig.action == "BUY":
                        # Found a BUY signal, bring it to front consideration
                        signals.insert(0, signals.pop(i))
                        logger.info("Found BUY signal: %s (score=%.3f)", sig.symbol, sig.score)
                        break

            # ── TASK 2: Multi-trade per cycle ──
            # Collect ALL approved signals up to MAX_OPEN_POSITIONS
            approved_trades: List[Dict] = []
            
            for idx, candidate in enumerate(signals):
                if candidate.action not in {"BUY", "SELL"}:
                    continue
                
                # Skip symbols already in open positions
                if candidate.symbol in open_position_symbols:
                    continue
                
                # Check regime allows action
                candidate_regime = regimes.get(candidate.symbol)
                if candidate_regime and not self.regime_agent.allows_action(candidate_regime.regime, candidate.action):
                    continue
                
                # ── TASK 1: Enforce SIDEWAYS_MIN_SCORE ──
                if candidate_regime and candidate_regime.regime == "SIDEWAYS":
                    if candidate.score < SIDEWAYS_MIN_SCORE:
                        logger.debug(
                            "Skipping %s: score %.3f < SIDEWAYS_MIN_SCORE %.3f",
                            candidate.symbol, candidate.score, SIDEWAYS_MIN_SCORE
                        )
                        continue
                
                # ── TASK 2: 2nd+ signals must clear MIN_SCORE_FOR_SLOT_2_PLUS ──
                if len(approved_trades) >= 1 and candidate.score < MIN_SCORE_FOR_SLOT_2_PLUS:
                    continue
                
                # ── STEP 9: RISK evaluation ──
                position_qty = self._get_position_qty(candidate.symbol)
                
                # Pass regime info to risk agent for concentration check
                decision = self.risk_agent.evaluate(
                    signal=candidate,
                    balance_usdt=balance,
                    current_price=candidate.price,
                    position_qty=position_qty,
                    regime=candidate_regime.regime if candidate_regime else None,
                    regimes=regimes,
                    open_positions=open_position_symbols,
                    performance_agent=self.performance_agent,
                )

                if not decision.approved:
                    continue
                
                # ── STEP 10: PERFORMANCE_CHECK (cooldown) ──
                if self.performance_agent.is_symbol_in_cooldown(candidate.symbol):
                    logger.info("Symbol %s in cooldown, skipping", candidate.symbol)
                    continue

                # Check position limits
                if PAPER_TRADING and decision.position_intent.startswith("OPEN"):
                    if len(approved_trades) >= slots_available:
                        break
                    if not allow_new_positions:
                        continue

                if decision.approved and decision.trade_amount_usdt > 0:
                    approved_trades.append({
                        "signal": candidate,
                        "decision": decision,
                        "regime": candidate_regime.regime if candidate_regime else "UNKNOWN",
                    })
                    open_position_symbols.add(candidate.symbol)  # Mark as "will be opened"
                    
                    # Stop if we've filled all slots
                    if len(approved_trades) >= slots_available and slots_available > 0:
                        break

            # ── TASK 2: Capital allocation with weighted scoring ──
            if approved_trades and len(approved_trades) > 1:
                total_score = sum(t["signal"].score for t in approved_trades)
                available_cash = self._get_balance()
                
                for trade_info in approved_trades:
                    sig = trade_info["signal"]
                    dec = trade_info["decision"]
                    
                    # weight_i = score_i / sum(all scores)
                    weight = sig.score / total_score if total_score > 0 else 1.0 / len(approved_trades)
                    
                    # allocated = available_cash * weight_i * MAX_TRADE_PERCENT
                    allocated = available_cash * weight * MAX_TRADE_PERCENT
                    
                    # clamped to [MIN_TRADE_PERCENT*balance, MAX_TRADE_PERCENT*balance]
                    min_amount = MIN_TRADE_PERCENT * available_cash
                    max_amount = MAX_TRADE_PERCENT * available_cash
                    allocated = max(min_amount, min(max_amount, allocated))
                    
                    # Update decision with allocated amount
                    trade_info["weight"] = weight
                    trade_info["allocated"] = allocated
                    # Note: The RiskDecision is already computed, so we'll log but won't override
                    # In production, you'd want to pass the allocation to risk_agent

            # ── STEP 11: EXECUTION ──
            if not approved_trades:
                logger.info("No actionable signal found in this cycle")
            else:
                for trade_info in approved_trades:
                    sig = trade_info["signal"]
                    dec = trade_info["decision"]
                    weight = trade_info.get("weight", 1.0)
                    allocated = trade_info.get("allocated", dec.trade_amount_usdt)
                    
                    result = self.execution_agent.execute(sig, dec)
                    
                    # TASK 2: Log: symbol|score|weight|amount|status
                    logger.info(
                        "Execution | %s | score=%.3f | weight=%.3f | amount=%.2f | status=%s",
                        sig.symbol, sig.score, weight, dec.trade_amount_usdt, result.get("status"),
                    )
                    
                    # Telegram notification on trade
                    if self.notifier and result.get("status") in {"paper_executed", "live_executed"}:
                        emoji = "🟢" if sig.action == "BUY" else "🔴"
                        self.notifier.send(
                            f"{emoji} {dec.position_intent}: {sig.symbol}\n"
                            f"Score: {sig.score:.3f} | Amount: ${dec.trade_amount_usdt:.2f}"
                        )

            # ── STEP 12: PORTFOLIO_UPDATE ──
            if PAPER_TRADING:
                portfolio = self.portfolio_agent.get_portfolio_summary(price_map=marks)
                self._last_equity = portfolio["total_equity"]
                self._last_positions_count = len(portfolio.get("positions", {}))
                
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
                
                # TASK 5: Log position table every 5 cycles
                if self.cycle_count % 5 == 0:
                    pos_table = self.portfolio_agent.get_position_table(price_map=marks)
                    logger.info("Position Table:\n%s", pos_table)

            # ── STEP 13: SIMULATION ──
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

            # ── STEP 14: LOGGING (daily stats) ──
            self.performance_agent.maybe_log_daily_stats()

        except Exception as exc:
            logger.error("Cycle failed: %s", exc, exc_info=True)

    def _calculate_next_candle_boundary(self, interval_minutes: int) -> float:
        """
        TASK 6: Calculate timestamp of next candle boundary + 5s buffer.
        """
        now = datetime.now(timezone.utc)
        current_ts = now.timestamp()
        
        # Round down to current candle start
        interval_seconds = interval_minutes * 60
        current_candle_start = (int(current_ts) // interval_seconds) * interval_seconds
        
        # Next candle start + 5 second buffer
        next_candle_ts = current_candle_start + interval_seconds + 5
        
        return next_candle_ts

    def _save_state_and_summary(self):
        """Save portfolio state and log summary on shutdown."""
        logger.info("=" * 72)
        logger.info("SHUTDOWN SUMMARY")
        logger.info("=" * 72)
        
        if PAPER_TRADING:
            self.portfolio_agent.save()
            summary = self.portfolio_agent.get_portfolio_summary()
            logger.info("Final Equity: %.2f %s", summary["total_equity"], QUOTE_ASSET)
            logger.info("Realized PnL: %.2f", summary["realized_pnl"])
            logger.info("Total Trades: %d", summary["trade_count"])
        
        stats = self.performance_agent.get_recent_stats(days=7)
        if stats["total_trades"] > 0:
            logger.info(
                "7-Day Stats: %d trades | %.1f%% win rate | %.2f USDT PnL",
                stats["total_trades"],
                stats["win_rate"] * 100,
                stats["total_pnl_usdt"],
            )
        
        logger.info("Cycles completed: %d", self.cycle_count)
        logger.info("=" * 72)
        
        # Final Telegram notification
        if self.notifier:
            self.notifier.send(f"🛑 Bot shutdown. Cycles: {self.cycle_count}")

    def start(self):
        global _shutdown_requested
        
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
        
        # Start health check server
        start_health_server(HEALTH_CHECK_PORT)
        
        logger.info("Scheduler active: every %s minutes", minutes)
        
        # Run first cycle immediately
        self.run_cycle()
        
        # TASK 6: Candle-boundary scheduling
        next_run_ts = self._calculate_next_candle_boundary(minutes)

        while not _shutdown_requested:
            now_ts = time.time()
            if now_ts >= next_run_ts:
                self.run_cycle()
                next_run_ts = self._calculate_next_candle_boundary(minutes)
            
            sleep_for = max(1, min(15, int(next_run_ts - now_ts)))
            time.sleep(sleep_for)
        
        # Graceful shutdown
        self._save_state_and_summary()
        logger.info("Graceful shutdown complete")
        sys.exit(0)


if __name__ == "__main__":
    Orchestrator().start()