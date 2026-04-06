# agents/execution_agent.py
# Responsibility: Execute approved trades and track portfolio state.
# Delegates all portfolio state to PortfolioAgent.

import logging
import random
from datetime import datetime, timezone
from typing import Dict, List, Optional, TYPE_CHECKING

from agents.analysis_agent import Signal
from agents.risk_agent import RiskDecision
from config import (
    MARKET_MODE,
    PAPER_ORDER_TYPE,
    PAPER_TRADING,
    QUOTE_ASSET,
    SIMULATION_FEE_RATE,
    SLIPPAGE_STD,
)

if TYPE_CHECKING:
    from agents.portfolio_agent import PortfolioAgent
    from agents.performance_agent import PerformanceAgent

logger = logging.getLogger("ExecutionAgent")

# Maximum slippage cap (+/- 0.15%)
MAX_SLIPPAGE_PCT = 0.0015


class ExecutionAgent:
    """
    Handles paper and live execution for spot and optional futures mode.
    
    TASK 1 Features:
    - Slippage simulation: actual_price = price * (1 + random.gauss(0, SLIPPAGE_STD))
      capped at +/-0.15%, logs slippage_bps per trade
    - LIMIT order simulation (when PAPER_ORDER_TYPE="LIMIT"):
      fill only if next candle high/low crosses limit price
      if not filled in 1 candle: cancel, log "LIMIT_EXPIRED"
    - Delegates all portfolio state reads/writes to PortfolioAgent
    - Logs position table every 5 cycles via PortfolioAgent.get_position_table()
    """

    def __init__(
        self,
        binance_client=None,
        portfolio_agent: Optional["PortfolioAgent"] = None,
        performance_agent: Optional["PerformanceAgent"] = None,
    ):
        self.client = binance_client
        self.market_mode = MARKET_MODE
        self.portfolio_agent = portfolio_agent
        self.performance_agent = performance_agent
        
        # Pending limit orders: {symbol: {order details}}
        self.pending_limit_orders: Dict[str, Dict] = {}
        
        # Cycle counter for position table logging
        self._cycle_count = 0
        
        mode = "PAPER" if PAPER_TRADING else "LIVE"
        order_type = PAPER_ORDER_TYPE if PAPER_TRADING else "MARKET"
        logger.info(
            "ExecutionAgent ready | mode=%s | market=%s | order_type=%s | slippage_std=%.4f",
            mode, self.market_mode, order_type, SLIPPAGE_STD
        )

    def _simulate_slippage(self, price: float) -> tuple[float, float]:
        """
        Simulate market slippage with Gaussian noise.
        
        Returns:
            (actual_price, slippage_bps): The actual execution price and slippage in basis points.
        """
        # Generate slippage factor: mean=0, std=SLIPPAGE_STD
        slippage_factor = random.gauss(0, SLIPPAGE_STD)
        
        # Cap at +/- 0.15% (15 bps)
        slippage_factor = max(-MAX_SLIPPAGE_PCT, min(MAX_SLIPPAGE_PCT, slippage_factor))
        
        actual_price = price * (1 + slippage_factor)
        slippage_bps = slippage_factor * 10000  # Convert to basis points
        
        return actual_price, slippage_bps

    def get_position_qty(self, symbol: str) -> float:
        """Get position quantity from PortfolioAgent."""
        if self.portfolio_agent:
            return self.portfolio_agent.get_position_qty(symbol)
        return 0.0

    def get_available_cash(self) -> float:
        """Get available cash from PortfolioAgent."""
        if self.portfolio_agent:
            return self.portfolio_agent.get_cash()
        return 0.0

    def get_open_positions_count(self) -> int:
        """Get open positions count from PortfolioAgent."""
        if self.portfolio_agent:
            return self.portfolio_agent.get_open_positions_count()
        return 0

    def increment_cycle(self, price_map: Optional[Dict[str, float]] = None) -> None:
        """
        Increment cycle counter and log position table every 5 cycles.
        Called from Orchestrator each cycle.
        """
        self._cycle_count += 1
        
        # Log position table every 5 cycles
        if self._cycle_count % 5 == 0 and self.portfolio_agent:
            pos_table = self.portfolio_agent.get_position_table(price_map)
            logger.info("Position Table (cycle %d):\n%s", self._cycle_count, pos_table)

    def add_pending_limit_order(
        self, symbol: str, signal: Signal, decision: RiskDecision
    ) -> Dict:
        """
        Add a pending limit order to be checked on next candle.
        
        For LIMIT orders, we don't execute immediately. Instead, we store the
        order and check on the next candle if the high/low crosses the limit price.
        """
        self.pending_limit_orders[symbol] = {
            "signal": signal,
            "decision": decision,
            "limit_price": signal.price,
            "side": decision.position_intent,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        logger.info(
            "[LIMIT_PENDING] %s %s @ %.6f",
            decision.position_intent, symbol, signal.price
        )
        return {"status": "limit_pending", "symbol": symbol, "limit_price": signal.price}

    def process_pending_limit_orders(self, candle_data: Dict[str, Dict]) -> List[Dict]:
        """
        Process pending limit orders against new candle data.
        
        For each pending order:
        - BUY: fill if candle low <= limit_price
        - SELL: fill if candle high >= limit_price
        - If not filled, cancel and log "LIMIT_EXPIRED"
        
        Args:
            candle_data: Dict of {symbol: {high, low, close, ...}}
            
        Returns:
            List of execution results.
        """
        results = []
        symbols_to_remove = []
        
        for symbol, order in self.pending_limit_orders.items():
            candle = candle_data.get(symbol)
            if not candle:
                # No candle data, expire the order
                logger.warning("LIMIT_EXPIRED %s: No candle data", symbol)
                symbols_to_remove.append(symbol)
                results.append({
                    "status": "limit_expired",
                    "symbol": symbol,
                    "reason": "No candle data"
                })
                continue
            
            candle_high = float(candle.get("high", 0))
            candle_low = float(candle.get("low", 0))
            limit_price = order["limit_price"]
            intent = order["side"]
            
            filled = False
            
            # Check if limit order should fill
            if intent in {"OPEN_LONG", "CLOSE_SHORT"}:
                # BUY order: fills if candle low <= limit_price
                if candle_low <= limit_price:
                    filled = True
            elif intent in {"OPEN_SHORT", "CLOSE_LONG"}:
                # SELL order: fills if candle high >= limit_price
                if candle_high >= limit_price:
                    filled = True
            
            if filled:
                # Execute the order at limit price
                signal = order["signal"]
                decision = order["decision"]
                
                # Override signal price with limit price for execution
                filled_signal = Signal(
                    symbol=signal.symbol,
                    action=signal.action,
                    confidence=signal.confidence,
                    score=signal.score,
                    buy_score=signal.buy_score,
                    sell_score=signal.sell_score,
                    reason=signal.reason,
                    price=limit_price,  # Execute at limit price
                    rsi=signal.rsi,
                    macd=signal.macd,
                    macd_signal=signal.macd_signal,
                    atr_pct=signal.atr_pct,
                )
                
                if self.market_mode == "FUTURES":
                    result = self._paper_trade_futures(filled_signal, decision)
                else:
                    result = self._paper_trade_spot(filled_signal, decision)
                
                result["order_type"] = "LIMIT"
                result["limit_price"] = limit_price
                results.append(result)
                logger.info(
                    "[LIMIT_FILLED] %s %s @ %.6f",
                    intent, symbol, limit_price
                )
            else:
                # Order not filled, expire it
                logger.info(
                    "LIMIT_EXPIRED %s: high=%.6f low=%.6f limit=%.6f",
                    symbol, candle_high, candle_low, limit_price
                )
                results.append({
                    "status": "limit_expired",
                    "symbol": symbol,
                    "reason": f"Price did not cross limit ({limit_price:.6f})"
                })
            
            symbols_to_remove.append(symbol)
        
        # Clean up processed orders
        for symbol in symbols_to_remove:
            self.pending_limit_orders.pop(symbol, None)
        
        return results

    def execute(self, signal: Signal, decision: RiskDecision) -> dict:
        if not decision.approved or decision.trade_amount_usdt <= 0:
            logger.info("Execution skipped [%s]: %s", signal.symbol, decision.reason)
            return {"status": "skipped", "symbol": signal.symbol, "reason": decision.reason}

        if PAPER_TRADING:
            # Check if LIMIT order type is configured
            if PAPER_ORDER_TYPE.upper() == "LIMIT":
                return self.add_pending_limit_order(signal.symbol, signal, decision)
            
            if self.market_mode == "FUTURES":
                return self._paper_trade_futures(signal, decision)
            return self._paper_trade_spot(signal, decision)

        return self._live_trade(signal, decision)

    def _paper_trade_spot(self, signal: Signal, decision: RiskDecision) -> dict:
        """Execute a paper trade in SPOT mode using PortfolioAgent."""
        if not self.portfolio_agent:
            return {"status": "failed", "symbol": signal.symbol, "reason": "PortfolioAgent not initialized"}
        
        symbol = signal.symbol
        base_price = signal.price
        fee_rate = SIMULATION_FEE_RATE
        intent = decision.position_intent

        # Apply slippage simulation
        actual_price, slippage_bps = self._simulate_slippage(base_price)
        
        cash = self.portfolio_agent.get_cash()
        current_pos = self.portfolio_agent.get_position(symbol) or {
            "qty": 0.0,
            "avg_price": 0.0,
            "stop_loss": 0.0,
            "take_profit": 0.0,
            "trailing_stop": 0.0,
            "trailing_gap_pct": 0.015,
            "highest_price": 0.0,
            "lowest_price": 0.0,
            "leverage": 1,
            "margin": 0.0,
        }

        if intent == "OPEN_LONG":
            amount = decision.trade_amount_usdt
            if cash < amount:
                return {"status": "failed", "symbol": symbol, "reason": "Insufficient paper cash"}

            qty_bought = (amount * (1.0 - fee_rate)) / actual_price
            new_qty = float(current_pos.get("qty", 0.0)) + qty_bought
            old_qty = float(current_pos.get("qty", 0.0))
            old_avg = float(current_pos.get("avg_price", 0.0))
            new_avg = (
                ((old_qty * old_avg) + (qty_bought * actual_price)) / new_qty
                if new_qty > 0
                else 0.0
            )

            trailing_gap_pct = max(0.003, 1.0 - (decision.trailing_stop_price / actual_price)) if decision.trailing_stop_price > 0 else 0.015
            highest_price = max(float(current_pos.get("highest_price", 0.0)), actual_price)

            self.portfolio_agent.set_position(symbol, {
                "qty": new_qty,
                "avg_price": new_avg,
                "stop_loss": decision.stop_loss_price,
                "take_profit": decision.take_profit_price,
                "trailing_stop": decision.trailing_stop_price,
                "trailing_gap_pct": trailing_gap_pct,
                "highest_price": highest_price,
                "lowest_price": 0.0,
                "leverage": 1,
                "margin": 0.0,
            })
            self.portfolio_agent.adjust_cash(-amount)
            self.portfolio_agent.increment_trade_count()

            trade = {
                "time": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol,
                "intent": intent,
                "action": "BUY",
                "price": actual_price,
                "base_price": base_price,
                "slippage_bps": round(slippage_bps, 2),
                "qty": qty_bought,
                "notional": amount,
                "fee": amount * fee_rate,
            }
            # Note: We don't log OPEN trades to performance - only CLOSE trades with realized PnL

        elif intent == "CLOSE_LONG":
            qty_held = max(0.0, float(current_pos.get("qty", 0.0)))
            qty_to_sell = min(qty_held, decision.quantity)
            if qty_to_sell <= 0:
                return {"status": "failed", "symbol": symbol, "reason": "No long position to sell"}

            gross = qty_to_sell * actual_price
            net = gross * (1.0 - fee_rate)
            avg_price = float(current_pos.get("avg_price", actual_price))
            realized = (actual_price - avg_price) * qty_to_sell - (gross * fee_rate)

            remaining = qty_held - qty_to_sell
            if remaining > 0:
                current_pos["qty"] = remaining
                self.portfolio_agent.set_position(symbol, current_pos)
            else:
                self.portfolio_agent.remove_position(symbol)

            self.portfolio_agent.adjust_cash(net)
            self.portfolio_agent.add_realized_pnl(realized)
            self.portfolio_agent.increment_trade_count()

            # Calculate PnL percentage
            pnl_pct = ((actual_price - avg_price) / avg_price * 100.0) if avg_price > 0 else 0.0

            trade = {
                "time": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol,
                "intent": intent,
                "action": "SELL",
                "price": actual_price,
                "base_price": base_price,
                "slippage_bps": round(slippage_bps, 2),
                "qty": qty_to_sell,
                "notional": gross,
                "fee": gross * fee_rate,
                "realized_pnl": realized,
            }
            
            # Log trade to performance agent
            if self.performance_agent:
                self.performance_agent.log_trade(
                    symbol=symbol,
                    side="LONG",
                    entry_price=avg_price,
                    exit_price=actual_price,
                    pnl_pct=pnl_pct,
                    pnl_usdt=realized,
                    duration_hours=0.0,  # Could be calculated if we track entry time
                )
        else:
            return {"status": "skipped", "symbol": symbol, "reason": f"Unsupported spot intent: {intent}"}

        logger.info(
            "[PAPER-SPOT] %s %s qty=%.6f @ %.6f (slip=%+.1fbps) | cash=%.2f",
            intent,
            symbol,
            trade["qty"],
            actual_price,
            slippage_bps,
            self.portfolio_agent.get_cash(),
        )
        return {"status": "paper_executed", "symbol": symbol, "trade": trade}

    def _paper_trade_futures(self, signal: Signal, decision: RiskDecision) -> dict:
        """Execute a paper trade in FUTURES mode using PortfolioAgent."""
        if not self.portfolio_agent:
            return {"status": "failed", "symbol": signal.symbol, "reason": "PortfolioAgent not initialized"}
        
        symbol = signal.symbol
        base_price = signal.price
        fee_rate = SIMULATION_FEE_RATE
        intent = decision.position_intent

        # Apply slippage simulation
        actual_price, slippage_bps = self._simulate_slippage(base_price)

        pos = self.portfolio_agent.get_position(symbol) or {
            "qty": 0.0,
            "avg_price": 0.0,
            "stop_loss": 0.0,
            "take_profit": 0.0,
            "trailing_stop": 0.0,
            "trailing_gap_pct": 0.015,
            "highest_price": 0.0,
            "lowest_price": 0.0,
            "leverage": 1,
            "margin": 0.0,
        }

        if intent in {"OPEN_LONG", "OPEN_SHORT"}:
            margin = decision.trade_amount_usdt
            notional = decision.notional_usdt
            if self.portfolio_agent.get_cash() < margin:
                return {"status": "failed", "symbol": symbol, "reason": "Insufficient margin cash"}

            qty_delta = abs(decision.quantity)
            signed_qty_delta = qty_delta if intent == "OPEN_LONG" else -qty_delta

            if float(pos.get("qty", 0.0)) != 0 and (float(pos.get("qty", 0.0)) * signed_qty_delta) < 0:
                return {"status": "failed", "symbol": symbol, "reason": "Opposite position exists; close first"}

            fee = notional * fee_rate
            self.portfolio_agent.adjust_cash(-(margin + fee))
            self.portfolio_agent.adjust_used_margin(margin)

            prev_qty = float(pos.get("qty", 0.0))
            new_qty = prev_qty + signed_qty_delta
            if new_qty == 0:
                self.portfolio_agent.remove_position(symbol)
            else:
                weighted_avg = (
                    (abs(prev_qty) * float(pos.get("avg_price", 0.0)) + qty_delta * actual_price) / abs(new_qty)
                    if prev_qty != 0
                    else actual_price
                )
                trailing_gap = max(0.003, abs(1.0 - (decision.trailing_stop_price / actual_price))) if decision.trailing_stop_price > 0 else 0.015
                self.portfolio_agent.set_position(symbol, {
                    "qty": new_qty,
                    "avg_price": weighted_avg,
                    "stop_loss": decision.stop_loss_price,
                    "take_profit": decision.take_profit_price,
                    "trailing_stop": decision.trailing_stop_price,
                    "trailing_gap_pct": trailing_gap,
                    "highest_price": max(float(pos.get("highest_price", 0.0)), actual_price),
                    "lowest_price": actual_price if float(pos.get("lowest_price", 0.0)) == 0 else min(float(pos.get("lowest_price", actual_price)), actual_price),
                    "leverage": decision.leverage,
                    "margin": float(pos.get("margin", 0.0)) + margin,
                })

            self.portfolio_agent.increment_trade_count()

            trade = {
                "time": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol,
                "intent": intent,
                "action": "BUY" if intent == "OPEN_LONG" else "SELL",
                "price": actual_price,
                "base_price": base_price,
                "slippage_bps": round(slippage_bps, 2),
                "qty": qty_delta,
                "notional": notional,
                "margin": margin,
                "leverage": decision.leverage,
                "fee": fee,
            }

        elif intent in {"CLOSE_LONG", "CLOSE_SHORT"}:
            qty_existing = float(pos.get("qty", 0.0))
            if qty_existing == 0:
                return {"status": "failed", "symbol": symbol, "reason": "No futures position to close"}

            qty_to_close = min(abs(qty_existing), abs(decision.quantity))
            if qty_to_close <= 0:
                return {"status": "failed", "symbol": symbol, "reason": "Close quantity is zero"}

            gross = qty_to_close * actual_price
            fee = gross * fee_rate
            avg_price = float(pos.get("avg_price", actual_price))
            direction = 1.0 if qty_existing > 0 else -1.0
            pnl = (actual_price - avg_price) * qty_to_close * direction

            margin_used = float(pos.get("margin", 0.0))
            margin_release = margin_used * (qty_to_close / abs(qty_existing))
            self.portfolio_agent.adjust_cash(margin_release + pnl - fee)
            self.portfolio_agent.adjust_used_margin(-margin_release)
            self.portfolio_agent.add_realized_pnl(pnl - fee)

            remaining_qty = qty_existing - (qty_to_close * direction)
            if abs(remaining_qty) < 1e-12:
                self.portfolio_agent.remove_position(symbol)
            else:
                pos["qty"] = remaining_qty
                pos["margin"] = max(0.0, margin_used - margin_release)
                self.portfolio_agent.set_position(symbol, pos)

            self.portfolio_agent.increment_trade_count()

            # Calculate PnL percentage
            pnl_pct = ((actual_price - avg_price) / avg_price * 100.0 * direction) if avg_price > 0 else 0.0

            trade = {
                "time": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol,
                "intent": intent,
                "action": "SELL" if intent == "CLOSE_LONG" else "BUY",
                "price": actual_price,
                "base_price": base_price,
                "slippage_bps": round(slippage_bps, 2),
                "qty": qty_to_close,
                "notional": gross,
                "fee": fee,
                "realized_pnl": pnl - fee,
            }
            
            # Log trade to performance agent
            if self.performance_agent:
                side = "LONG" if intent == "CLOSE_LONG" else "SHORT"
                self.performance_agent.log_trade(
                    symbol=symbol,
                    side=side,
                    entry_price=avg_price,
                    exit_price=actual_price,
                    pnl_pct=pnl_pct,
                    pnl_usdt=pnl - fee,
                    duration_hours=0.0,
                )
        else:
            return {"status": "skipped", "symbol": symbol, "reason": f"Unsupported futures intent: {intent}"}

        logger.info(
            "[PAPER-FUTURES] %s %s qty=%.6f @ %.6f (slip=%+.1fbps) | cash=%.2f | used_margin=%.2f",
            intent,
            symbol,
            trade["qty"],
            actual_price,
            slippage_bps,
            self.portfolio_agent.get_cash(),
            self.portfolio_agent.get_used_margin(),
        )
        return {"status": "paper_executed", "symbol": symbol, "trade": trade}

    def _paper_close_position(self, symbol: str, price: float, reason: str) -> dict:
        """Force-close full paper position due to risk exit conditions."""
        qty = self.get_position_qty(symbol)
        if qty == 0:
            return {"status": "skipped", "symbol": symbol, "reason": "No position"}

        intent = "CLOSE_LONG" if qty > 0 else "CLOSE_SHORT"
        synthetic_decision = RiskDecision(
            symbol=symbol,
            approved=True,
            position_intent=intent,
            leverage=1,
            trade_amount_usdt=abs(qty) * price,
            notional_usdt=abs(qty) * price,
            quantity=abs(qty),
            stop_loss_price=0.0,
            take_profit_price=0.0,
            trailing_stop_price=0.0,
            reason=reason,
        )
        synthetic_signal = Signal(
            symbol=symbol,
            action="SELL" if qty > 0 else "BUY",
            confidence=1.0,
            score=1.0,
            buy_score=0.0,
            sell_score=0.0,
            reason=reason,
            price=price,
            rsi=50.0,
            macd=0.0,
            macd_signal=0.0,
            atr_pct=0.0,
        )
        result = (
            self._paper_trade_futures(synthetic_signal, synthetic_decision)
            if self.market_mode == "FUTURES"
            else self._paper_trade_spot(synthetic_signal, synthetic_decision)
        )
        if result.get("status") == "paper_executed":
            result["trade"]["exit_reason"] = reason
        return result

    def apply_risk_exits(self, price_map: Dict[str, float]) -> List[dict]:
        """Apply stop loss / take profit / trailing stop exits on paper positions."""
        if not PAPER_TRADING or not self.portfolio_agent:
            return []

        self.portfolio_agent.reset_day_if_needed()
        exits: List[dict] = []
        positions = self.portfolio_agent.get_all_positions()

        for symbol in list(positions.keys()):
            pos = positions.get(symbol, {})
            qty = float(pos.get("qty", 0.0))
            if qty == 0:
                continue

            price = price_map.get(symbol)
            if price is None:
                continue

            gap = max(0.003, float(pos.get("trailing_gap_pct", 0.015)))
            reason = None

            if qty > 0:
                highest_price = max(float(pos.get("highest_price", price)), price)
                trailing_stop = max(float(pos.get("trailing_stop", 0.0)), highest_price * (1.0 - gap))
                
                # Update position tracking values
                self.portfolio_agent.update_position_field(symbol, "highest_price", highest_price)
                self.portfolio_agent.update_position_field(symbol, "trailing_stop", trailing_stop)

                if float(pos.get("stop_loss", 0.0)) > 0 and price <= float(pos.get("stop_loss", 0.0)):
                    reason = "stop_loss"
                elif float(pos.get("take_profit", 0.0)) > 0 and price >= float(pos.get("take_profit", 0.0)):
                    reason = "take_profit"
                elif trailing_stop > 0 and price <= trailing_stop:
                    reason = "trailing_stop"
            else:
                lowest_price = price if float(pos.get("lowest_price", 0.0)) == 0 else min(float(pos.get("lowest_price", price)), price)
                trailing_stop = min(
                    float(pos.get("trailing_stop", price * (1.0 + gap))) or (price * (1.0 + gap)),
                    lowest_price * (1.0 + gap),
                )
                
                # Update position tracking values
                self.portfolio_agent.update_position_field(symbol, "lowest_price", lowest_price)
                self.portfolio_agent.update_position_field(symbol, "trailing_stop", trailing_stop)

                if float(pos.get("stop_loss", 0.0)) > 0 and price >= float(pos.get("stop_loss", 0.0)):
                    reason = "stop_loss"
                elif float(pos.get("take_profit", 0.0)) > 0 and price <= float(pos.get("take_profit", 0.0)):
                    reason = "take_profit"
                elif trailing_stop > 0 and price >= trailing_stop:
                    reason = "trailing_stop"

            if reason:
                exits.append(self._paper_close_position(symbol=symbol, price=price, reason=reason))

        return exits

    def _live_trade(self, signal: Signal, decision: RiskDecision) -> dict:
        if not self.client:
            raise ValueError("Binance client not provided for live trading")

        try:
            if self.market_mode == "FUTURES":
                self.client.futures_change_leverage(symbol=signal.symbol, leverage=decision.leverage)

                if decision.position_intent == "OPEN_LONG":
                    order = self.client.futures_create_order(
                        symbol=signal.symbol,
                        side="BUY",
                        type="MARKET",
                        quantity=decision.quantity,
                    )
                elif decision.position_intent == "OPEN_SHORT":
                    order = self.client.futures_create_order(
                        symbol=signal.symbol,
                        side="SELL",
                        type="MARKET",
                        quantity=decision.quantity,
                    )
                elif decision.position_intent == "CLOSE_LONG":
                    order = self.client.futures_create_order(
                        symbol=signal.symbol,
                        side="SELL",
                        type="MARKET",
                        quantity=decision.quantity,
                        reduceOnly=True,
                    )
                elif decision.position_intent == "CLOSE_SHORT":
                    order = self.client.futures_create_order(
                        symbol=signal.symbol,
                        side="BUY",
                        type="MARKET",
                        quantity=decision.quantity,
                        reduceOnly=True,
                    )
                else:
                    return {"status": "skipped", "symbol": signal.symbol, "reason": "Unsupported futures intent"}
            else:
                if decision.position_intent == "OPEN_LONG":
                    order = self.client.order_market_buy(
                        symbol=signal.symbol,
                        quoteOrderQty=decision.trade_amount_usdt,
                    )
                elif decision.position_intent == "CLOSE_LONG":
                    order = self.client.order_market_sell(
                        symbol=signal.symbol,
                        quantity=decision.quantity,
                    )
                else:
                    return {"status": "skipped", "symbol": signal.symbol, "reason": "Unsupported spot intent"}

            logger.info("[LIVE] %s executed on %s", decision.position_intent, signal.symbol)
            return {"status": "live_executed", "symbol": signal.symbol, "order": order}
        except Exception as exc:
            logger.error("Live order failed on %s: %s", signal.symbol, exc)
            return {"status": "failed", "symbol": signal.symbol, "reason": str(exc)}

    def get_portfolio_summary(self, price_map: Optional[Dict[str, float]] = None) -> dict:
        """Get portfolio summary from PortfolioAgent."""
        if self.portfolio_agent:
            return self.portfolio_agent.get_portfolio_summary(price_map)
        
        # Fallback empty summary
        return {
            "quote_asset": QUOTE_ASSET,
            "mode": self.market_mode,
            "cash": 0.0,
            "used_margin": 0.0,
            "positions": {},
            "positions_value": 0.0,
            "unrealized_pnl": 0.0,
            "total_equity": 0.0,
            "realized_pnl": 0.0,
            "daily_realized_pnl": 0.0,
            "day_pnl_pct": 0.0,
            "trade_count": 0,
        }
