# agents/execution_agent.py
# Responsibility: Execute approved trades and track portfolio state.

import logging
from datetime import datetime, timezone
from typing import Dict, List

from agents.analysis_agent import Signal
from agents.risk_agent import RiskDecision
from config import (
    MARKET_MODE,
    PAPER_STARTING_USDT,
    PAPER_TRADING,
    QUOTE_ASSET,
    SIMULATION_FEE_RATE,
)

logger = logging.getLogger("ExecutionAgent")


class ExecutionAgent:
    """Handles paper and live execution for spot and optional futures mode."""

    def __init__(self, binance_client=None):
        self.client = binance_client
        self.market_mode = MARKET_MODE
        self.paper_portfolio = {
            "quote_asset": QUOTE_ASSET,
            "mode": self.market_mode,
            "cash": PAPER_STARTING_USDT,
            "used_margin": 0.0,
            "positions": {},  # symbol -> signed qty + metadata
            "trades": [],
            "realized_pnl": 0.0,
            "daily_realized_pnl": 0.0,
            "day_start_equity": PAPER_STARTING_USDT,
            "day": datetime.now(timezone.utc).date().isoformat(),
        }
        mode = "PAPER" if PAPER_TRADING else "LIVE"
        logger.info("ExecutionAgent ready | mode=%s | market=%s", mode, self.market_mode)

    def _reset_day_if_needed(self):
        today = datetime.now(timezone.utc).date().isoformat()
        if self.paper_portfolio["day"] != today:
            summary = self.get_portfolio_summary()
            self.paper_portfolio["day"] = today
            self.paper_portfolio["day_start_equity"] = summary["total_equity"]
            self.paper_portfolio["daily_realized_pnl"] = 0.0

    def get_position_qty(self, symbol: str) -> float:
        pos = self.paper_portfolio["positions"].get(symbol, {})
        return float(pos.get("qty", 0.0))

    def get_available_cash(self) -> float:
        return float(self.paper_portfolio["cash"])

    def get_open_positions_count(self) -> int:
        return len(self.paper_portfolio["positions"])

    def execute(self, signal: Signal, decision: RiskDecision) -> dict:
        if not decision.approved or decision.trade_amount_usdt <= 0:
            logger.info("Execution skipped [%s]: %s", signal.symbol, decision.reason)
            return {"status": "skipped", "symbol": signal.symbol, "reason": decision.reason}

        if PAPER_TRADING:
            if self.market_mode == "FUTURES":
                return self._paper_trade_futures(signal, decision)
            return self._paper_trade_spot(signal, decision)

        return self._live_trade(signal, decision)

    def _paper_trade_spot(self, signal: Signal, decision: RiskDecision) -> dict:
        symbol = signal.symbol
        price = signal.price
        fee_rate = SIMULATION_FEE_RATE
        intent = decision.position_intent

        cash = self.paper_portfolio["cash"]
        positions: Dict[str, Dict[str, float]] = self.paper_portfolio["positions"]
        current_pos = positions.get(
            symbol,
            {
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
            },
        )

        if intent == "OPEN_LONG":
            amount = decision.trade_amount_usdt
            if cash < amount:
                return {"status": "failed", "symbol": symbol, "reason": "Insufficient paper cash"}

            qty_bought = (amount * (1.0 - fee_rate)) / price
            new_qty = current_pos["qty"] + qty_bought
            new_avg = (
                ((current_pos["qty"] * current_pos["avg_price"]) + (qty_bought * price)) / new_qty
                if new_qty > 0
                else 0.0
            )

            trailing_gap_pct = max(0.003, 1.0 - (decision.trailing_stop_price / price))
            highest_price = max(current_pos.get("highest_price", 0.0), price)

            positions[symbol] = {
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
            }
            self.paper_portfolio["cash"] = cash - amount

            trade = {
                "time": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol,
                "intent": intent,
                "action": "BUY",
                "price": price,
                "qty": qty_bought,
                "notional": amount,
                "fee": amount * fee_rate,
            }

        elif intent == "CLOSE_LONG":
            qty_held = max(0.0, current_pos["qty"])
            qty_to_sell = min(qty_held, decision.quantity)
            if qty_to_sell <= 0:
                return {"status": "failed", "symbol": symbol, "reason": "No long position to sell"}

            gross = qty_to_sell * price
            net = gross * (1.0 - fee_rate)
            avg_price = current_pos["avg_price"]
            realized = (price - avg_price) * qty_to_sell - (gross * fee_rate)

            remaining = qty_held - qty_to_sell
            if remaining > 0:
                current_pos["qty"] = remaining
                positions[symbol] = current_pos
            else:
                positions.pop(symbol, None)

            self.paper_portfolio["cash"] = cash + net
            self.paper_portfolio["realized_pnl"] += realized
            self.paper_portfolio["daily_realized_pnl"] += realized

            trade = {
                "time": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol,
                "intent": intent,
                "action": "SELL",
                "price": price,
                "qty": qty_to_sell,
                "notional": gross,
                "fee": gross * fee_rate,
                "realized_pnl": realized,
            }
        else:
            return {"status": "skipped", "symbol": symbol, "reason": f"Unsupported spot intent: {intent}"}

        self.paper_portfolio["trades"].append(trade)
        logger.info(
            "[PAPER-SPOT] %s %s qty=%.6f @ %.6f | cash=%.2f",
            intent,
            symbol,
            trade["qty"],
            price,
            self.paper_portfolio["cash"],
        )
        return {"status": "paper_executed", "symbol": symbol, "trade": trade}

    def _paper_trade_futures(self, signal: Signal, decision: RiskDecision) -> dict:
        symbol = signal.symbol
        price = signal.price
        fee_rate = SIMULATION_FEE_RATE
        intent = decision.position_intent

        positions: Dict[str, Dict[str, float]] = self.paper_portfolio["positions"]
        pos = positions.get(
            symbol,
            {
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
            },
        )

        if intent in {"OPEN_LONG", "OPEN_SHORT"}:
            margin = decision.trade_amount_usdt
            notional = decision.notional_usdt
            if self.paper_portfolio["cash"] < margin:
                return {"status": "failed", "symbol": symbol, "reason": "Insufficient margin cash"}

            qty_delta = abs(decision.quantity)
            signed_qty_delta = qty_delta if intent == "OPEN_LONG" else -qty_delta

            if pos["qty"] != 0 and (pos["qty"] * signed_qty_delta) < 0:
                return {"status": "failed", "symbol": symbol, "reason": "Opposite position exists; close first"}

            fee = notional * fee_rate
            self.paper_portfolio["cash"] -= (margin + fee)
            self.paper_portfolio["used_margin"] += margin

            prev_qty = pos["qty"]
            new_qty = prev_qty + signed_qty_delta
            if new_qty == 0:
                positions.pop(symbol, None)
            else:
                weighted_avg = (
                    (abs(prev_qty) * pos["avg_price"] + qty_delta * price) / abs(new_qty)
                    if prev_qty != 0
                    else price
                )
                trailing_gap = max(0.003, abs(1.0 - (decision.trailing_stop_price / price)))
                positions[symbol] = {
                    "qty": new_qty,
                    "avg_price": weighted_avg,
                    "stop_loss": decision.stop_loss_price,
                    "take_profit": decision.take_profit_price,
                    "trailing_stop": decision.trailing_stop_price,
                    "trailing_gap_pct": trailing_gap,
                    "highest_price": max(pos.get("highest_price", 0.0), price),
                    "lowest_price": price if pos.get("lowest_price", 0.0) == 0 else min(pos.get("lowest_price", price), price),
                    "leverage": decision.leverage,
                    "margin": pos.get("margin", 0.0) + margin,
                }

            trade = {
                "time": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol,
                "intent": intent,
                "action": "BUY" if intent == "OPEN_LONG" else "SELL",
                "price": price,
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

            gross = qty_to_close * price
            fee = gross * fee_rate
            avg_price = float(pos.get("avg_price", price))
            direction = 1.0 if qty_existing > 0 else -1.0
            pnl = (price - avg_price) * qty_to_close * direction

            margin_used = float(pos.get("margin", 0.0))
            margin_release = margin_used * (qty_to_close / abs(qty_existing))
            self.paper_portfolio["cash"] += (margin_release + pnl - fee)
            self.paper_portfolio["used_margin"] -= margin_release
            self.paper_portfolio["realized_pnl"] += pnl - fee
            self.paper_portfolio["daily_realized_pnl"] += pnl - fee

            remaining_qty = qty_existing - (qty_to_close * direction)
            if abs(remaining_qty) < 1e-12:
                positions.pop(symbol, None)
            else:
                pos["qty"] = remaining_qty
                pos["margin"] = max(0.0, margin_used - margin_release)
                positions[symbol] = pos

            trade = {
                "time": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol,
                "intent": intent,
                "action": "SELL" if intent == "CLOSE_LONG" else "BUY",
                "price": price,
                "qty": qty_to_close,
                "notional": gross,
                "fee": fee,
                "realized_pnl": pnl - fee,
            }
        else:
            return {"status": "skipped", "symbol": symbol, "reason": f"Unsupported futures intent: {intent}"}

        self.paper_portfolio["trades"].append(trade)
        logger.info(
            "[PAPER-FUTURES] %s %s qty=%.6f @ %.6f | cash=%.2f | used_margin=%.2f",
            intent,
            symbol,
            trade["qty"],
            price,
            self.paper_portfolio["cash"],
            self.paper_portfolio["used_margin"],
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
        if not PAPER_TRADING:
            return []

        self._reset_day_if_needed()
        exits: List[dict] = []
        positions: Dict[str, Dict[str, float]] = self.paper_portfolio["positions"]

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
                pos["highest_price"] = highest_price
                pos["trailing_stop"] = trailing_stop

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
                pos["lowest_price"] = lowest_price
                pos["trailing_stop"] = trailing_stop

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

    def get_portfolio_summary(self, price_map: Dict[str, float] | None = None) -> dict:
        price_map = price_map or {}
        self._reset_day_if_needed()

        positions_metric = 0.0
        unrealized_pnl = 0.0
        snapshot_positions = {}

        for symbol, pos in self.paper_portfolio["positions"].items():
            mark_price = float(price_map.get(symbol, pos.get("avg_price", 0.0)))
            qty = float(pos.get("qty", 0.0))
            avg = float(pos.get("avg_price", mark_price))

            if self.market_mode == "FUTURES":
                pnl = (mark_price - avg) * qty
                unrealized_pnl += pnl
                positions_metric += abs(qty) * mark_price
                display_value = pnl
            else:
                display_value = qty * mark_price
                positions_metric += display_value

            snapshot_positions[symbol] = {
                "qty": qty,
                "avg_price": avg,
                "mark_price": mark_price,
                "value": display_value,
            }

        if self.market_mode == "FUTURES":
            total_equity = self.paper_portfolio["cash"] + self.paper_portfolio["used_margin"] + unrealized_pnl
        else:
            total_equity = self.paper_portfolio["cash"] + positions_metric

        day_start = float(self.paper_portfolio.get("day_start_equity", total_equity))
        day_pnl_pct = ((total_equity - day_start) / day_start * 100.0) if day_start > 0 else 0.0

        return {
            "quote_asset": self.paper_portfolio["quote_asset"],
            "mode": self.market_mode,
            "cash": self.paper_portfolio["cash"],
            "used_margin": self.paper_portfolio["used_margin"],
            "positions": snapshot_positions,
            "positions_value": positions_metric,
            "unrealized_pnl": unrealized_pnl,
            "total_equity": total_equity,
            "realized_pnl": self.paper_portfolio["realized_pnl"],
            "daily_realized_pnl": self.paper_portfolio["daily_realized_pnl"],
            "day_pnl_pct": day_pnl_pct,
            "trade_count": len(self.paper_portfolio["trades"]),
        }
