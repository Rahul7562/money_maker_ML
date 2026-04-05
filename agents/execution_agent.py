# agents/execution_agent.py
# Responsibility: Execute approved trades and track portfolio state.

import logging
from datetime import datetime
from typing import Dict, List

from agents.analysis_agent import Signal
from agents.risk_agent import RiskDecision
from config import PAPER_STARTING_USDT, PAPER_TRADING, QUOTE_ASSET, SIMULATION_FEE_RATE

logger = logging.getLogger("ExecutionAgent")


class ExecutionAgent:
    """Handles paper and live execution for multi-symbol trading."""

    def __init__(self, binance_client=None):
        self.client = binance_client
        self.paper_portfolio = {
            "quote_asset": QUOTE_ASSET,
            "cash": PAPER_STARTING_USDT,
            "positions": {},  # symbol -> {qty, avg_price}
            "trades": [],
            "realized_pnl": 0.0,
        }
        mode = "PAPER" if PAPER_TRADING else "LIVE"
        logger.info("ExecutionAgent ready | mode=%s", mode)

    def get_position_qty(self, symbol: str) -> float:
        pos = self.paper_portfolio["positions"].get(symbol, {})
        return float(pos.get("qty", 0.0))

    def get_available_cash(self) -> float:
        return float(self.paper_portfolio["cash"])

    def execute(self, signal: Signal, decision: RiskDecision) -> dict:
        if not decision.approved or decision.trade_amount_usdt <= 0:
            logger.info("Execution skipped [%s]: %s", signal.symbol, decision.reason)
            return {"status": "skipped", "symbol": signal.symbol, "reason": decision.reason}

        if PAPER_TRADING:
            return self._paper_trade(signal, decision)
        return self._live_trade(signal, decision)

    def _paper_trade(self, signal: Signal, decision: RiskDecision) -> dict:
        symbol = signal.symbol
        price = signal.price
        fee_rate = SIMULATION_FEE_RATE

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
            },
        )

        if signal.action == "BUY":
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
            }
            self.paper_portfolio["cash"] = cash - amount

            trade = {
                "time": datetime.utcnow().isoformat(),
                "symbol": symbol,
                "action": "BUY",
                "price": price,
                "qty": qty_bought,
                "notional": amount,
                "fee": amount * fee_rate,
                "sl": decision.stop_loss_price,
                "tp": decision.take_profit_price,
            }

        elif signal.action == "SELL":
            qty_held = current_pos["qty"]
            qty_to_sell = min(qty_held, decision.quantity)
            if qty_to_sell <= 0:
                return {"status": "failed", "symbol": symbol, "reason": "No position to sell"}

            gross = qty_to_sell * price
            net = gross * (1.0 - fee_rate)
            avg_price = current_pos["avg_price"]
            realized = (price - avg_price) * qty_to_sell - (gross * fee_rate)

            remaining = qty_held - qty_to_sell
            if remaining > 0:
                current_pos["qty"] = remaining
                current_pos["avg_price"] = avg_price
                positions[symbol] = current_pos
            else:
                positions.pop(symbol, None)

            self.paper_portfolio["cash"] = cash + net
            self.paper_portfolio["realized_pnl"] += realized

            trade = {
                "time": datetime.utcnow().isoformat(),
                "symbol": symbol,
                "action": "SELL",
                "price": price,
                "qty": qty_to_sell,
                "notional": gross,
                "fee": gross * fee_rate,
                "realized_pnl": realized,
            }
        else:
            return {"status": "skipped", "symbol": symbol, "reason": "HOLD action"}

        self.paper_portfolio["trades"].append(trade)
        logger.info(
            "[PAPER] %s %s qty=%.6f @ %.6f | cash=%.2f",
            signal.action,
            symbol,
            trade["qty"],
            price,
            self.paper_portfolio["cash"],
        )
        return {"status": "paper_executed", "symbol": symbol, "trade": trade}

    def _paper_close_position(self, symbol: str, price: float, reason: str) -> dict:
        """Force-close a paper position due to risk exit conditions."""
        positions: Dict[str, Dict[str, float]] = self.paper_portfolio["positions"]
        pos = positions.get(symbol)
        if not pos:
            return {"status": "skipped", "symbol": symbol, "reason": "Position not found"}

        qty = float(pos.get("qty", 0.0))
        if qty <= 0:
            positions.pop(symbol, None)
            return {"status": "skipped", "symbol": symbol, "reason": "Empty position"}

        fee_rate = SIMULATION_FEE_RATE
        gross = qty * price
        net = gross * (1.0 - fee_rate)
        avg_price = float(pos.get("avg_price", price))
        realized = (price - avg_price) * qty - (gross * fee_rate)

        self.paper_portfolio["cash"] += net
        self.paper_portfolio["realized_pnl"] += realized
        positions.pop(symbol, None)

        trade = {
            "time": datetime.utcnow().isoformat(),
            "symbol": symbol,
            "action": "SELL",
            "price": price,
            "qty": qty,
            "notional": gross,
            "fee": gross * fee_rate,
            "realized_pnl": realized,
            "exit_reason": reason,
        }
        self.paper_portfolio["trades"].append(trade)
        logger.info("[PAPER] AUTO-EXIT %s qty=%.6f @ %.6f (%s)", symbol, qty, price, reason)
        return {"status": "paper_executed", "symbol": symbol, "trade": trade}

    def apply_risk_exits(self, price_map: Dict[str, float]) -> List[dict]:
        """Apply stop loss / take profit / trailing stop exits on paper positions."""
        if not PAPER_TRADING:
            return []

        exits: List[dict] = []
        positions: Dict[str, Dict[str, float]] = self.paper_portfolio["positions"]

        for symbol in list(positions.keys()):
            pos = positions.get(symbol, {})
            price = price_map.get(symbol)
            if price is None:
                continue

            highest_price = max(float(pos.get("highest_price", price)), price)
            gap = max(0.003, float(pos.get("trailing_gap_pct", 0.015)))
            trailing_stop = max(float(pos.get("trailing_stop", 0.0)), highest_price * (1.0 - gap))
            pos["highest_price"] = highest_price
            pos["trailing_stop"] = trailing_stop

            stop_loss = float(pos.get("stop_loss", 0.0))
            take_profit = float(pos.get("take_profit", 0.0))

            reason = None
            if stop_loss > 0 and price <= stop_loss:
                reason = "stop_loss"
            elif take_profit > 0 and price >= take_profit:
                reason = "take_profit"
            elif trailing_stop > 0 and price <= trailing_stop:
                reason = "trailing_stop"

            if reason:
                exits.append(self._paper_close_position(symbol=symbol, price=price, reason=reason))

        return exits

    def _live_trade(self, signal: Signal, decision: RiskDecision) -> dict:
        if not self.client:
            raise ValueError("Binance client not provided for live trading")

        try:
            if signal.action == "BUY":
                order = self.client.order_market_buy(
                    symbol=signal.symbol,
                    quoteOrderQty=decision.trade_amount_usdt,
                )
            elif signal.action == "SELL":
                order = self.client.order_market_sell(
                    symbol=signal.symbol,
                    quantity=decision.quantity,
                )
            else:
                return {"status": "skipped", "symbol": signal.symbol, "reason": "HOLD action"}

            logger.info("[LIVE] %s order placed on %s", signal.action, signal.symbol)
            return {"status": "live_executed", "symbol": signal.symbol, "order": order}
        except Exception as exc:
            logger.error("Live order failed on %s: %s", signal.symbol, exc)
            return {"status": "failed", "symbol": signal.symbol, "reason": str(exc)}

    def get_portfolio_summary(self, price_map: Dict[str, float] | None = None) -> dict:
        price_map = price_map or {}
        positions_value = 0.0
        snapshot_positions = {}

        for symbol, pos in self.paper_portfolio["positions"].items():
            mark_price = price_map.get(symbol, pos["avg_price"])
            value = pos["qty"] * mark_price
            positions_value += value
            snapshot_positions[symbol] = {
                "qty": pos["qty"],
                "avg_price": pos["avg_price"],
                "mark_price": mark_price,
                "value": value,
            }

        total_equity = self.paper_portfolio["cash"] + positions_value
        return {
            "quote_asset": self.paper_portfolio["quote_asset"],
            "cash": self.paper_portfolio["cash"],
            "positions": snapshot_positions,
            "positions_value": positions_value,
            "total_equity": total_equity,
            "realized_pnl": self.paper_portfolio["realized_pnl"],
            "trade_count": len(self.paper_portfolio["trades"]),
        }
