# agents/risk_agent.py
# Responsibility: Position sizing and account protection for every signal.

import logging
from dataclasses import dataclass

from agents.analysis_agent import Signal
from config import (
    MAX_TRADE_PERCENT,
    MIN_BALANCE_USDT,
    MIN_CONFIDENCE,
    MIN_TRADE_PERCENT,
    STOP_LOSS_PERCENT,
    TAKE_PROFIT_PERCENT,
    TRAILING_STOP_MULTIPLIER,
)

logger = logging.getLogger("RiskAgent")


@dataclass
class RiskDecision:
    symbol: str
    approved: bool
    trade_amount_usdt: float
    quantity: float
    stop_loss_price: float
    take_profit_price: float
    trailing_stop_price: float
    reason: str


class RiskAgent:
    """Applies hard risk rules and dynamic sizing before execution."""

    def _clamp(self, value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def evaluate(
        self,
        signal: Signal,
        balance_usdt: float,
        current_price: float,
        position_qty: float = 0.0,
    ) -> RiskDecision:
        if signal.action == "HOLD":
            return RiskDecision(
                symbol=signal.symbol,
                approved=True,
                trade_amount_usdt=0.0,
                quantity=0.0,
                stop_loss_price=0.0,
                take_profit_price=0.0,
                trailing_stop_price=0.0,
                reason="HOLD signal",
            )

        if signal.confidence < MIN_CONFIDENCE:
            return self._reject(signal.symbol, f"Low confidence: {signal.confidence:.2f}")

        if signal.action == "BUY" and balance_usdt < MIN_BALANCE_USDT:
            return self._reject(
                signal.symbol,
                f"Balance too low: {balance_usdt:.2f} USDT < {MIN_BALANCE_USDT:.2f}",
            )

        if signal.action == "SELL" and position_qty <= 0:
            return self._reject(signal.symbol, "No existing position to sell")

        conf_scale = self._clamp((signal.confidence - MIN_CONFIDENCE) / (1.0 - MIN_CONFIDENCE), 0.0, 1.0)
        raw_fraction = MIN_TRADE_PERCENT + (MAX_TRADE_PERCENT - MIN_TRADE_PERCENT) * conf_scale

        vol_penalty = self._clamp(signal.atr_pct * 7.5, 0.0, 0.45)
        size_fraction = raw_fraction * (1.0 - vol_penalty)

        # Keep a floor so strong but volatile symbols are still tradable with small risk.
        size_fraction = max(size_fraction, MIN_TRADE_PERCENT * 0.75)
        trade_amount = round(balance_usdt * size_fraction, 2)

        if signal.action == "BUY" and (trade_amount < 5.0 or trade_amount > balance_usdt):
            return self._reject(
                signal.symbol,
                f"Invalid BUY amount {trade_amount:.2f} for balance {balance_usdt:.2f}",
            )

        if signal.action == "SELL":
            notional = position_qty * current_price
            trade_amount = round(max(5.0, min(notional, balance_usdt * MAX_TRADE_PERCENT)), 2)

        sl_pct = max(STOP_LOSS_PERCENT, signal.atr_pct * TRAILING_STOP_MULTIPLIER)
        tp_pct = max(TAKE_PROFIT_PERCENT, sl_pct * 1.8)

        if signal.action == "BUY":
            stop_loss = round(current_price * (1.0 - sl_pct), 8)
            take_profit = round(current_price * (1.0 + tp_pct), 8)
            trailing_stop = round(current_price * (1.0 - sl_pct * 0.7), 8)
            qty = round(trade_amount / current_price, 8)
        else:
            stop_loss = round(current_price * (1.0 + sl_pct), 8)
            take_profit = round(current_price * (1.0 - tp_pct), 8)
            trailing_stop = round(current_price * (1.0 + sl_pct * 0.7), 8)
            qty = round(min(position_qty, trade_amount / current_price), 8)

        decision = RiskDecision(
            symbol=signal.symbol,
            approved=True,
            trade_amount_usdt=trade_amount,
            quantity=qty,
            stop_loss_price=stop_loss,
            take_profit_price=take_profit,
            trailing_stop_price=trailing_stop,
            reason=(
                f"Approved {signal.action} {signal.symbol} | conf={signal.confidence:.2f} "
                f"| size={size_fraction*100:.1f}% | SL={stop_loss} | TP={take_profit}"
            ),
        )
        logger.info("Risk -> APPROVED | %s", decision.reason)
        return decision

    def _reject(self, symbol: str, reason: str) -> RiskDecision:
        logger.warning("Risk -> REJECTED [%s] %s", symbol, reason)
        return RiskDecision(
            symbol=symbol,
            approved=False,
            trade_amount_usdt=0.0,
            quantity=0.0,
            stop_loss_price=0.0,
            take_profit_price=0.0,
            trailing_stop_price=0.0,
            reason=reason,
        )
