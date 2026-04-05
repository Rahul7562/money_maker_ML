# agents/risk_agent.py
# Responsibility: Position sizing and account protection for every signal.

import logging
from dataclasses import dataclass

from agents.analysis_agent import Signal
from config import (
    FUTURES_DEFAULT_LEVERAGE,
    FUTURES_ENABLE_SHORTS,
    FUTURES_MAINT_MARGIN_BUFFER,
    FUTURES_MAX_LEVERAGE,
    MARKET_MODE,
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
    position_intent: str
    leverage: int
    trade_amount_usdt: float
    notional_usdt: float
    quantity: float
    stop_loss_price: float
    take_profit_price: float
    trailing_stop_price: float
    reason: str


class RiskAgent:
    """Applies hard risk rules and dynamic sizing before execution."""

    def __init__(self):
        self.market_mode = MARKET_MODE
        self.futures_shorts = FUTURES_ENABLE_SHORTS

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
                position_intent="NONE",
                leverage=1,
                trade_amount_usdt=0.0,
                notional_usdt=0.0,
                quantity=0.0,
                stop_loss_price=0.0,
                take_profit_price=0.0,
                trailing_stop_price=0.0,
                reason="HOLD signal",
            )

        if signal.confidence < MIN_CONFIDENCE:
            return self._reject(signal.symbol, f"Low confidence: {signal.confidence:.2f}")

        if signal.action == "BUY" and position_qty >= 0 and balance_usdt < MIN_BALANCE_USDT:
            return self._reject(
                signal.symbol,
                f"Balance too low: {balance_usdt:.2f} USDT < {MIN_BALANCE_USDT:.2f}",
            )

        if signal.action == "BUY":
            if position_qty < 0 and self.market_mode == "FUTURES":
                intent = "CLOSE_SHORT"
            else:
                intent = "OPEN_LONG"
        else:
            if position_qty > 0:
                intent = "CLOSE_LONG"
            elif self.market_mode == "FUTURES" and self.futures_shorts:
                intent = "OPEN_SHORT"
            else:
                return self._reject(signal.symbol, "No position to sell and shorting disabled")

        conf_scale = self._clamp((signal.confidence - MIN_CONFIDENCE) / (1.0 - MIN_CONFIDENCE), 0.0, 1.0)
        raw_fraction = MIN_TRADE_PERCENT + (MAX_TRADE_PERCENT - MIN_TRADE_PERCENT) * conf_scale

        vol_penalty = self._clamp(signal.atr_pct * 7.5, 0.0, 0.45)
        size_fraction = raw_fraction * (1.0 - vol_penalty)

        # Keep a floor so strong but volatile symbols are still tradable with small risk.
        size_fraction = max(size_fraction, MIN_TRADE_PERCENT * 0.75)

        leverage = 1
        trade_amount = 0.0
        notional_usdt = 0.0

        if intent in {"OPEN_LONG", "OPEN_SHORT"}:
            max_margin = balance_usdt * (1.0 - FUTURES_MAINT_MARGIN_BUFFER)
            trade_amount = round(min(balance_usdt * size_fraction, max_margin), 2)
            if trade_amount < 5.0:
                return self._reject(signal.symbol, "Trade amount below minimum notional")

            if self.market_mode == "FUTURES":
                leverage_raw = FUTURES_DEFAULT_LEVERAGE + int(round(conf_scale * (FUTURES_MAX_LEVERAGE - FUTURES_DEFAULT_LEVERAGE)))
                leverage = int(self._clamp(leverage_raw, 1, FUTURES_MAX_LEVERAGE))
            notional_usdt = round(trade_amount * leverage, 2)
        else:
            notional_usdt = round(abs(position_qty) * current_price, 2)
            trade_amount = notional_usdt

        sl_pct = max(STOP_LOSS_PERCENT, signal.atr_pct * TRAILING_STOP_MULTIPLIER)
        tp_pct = max(TAKE_PROFIT_PERCENT, sl_pct * 1.8)

        if intent == "OPEN_LONG":
            stop_loss = round(current_price * (1.0 - sl_pct), 8)
            take_profit = round(current_price * (1.0 + tp_pct), 8)
            trailing_stop = round(current_price * (1.0 - sl_pct * 0.7), 8)
            qty = round(notional_usdt / current_price, 8)
        elif intent == "OPEN_SHORT":
            stop_loss = round(current_price * (1.0 + sl_pct), 8)
            take_profit = round(current_price * (1.0 - tp_pct), 8)
            trailing_stop = round(current_price * (1.0 + sl_pct * 0.7), 8)
            qty = round(notional_usdt / current_price, 8)
        else:
            # Closing position now; no fresh SL/TP required.
            stop_loss = 0.0
            take_profit = 0.0
            trailing_stop = 0.0
            qty = round(abs(position_qty), 8)

        if qty <= 0:
            return self._reject(signal.symbol, "Computed quantity is zero")

        decision = RiskDecision(
            symbol=signal.symbol,
            approved=True,
            position_intent=intent,
            leverage=leverage,
            trade_amount_usdt=trade_amount,
            notional_usdt=notional_usdt,
            quantity=qty,
            stop_loss_price=stop_loss,
            take_profit_price=take_profit,
            trailing_stop_price=trailing_stop,
            reason=(
                f"Approved {intent} {signal.symbol} | conf={signal.confidence:.2f} "
                f"| lev={leverage}x | notional={notional_usdt:.2f} | SL={stop_loss} | TP={take_profit}"
            ),
        )
        logger.info("Risk -> APPROVED | %s", decision.reason)
        return decision

    def _reject(self, symbol: str, reason: str) -> RiskDecision:
        logger.warning("Risk -> REJECTED [%s] %s", symbol, reason)
        return RiskDecision(
            symbol=symbol,
            approved=False,
            position_intent="NONE",
            leverage=1,
            trade_amount_usdt=0.0,
            notional_usdt=0.0,
            quantity=0.0,
            stop_loss_price=0.0,
            take_profit_price=0.0,
            trailing_stop_price=0.0,
            reason=reason,
        )
