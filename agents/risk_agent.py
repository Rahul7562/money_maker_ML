# agents/risk_agent.py
# Responsibility: Position sizing and account protection for every signal.
# TASK 3: ATR-based stops, Kelly sizing, concentration checks, cooldown

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from agents.analysis_agent import Signal
from config import (
    ATR_STOP_MULTIPLIER,
    FUTURES_DEFAULT_LEVERAGE,
    FUTURES_ENABLE_SHORTS,
    FUTURES_MAINT_MARGIN_BUFFER,
    FUTURES_MAX_LEVERAGE,
    KELLY_ENABLED,
    KELLY_MIN_TRADES,
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
    """Decision output from risk evaluation."""
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

    def _calculate_atr_stop(self, atr_pct: float, is_long: bool, current_price: float) -> float:
        """
        TASK 3.1: ATR-based stop calculation.
        stop = ATR_STOP_MULTIPLIER * ATR, capped at STOP_LOSS_PERCENT maximum.
        """
        atr_stop_pct = ATR_STOP_MULTIPLIER * atr_pct
        # Cap at STOP_LOSS_PERCENT
        final_stop_pct = min(atr_stop_pct, STOP_LOSS_PERCENT)
        
        if is_long:
            return current_price * (1.0 - final_stop_pct)
        else:
            return current_price * (1.0 + final_stop_pct)

    def _calculate_kelly_fraction(self, performance_agent: Optional[Any]) -> float:
        """
        TASK 3.2: Kelly criterion position sizing.
        kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        Uses half-kelly, fallback to fixed % if <20 trades.
        Always capped at MAX_TRADE_PERCENT.
        """
        if not KELLY_ENABLED or performance_agent is None:
            return MAX_TRADE_PERCENT
        
        try:
            trade_count = performance_agent.get_trade_count()
            
            # Fallback to fixed % if insufficient trade history
            if trade_count < KELLY_MIN_TRADES:
                return MAX_TRADE_PERCENT
            
            # Get win rate and average win/loss
            stats = performance_agent.get_recent_stats(days=30)
            win_rate = stats.get("win_rate", 0.5)
            
            avg_win, avg_loss = performance_agent.get_avg_win_loss()
            
            if avg_win <= 0 or avg_loss <= 0:
                return MAX_TRADE_PERCENT
            
            # Kelly formula: (W * avg_win - L * avg_loss) / avg_win
            # Where W = win_rate, L = 1 - win_rate
            kelly = (win_rate * avg_win - (1.0 - win_rate) * avg_loss) / avg_win
            
            # Half-Kelly for safety
            half_kelly = kelly / 2.0
            
            # Clamp to valid range
            if half_kelly <= 0:
                return MIN_TRADE_PERCENT
            
            return self._clamp(half_kelly / 100.0, MIN_TRADE_PERCENT, MAX_TRADE_PERCENT)
            
        except Exception as e:
            logger.debug("Kelly calculation failed: %s, using default sizing", e)
            return MAX_TRADE_PERCENT

    def _check_concentration(
        self,
        symbol_regime: Optional[str],
        regimes: Optional[Dict[str, Any]],
        open_positions: Optional[Set[str]],
    ) -> float:
        """
        TASK 3.3: Concentration check.
        If >2 positions in the same regime, reduce size by 25%.
        Returns a multiplier (1.0 or 0.75).
        """
        if not regimes or not open_positions:
            return 1.0
        
        if symbol_regime is None:
            return 1.0
        
        # Count positions in the same regime
        same_regime_count = 0
        for pos_symbol in open_positions:
            pos_regime = regimes.get(pos_symbol)
            if pos_regime and pos_regime.regime == symbol_regime:
                same_regime_count += 1
        
        # If >2 positions in same regime, reduce size by 25%
        if same_regime_count >= 2:
            logger.info(
                "Concentration check: %d positions in %s regime, reducing size by 25%%",
                same_regime_count, symbol_regime
            )
            return 0.75
        
        return 1.0

    def _check_cooldown(self, symbol: str, performance_agent: Optional[Any]) -> bool:
        """
        TASK 3.4: Cooldown check.
        Reject if symbol has >= 3 consecutive losses.
        Returns True if should reject.
        """
        if performance_agent is None:
            return False
        
        try:
            consecutive_losses = performance_agent.get_symbol_consecutive_losses(symbol)
            if consecutive_losses >= 3:
                logger.warning(
                    "Cooldown active for %s: %d consecutive losses",
                    symbol, consecutive_losses
                )
                return True
            return False
        except Exception:
            return False

    def evaluate(
        self,
        signal: Signal,
        balance_usdt: float,
        current_price: float,
        position_qty: float = 0.0,
        regime: Optional[str] = None,
        regimes: Optional[Dict[str, Any]] = None,
        open_positions: Optional[Set[str]] = None,
        performance_agent: Optional[Any] = None,
    ) -> RiskDecision:
        """
        Evaluate a signal and return a risk decision.
        
        TASK 3 upgrades:
        1. ATR-based stop (capped at STOP_LOSS_PERCENT)
        2. Kelly sizing (half-kelly, fallback to fixed %)
        3. Concentration check (>2 same regime → size * 0.75)
        4. Cooldown (reject if >= 3 consecutive losses)
        """
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

        # TASK 3.4: Cooldown check
        if self._check_cooldown(signal.symbol, performance_agent):
            return self._reject(signal.symbol, "Symbol in cooldown (3+ consecutive losses)")

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

        # TASK 3.2: Kelly sizing
        kelly_fraction = self._calculate_kelly_fraction(performance_agent)
        
        conf_scale = self._clamp((signal.confidence - MIN_CONFIDENCE) / (1.0 - MIN_CONFIDENCE), 0.0, 1.0)
        raw_fraction = MIN_TRADE_PERCENT + (kelly_fraction - MIN_TRADE_PERCENT) * conf_scale

        vol_penalty = self._clamp(signal.atr_pct * 7.5, 0.0, 0.45)
        size_fraction = raw_fraction * (1.0 - vol_penalty)

        # Keep a floor so strong but volatile symbols are still tradable with small risk.
        size_fraction = max(size_fraction, MIN_TRADE_PERCENT * 0.75)
        
        # TASK 3.3: Concentration check
        concentration_multiplier = self._check_concentration(regime, regimes, open_positions)
        size_fraction *= concentration_multiplier

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

        # TASK 3.1: ATR-based stop loss
        is_long = intent != "OPEN_SHORT"
        atr_stop = self._calculate_atr_stop(signal.atr_pct, is_long, current_price)

        # Use ATR-based risk when available, with safe fallback bounds.
        atr_based_pct = signal.atr_pct * ATR_STOP_MULTIPLIER
        if atr_based_pct <= 0:
            atr_based_pct = STOP_LOSS_PERCENT
        sl_pct = self._clamp(atr_based_pct, 0.002, STOP_LOSS_PERCENT)
        tp_pct = max(TAKE_PROFIT_PERCENT, sl_pct * 1.8)

        if intent == "OPEN_LONG":
            stop_loss = round(
                atr_stop if 0 < atr_stop < current_price else current_price * (1.0 - sl_pct),
                8,
            )
            take_profit = round(current_price * (1.0 + tp_pct), 8)
            trailing_stop = round(current_price * (1.0 - sl_pct * 0.7), 8)
            qty = round(notional_usdt / current_price, 8)
        elif intent == "OPEN_SHORT":
            stop_loss = round(
                atr_stop if atr_stop > current_price else current_price * (1.0 + sl_pct),
                8,
            )
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
