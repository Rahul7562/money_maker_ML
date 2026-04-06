# agents/portfolio_agent.py
# Responsibility: Centralize all portfolio state with JSON persistence.

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from config import (
    MARKET_MODE,
    PAPER_STARTING_USDT,
    QUOTE_ASSET,
    STATE_DIR,
)

logger = logging.getLogger("PortfolioAgent")


class PortfolioAgent:
    """
    Centralizes all portfolio state tracking with persistence to JSON.
    
    Tracks: cash, positions, realized_pnl, trade_count, day_start_equity.
    Persists to state/portfolio.json after every change.
    Loads from state/portfolio.json on startup.
    """

    def __init__(self) -> None:
        """Initialize PortfolioAgent with state persistence."""
        self.state_dir = Path(STATE_DIR)
        self.portfolio_file = self.state_dir / "portfolio.json"
        self.market_mode = MARKET_MODE
        self.quote_asset = QUOTE_ASSET
        
        # Ensure state directory exists
        self._ensure_state_dir()
        
        # Initialize portfolio state
        self.portfolio: Dict[str, Any] = self._load_or_create_portfolio()
        
        logger.info(
            "PortfolioAgent ready | mode=%s | cash=%.2f | positions=%d",
            self.market_mode,
            self.portfolio["cash"],
            len(self.portfolio["positions"]),
        )

    def _ensure_state_dir(self) -> None:
        """Create state directory if it doesn't exist."""
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def _load_or_create_portfolio(self) -> Dict[str, Any]:
        """Load portfolio from JSON or create new one."""
        if self.portfolio_file.exists():
            try:
                with open(self.portfolio_file, "r", encoding="utf-8") as f:
                    portfolio = json.load(f)
                logger.info("Loaded portfolio from %s", self.portfolio_file)
                # Ensure all required fields exist
                return self._validate_portfolio(portfolio)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("Failed to load portfolio: %s, creating new", e)
        
        return self._create_new_portfolio()

    def _create_new_portfolio(self) -> Dict[str, Any]:
        """Create a fresh portfolio state."""
        return {
            "quote_asset": self.quote_asset,
            "mode": self.market_mode,
            "cash": PAPER_STARTING_USDT,
            "used_margin": 0.0,
            "positions": {},
            "realized_pnl": 0.0,
            "daily_realized_pnl": 0.0,
            "day_start_equity": PAPER_STARTING_USDT,
            "day": datetime.now(timezone.utc).date().isoformat(),
            "trade_count": 0,
        }

    def _validate_portfolio(self, portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure portfolio has all required fields."""
        defaults = self._create_new_portfolio()
        for key, default_value in defaults.items():
            if key not in portfolio:
                portfolio[key] = default_value
        return portfolio

    def save(self) -> None:
        """Persist portfolio state to JSON file."""
        self._ensure_state_dir()
        try:
            with open(self.portfolio_file, "w", encoding="utf-8") as f:
                json.dump(self.portfolio, f, indent=2, default=str)
            logger.debug("Portfolio saved to %s", self.portfolio_file)
        except IOError as e:
            logger.error("Failed to save portfolio: %s", e)

    def reset_day_if_needed(self) -> None:
        """Reset daily stats if a new day has started."""
        today = datetime.now(timezone.utc).date().isoformat()
        if self.portfolio["day"] != today:
            summary = self.get_portfolio_summary()
            self.portfolio["day"] = today
            self.portfolio["day_start_equity"] = summary["total_equity"]
            self.portfolio["daily_realized_pnl"] = 0.0
            self.save()

    def get_cash(self) -> float:
        """Get available cash."""
        return float(self.portfolio["cash"])

    def set_cash(self, amount: float) -> None:
        """Set cash amount and persist."""
        self.portfolio["cash"] = amount
        self.save()

    def adjust_cash(self, delta: float) -> None:
        """Adjust cash by delta amount and persist."""
        self.portfolio["cash"] += delta
        self.save()

    def get_used_margin(self) -> float:
        """Get used margin (futures mode)."""
        return float(self.portfolio.get("used_margin", 0.0))

    def set_used_margin(self, amount: float) -> None:
        """Set used margin and persist."""
        self.portfolio["used_margin"] = amount
        self.save()

    def adjust_used_margin(self, delta: float) -> None:
        """Adjust used margin by delta."""
        self.portfolio["used_margin"] = self.portfolio.get("used_margin", 0.0) + delta
        self.save()

    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get position data for a symbol, or None if not exists."""
        return self.portfolio["positions"].get(symbol)

    def get_position_qty(self, symbol: str) -> float:
        """Get position quantity for a symbol (signed for futures)."""
        pos = self.get_position(symbol)
        return float(pos.get("qty", 0.0)) if pos else 0.0

    def get_all_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get all open positions."""
        return self.portfolio["positions"].copy()

    def get_open_positions_count(self) -> int:
        """Get count of open positions."""
        return len(self.portfolio["positions"])

    def get_position_symbols(self) -> list:
        """Get list of symbols with open positions."""
        return list(self.portfolio["positions"].keys())

    def set_position(self, symbol: str, position_data: Dict[str, Any]) -> None:
        """Set/update a position and persist."""
        position_data["entry_time"] = position_data.get(
            "entry_time", datetime.now(timezone.utc).isoformat()
        )
        self.portfolio["positions"][symbol] = position_data
        self.save()

    def remove_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Remove a position and persist. Returns removed position or None."""
        pos = self.portfolio["positions"].pop(symbol, None)
        if pos:
            self.save()
        return pos

    def update_position_field(self, symbol: str, field: str, value: Any) -> None:
        """Update a single field in a position."""
        if symbol in self.portfolio["positions"]:
            self.portfolio["positions"][symbol][field] = value
            self.save()

    def add_realized_pnl(self, pnl: float) -> None:
        """Add to realized PnL (both total and daily)."""
        self.portfolio["realized_pnl"] += pnl
        self.portfolio["daily_realized_pnl"] += pnl
        self.save()

    def get_realized_pnl(self) -> float:
        """Get total realized PnL."""
        return float(self.portfolio["realized_pnl"])

    def get_daily_realized_pnl(self) -> float:
        """Get daily realized PnL."""
        return float(self.portfolio.get("daily_realized_pnl", 0.0))

    def increment_trade_count(self) -> int:
        """Increment trade count and return new value."""
        self.portfolio["trade_count"] = self.portfolio.get("trade_count", 0) + 1
        self.save()
        return self.portfolio["trade_count"]

    def get_trade_count(self) -> int:
        """Get total trade count."""
        return int(self.portfolio.get("trade_count", 0))

    def get_portfolio_summary(self, price_map: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Calculate and return a full portfolio summary.
        
        Args:
            price_map: Dict mapping symbols to current prices.
            
        Returns:
            Dict with cash, positions_value, unrealized_pnl, total_equity, etc.
        """
        self.reset_day_if_needed()
        price_map = price_map or {}
        
        positions_metric = 0.0
        unrealized_pnl = 0.0
        snapshot_positions = {}

        for symbol, pos in self.portfolio["positions"].items():
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
                "side": "LONG" if qty > 0 else "SHORT" if qty < 0 else "FLAT",
                "stop_loss": pos.get("stop_loss", 0.0),
                "take_profit": pos.get("take_profit", 0.0),
                "entry_time": pos.get("entry_time"),
            }

        if self.market_mode == "FUTURES":
            total_equity = self.portfolio["cash"] + self.portfolio.get("used_margin", 0.0) + unrealized_pnl
        else:
            total_equity = self.portfolio["cash"] + positions_metric

        day_start = float(self.portfolio.get("day_start_equity", total_equity))
        day_pnl_pct = ((total_equity - day_start) / day_start * 100.0) if day_start > 0 else 0.0

        return {
            "quote_asset": self.portfolio["quote_asset"],
            "mode": self.market_mode,
            "cash": self.portfolio["cash"],
            "used_margin": self.portfolio.get("used_margin", 0.0),
            "positions": snapshot_positions,
            "positions_value": positions_metric,
            "unrealized_pnl": unrealized_pnl,
            "total_equity": total_equity,
            "realized_pnl": self.portfolio["realized_pnl"],
            "daily_realized_pnl": self.portfolio.get("daily_realized_pnl", 0.0),
            "day_pnl_pct": day_pnl_pct,
            "trade_count": self.portfolio.get("trade_count", 0),
        }

    def get_position_table(self, price_map: Optional[Dict[str, float]] = None) -> str:
        """
        Generate a formatted position table string for logging.
        
        Format: Symbol | Side | Entry | Current | PnL% | Stop | Target | Age(h)
        
        Args:
            price_map: Dict mapping symbols to current prices.
            
        Returns:
            Formatted string table of positions.
        """
        price_map = price_map or {}
        
        if not self.portfolio["positions"]:
            return "No open positions"

        lines = ["Symbol | Side | Entry | Current | PnL% | Stop | Target | Age(h)"]
        lines.append("-" * 70)

        now = datetime.now(timezone.utc)
        
        for symbol, pos in self.portfolio["positions"].items():
            qty = float(pos.get("qty", 0.0))
            side = "LONG" if qty > 0 else "SHORT"
            entry = float(pos.get("avg_price", 0.0))
            current = float(price_map.get(symbol, entry))
            
            if side == "LONG":
                pnl_pct = ((current - entry) / entry * 100.0) if entry > 0 else 0.0
            else:
                pnl_pct = ((entry - current) / entry * 100.0) if entry > 0 else 0.0
            
            stop = float(pos.get("stop_loss", 0.0))
            target = float(pos.get("take_profit", 0.0))
            
            entry_time_str = pos.get("entry_time")
            if entry_time_str:
                try:
                    entry_time = datetime.fromisoformat(entry_time_str.replace("Z", "+00:00"))
                    age_hours = (now - entry_time).total_seconds() / 3600.0
                except (ValueError, TypeError):
                    age_hours = 0.0
            else:
                age_hours = 0.0

            lines.append(
                f"{symbol:12} | {side:5} | {entry:10.6f} | {current:10.6f} | "
                f"{pnl_pct:+6.2f}% | {stop:10.6f} | {target:10.6f} | {age_hours:5.1f}"
            )

        return "\n".join(lines)

    def get_day_start_equity(self) -> float:
        """Get the equity value at the start of the current day."""
        return float(self.portfolio.get("day_start_equity", PAPER_STARTING_USDT))
