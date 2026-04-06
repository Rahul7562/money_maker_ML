# agents/performance_agent.py
# Responsibility: Track trade performance, win rates, cooldowns, and statistics.

import csv
import json
import logging
import math
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import (
    STATE_DIR,
    SYMBOL_COOLDOWN_HOURS,
)

logger = logging.getLogger("PerformanceAgent")


class PerformanceAgent:
    """
    Tracks and analyzes trading performance with persistent storage.
    
    - Logs every closed trade to state/performance_log.csv
    - Computes win rates, Sharpe ratio, profit factor, max drawdown
    - Manages symbol cooldowns based on consecutive losses
    - Persists cooldown state to state/cooldowns.json
    """

    def __init__(self) -> None:
        """Initialize PerformanceAgent with state persistence."""
        self.state_dir = Path(STATE_DIR)
        self.performance_file = self.state_dir / "performance_log.csv"
        self.cooldowns_file = self.state_dir / "cooldowns.json"
        
        # Ensure state directory exists
        self._ensure_state_dir()
        
        # Load cooldowns
        self.cooldowns: Dict[str, str] = self._load_cooldowns()
        
        # Cache for performance data
        self._trades_cache: Optional[List[Dict[str, Any]]] = None
        self._cache_timestamp: Optional[datetime] = None
        self._last_stats_log: Optional[datetime] = None
        
        # Initialize CSV file with headers if needed
        self._ensure_csv_headers()
        
        logger.info(
            "PerformanceAgent ready | log=%s | cooldowns=%d",
            self.performance_file,
            len(self.cooldowns),
        )

    def _ensure_state_dir(self) -> None:
        """Create state directory if it doesn't exist."""
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def _ensure_csv_headers(self) -> None:
        """Ensure CSV file has headers."""
        if not self.performance_file.exists():
            headers = [
                "timestamp", "symbol", "side", "entry_price", "exit_price",
                "pnl_pct", "pnl_usdt", "duration_hours", "regime",
                "ml_confidence", "signal_score"
            ]
            with open(self.performance_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(headers)

    def _load_cooldowns(self) -> Dict[str, str]:
        """Load cooldowns from JSON file."""
        if self.cooldowns_file.exists():
            try:
                with open(self.cooldowns_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Clean expired cooldowns
                now = datetime.now(timezone.utc)
                cleaned = {}
                for symbol, expiry_str in data.items():
                    try:
                        expiry = datetime.fromisoformat(expiry_str.replace("Z", "+00:00"))
                        if expiry > now:
                            cleaned[symbol] = expiry_str
                    except (ValueError, TypeError):
                        continue
                return cleaned
            except (json.JSONDecodeError, IOError) as e:
                logger.warning("Failed to load cooldowns: %s", e)
        return {}

    def _save_cooldowns(self) -> None:
        """Save cooldowns to JSON file."""
        try:
            with open(self.cooldowns_file, "w", encoding="utf-8") as f:
                json.dump(self.cooldowns, f, indent=2)
        except IOError as e:
            logger.error("Failed to save cooldowns: %s", e)

    def log_trade(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        exit_price: float,
        pnl_pct: float,
        pnl_usdt: float,
        duration_hours: float,
        regime: str = "",
        ml_confidence: float = 0.0,
        signal_score: float = 0.0,
    ) -> None:
        """
        Log a closed trade to the performance CSV.
        
        Args:
            symbol: Trading pair symbol
            side: "LONG" or "SHORT"
            entry_price: Entry price
            exit_price: Exit price
            pnl_pct: Percentage profit/loss
            pnl_usdt: Absolute profit/loss in USDT
            duration_hours: Trade duration in hours
            regime: Market regime at entry (BULL/BEAR/SIDEWAYS)
            ml_confidence: ML model confidence at entry
            signal_score: Technical signal score at entry
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        
        row = [
            timestamp, symbol, side, entry_price, exit_price,
            round(pnl_pct, 4), round(pnl_usdt, 4), round(duration_hours, 2),
            regime, round(ml_confidence, 4), round(signal_score, 4)
        ]
        
        try:
            with open(self.performance_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(row)
            
            # Invalidate cache
            self._trades_cache = None
            
            logger.info(
                "Trade logged | %s %s | PnL: %.2f%% (%.2f USDT) | Duration: %.1fh",
                symbol, side, pnl_pct, pnl_usdt, duration_hours
            )
            
            # Check for consecutive losses and apply cooldown
            self._check_and_apply_cooldown(symbol, pnl_pct > 0)
            
        except IOError as e:
            logger.error("Failed to log trade: %s", e)

    def _check_and_apply_cooldown(self, symbol: str, was_win: bool) -> None:
        """Check consecutive losses and apply cooldown if needed."""
        if was_win:
            # Remove from cooldown if present
            if symbol in self.cooldowns:
                del self.cooldowns[symbol]
                self._save_cooldowns()
            return
        
        # Check consecutive losses
        consecutive = self.get_symbol_consecutive_losses(symbol)
        if consecutive >= 3:
            expiry = datetime.now(timezone.utc) + timedelta(hours=SYMBOL_COOLDOWN_HOURS)
            self.cooldowns[symbol] = expiry.isoformat()
            self._save_cooldowns()
            logger.warning(
                "Symbol %s placed in cooldown until %s (%d consecutive losses)",
                symbol, expiry.isoformat(), consecutive
            )

    def _load_trades(self) -> List[Dict[str, Any]]:
        """Load all trades from CSV, with caching."""
        now = datetime.now(timezone.utc)
        
        # Return cached if fresh (< 1 minute old)
        if (self._trades_cache is not None and 
            self._cache_timestamp is not None and
            (now - self._cache_timestamp).total_seconds() < 60):
            return self._trades_cache
        
        trades = []
        if self.performance_file.exists():
            try:
                with open(self.performance_file, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            trades.append({
                                "timestamp": row["timestamp"],
                                "symbol": row["symbol"],
                                "side": row["side"],
                                "entry_price": float(row["entry_price"]),
                                "exit_price": float(row["exit_price"]),
                                "pnl_pct": float(row["pnl_pct"]),
                                "pnl_usdt": float(row["pnl_usdt"]),
                                "duration_hours": float(row["duration_hours"]),
                                "regime": row.get("regime", ""),
                                "ml_confidence": float(row.get("ml_confidence", 0)),
                                "signal_score": float(row.get("signal_score", 0)),
                            })
                        except (ValueError, KeyError):
                            continue
            except IOError as e:
                logger.error("Failed to load trades: %s", e)
        
        self._trades_cache = trades
        self._cache_timestamp = now
        return trades

    def get_symbol_win_rate(self, symbol: str) -> float:
        """
        Get win rate for a specific symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Win rate as float (0.0 to 1.0), or 0.5 if no trades
        """
        trades = self._load_trades()
        symbol_trades = [t for t in trades if t["symbol"] == symbol]
        
        if not symbol_trades:
            return 0.5  # Neutral if no history
        
        wins = sum(1 for t in symbol_trades if t["pnl_pct"] > 0)
        return wins / len(symbol_trades)

    def get_symbol_consecutive_losses(self, symbol: str) -> int:
        """
        Get count of consecutive losses for a symbol (most recent).
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Number of consecutive losses (0 if last trade was a win)
        """
        trades = self._load_trades()
        symbol_trades = [t for t in trades if t["symbol"] == symbol]
        
        if not symbol_trades:
            return 0
        
        # Sort by timestamp descending
        symbol_trades.sort(key=lambda t: t["timestamp"], reverse=True)
        
        consecutive = 0
        for trade in symbol_trades:
            if trade["pnl_pct"] <= 0:
                consecutive += 1
            else:
                break
        
        return consecutive

    def is_symbol_in_cooldown(self, symbol: str) -> bool:
        """
        Check if a symbol is currently in cooldown.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            True if symbol is in cooldown, False otherwise
        """
        if symbol not in self.cooldowns:
            return False
        
        try:
            expiry = datetime.fromisoformat(self.cooldowns[symbol].replace("Z", "+00:00"))
            if datetime.now(timezone.utc) >= expiry:
                # Expired, remove from cooldowns
                del self.cooldowns[symbol]
                self._save_cooldowns()
                return False
            return True
        except (ValueError, TypeError):
            del self.cooldowns[symbol]
            self._save_cooldowns()
            return False

    def get_recent_stats(self, days: int = 7) -> Dict[str, Any]:
        """
        Get performance statistics for recent trades.
        
        Args:
            days: Number of days to look back
            
        Returns:
            Dict with win_rate, profit_factor, sharpe, max_drawdown, 
            best_symbol, worst_symbol
        """
        trades = self._load_trades()
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        
        recent_trades = []
        for t in trades:
            try:
                ts = datetime.fromisoformat(t["timestamp"].replace("Z", "+00:00"))
                if ts >= cutoff:
                    recent_trades.append(t)
            except (ValueError, TypeError):
                continue
        
        if not recent_trades:
            return {
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "sharpe": 0.0,
                "max_drawdown": 0.0,
                "best_symbol": None,
                "worst_symbol": None,
                "total_trades": 0,
                "total_pnl_usdt": 0.0,
            }
        
        # Calculate metrics
        wins = [t for t in recent_trades if t["pnl_pct"] > 0]
        losses = [t for t in recent_trades if t["pnl_pct"] <= 0]
        
        win_rate = len(wins) / len(recent_trades) if recent_trades else 0.0
        
        gross_profit = sum(t["pnl_usdt"] for t in wins) if wins else 0.0
        gross_loss = abs(sum(t["pnl_usdt"] for t in losses)) if losses else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0
        
        # Sharpe ratio (simplified, daily returns approximation)
        returns = [t["pnl_pct"] / 100.0 for t in recent_trades]
        sharpe = self.get_sharpe_ratio(returns)
        
        # Max drawdown from cumulative PnL
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        for t in sorted(recent_trades, key=lambda x: x["timestamp"]):
            cumulative += t["pnl_pct"]
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd
        
        # Best and worst symbols
        symbol_pnl: Dict[str, float] = {}
        for t in recent_trades:
            symbol_pnl[t["symbol"]] = symbol_pnl.get(t["symbol"], 0.0) + t["pnl_usdt"]
        
        best_symbol = max(symbol_pnl.items(), key=lambda x: x[1])[0] if symbol_pnl else None
        worst_symbol = min(symbol_pnl.items(), key=lambda x: x[1])[0] if symbol_pnl else None
        
        return {
            "win_rate": round(win_rate, 4),
            "profit_factor": round(profit_factor, 4) if profit_factor != float("inf") else "inf",
            "sharpe": round(sharpe, 4),
            "max_drawdown": round(max_dd, 4),
            "best_symbol": best_symbol,
            "worst_symbol": worst_symbol,
            "total_trades": len(recent_trades),
            "total_pnl_usdt": round(sum(t["pnl_usdt"] for t in recent_trades), 2),
        }

    def get_sharpe_ratio(self, returns_list: List[float]) -> float:
        """
        Calculate Sharpe ratio from a list of returns.
        
        Args:
            returns_list: List of returns (as decimals, not percentages)
            
        Returns:
            Annualized Sharpe ratio (sqrt(252) annualization)
        """
        if not returns_list or len(returns_list) < 2:
            return 0.0
        
        mean_return = sum(returns_list) / len(returns_list)
        variance = sum((r - mean_return) ** 2 for r in returns_list) / (len(returns_list) - 1)
        std_dev = math.sqrt(variance) if variance > 0 else 0.0
        
        if std_dev == 0:
            return 0.0
        
        # Annualize using sqrt(252) for daily returns
        sharpe = (mean_return / std_dev) * math.sqrt(252)
        return sharpe

    def get_avg_win_loss(self) -> tuple:
        """
        Get average win and average loss amounts.
        
        Returns:
            Tuple of (avg_win_pct, avg_loss_pct)
        """
        trades = self._load_trades()
        
        wins = [t["pnl_pct"] for t in trades if t["pnl_pct"] > 0]
        losses = [abs(t["pnl_pct"]) for t in trades if t["pnl_pct"] <= 0]
        
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        
        return avg_win, avg_loss

    def get_trade_count(self) -> int:
        """Get total number of trades logged."""
        return len(self._load_trades())

    def maybe_log_daily_stats(self) -> None:
        """Log full stats summary if 24 hours have passed since last log."""
        now = datetime.now(timezone.utc)
        
        if self._last_stats_log is not None:
            if (now - self._last_stats_log).total_seconds() < 86400:  # 24 hours
                return
        
        stats = self.get_recent_stats(days=7)
        if stats["total_trades"] > 0:
            logger.info(
                "=== 7-Day Performance Summary ===\n"
                "Trades: %d | Win Rate: %.1f%% | PF: %s | Sharpe: %.2f | Max DD: %.2f%%\n"
                "Total PnL: %.2f USDT | Best: %s | Worst: %s",
                stats["total_trades"],
                stats["win_rate"] * 100,
                stats["profit_factor"],
                stats["sharpe"],
                stats["max_drawdown"],
                stats["total_pnl_usdt"],
                stats["best_symbol"],
                stats["worst_symbol"],
            )
        
        self._last_stats_log = now

    def get_all_cooldowns(self) -> Dict[str, str]:
        """Get all active cooldowns."""
        # Clean expired cooldowns
        now = datetime.now(timezone.utc)
        cleaned = {}
        for symbol, expiry_str in list(self.cooldowns.items()):
            try:
                expiry = datetime.fromisoformat(expiry_str.replace("Z", "+00:00"))
                if expiry > now:
                    cleaned[symbol] = expiry_str
            except (ValueError, TypeError):
                continue
        
        self.cooldowns = cleaned
        self._save_cooldowns()
        return cleaned
