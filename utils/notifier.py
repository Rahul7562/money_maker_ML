# utils/notifier.py
# TASK 6: Telegram notifications for trading bot events.

import logging
from typing import Optional

import requests

from config import (
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
    TELEGRAM_ENABLED,
)

logger = logging.getLogger("TelegramNotifier")


class TelegramNotifier:
    """
    Sends notifications to Telegram.
    
    Events to notify:
    - Trade executed
    - Daily P&L summary
    - Drawdown guard triggered
    - Bot restart/shutdown
    """

    def __init__(self) -> None:
        """Initialize TelegramNotifier."""
        self.enabled = TELEGRAM_ENABLED
        self.bot_token = TELEGRAM_BOT_TOKEN
        self.chat_id = TELEGRAM_CHAT_ID
        self.api_url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        
        if not self.enabled:
            logger.info("TelegramNotifier disabled")
        elif not self.bot_token or not self.chat_id:
            logger.warning("TelegramNotifier enabled but missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID")
            self.enabled = False
        else:
            logger.info("TelegramNotifier ready | chat_id=%s", self.chat_id)

    def send(self, message: str, parse_mode: str = "HTML") -> bool:
        """
        Send a message to Telegram.
        
        Args:
            message: Message text (supports HTML formatting)
            parse_mode: "HTML" or "Markdown"
            
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.enabled:
            return False
        
        try:
            payload = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": parse_mode,
                "disable_web_page_preview": True,
            }
            
            response = requests.post(self.api_url, json=payload, timeout=10)
            
            if response.status_code == 200:
                return True
            else:
                logger.warning(
                    "Telegram send failed: %d %s",
                    response.status_code,
                    response.text[:100],
                )
                return False
                
        except requests.RequestException as e:
            logger.warning("Telegram send error: %s", e)
            return False

    def send_trade_executed(
        self,
        symbol: str,
        action: str,
        intent: str,
        price: float,
        amount: float,
        score: float,
    ) -> bool:
        """Send notification for trade execution."""
        emoji = "🟢" if action == "BUY" else "🔴"
        message = (
            f"{emoji} <b>{intent}</b>: {symbol}\n"
            f"Price: ${price:.6f}\n"
            f"Amount: ${amount:.2f}\n"
            f"Score: {score:.3f}"
        )
        return self.send(message)

    def send_daily_pnl(
        self,
        equity: float,
        day_pnl_pct: float,
        realized_pnl: float,
        positions_count: int,
    ) -> bool:
        """Send daily P&L summary."""
        emoji = "📈" if day_pnl_pct >= 0 else "📉"
        message = (
            f"{emoji} <b>Daily Summary</b>\n"
            f"Equity: ${equity:.2f}\n"
            f"Day P&L: {day_pnl_pct:+.2f}%\n"
            f"Realized: ${realized_pnl:+.2f}\n"
            f"Positions: {positions_count}"
        )
        return self.send(message)

    def send_drawdown_alert(self, current_drawdown: float, threshold: float) -> bool:
        """Send drawdown guard alert."""
        message = (
            f"⚠️ <b>Drawdown Guard Active</b>\n"
            f"Current: {current_drawdown:.2f}%\n"
            f"Threshold: {threshold:.2f}%\n"
            f"New positions blocked."
        )
        return self.send(message)

    def send_bot_status(self, status: str, extra: str = "") -> bool:
        """Send bot status notification (start/stop/restart)."""
        emoji_map = {
            "start": "🤖",
            "stop": "🛑",
            "restart": "🔄",
            "error": "❌",
        }
        emoji = emoji_map.get(status.lower(), "ℹ️")
        message = f"{emoji} <b>Bot {status.upper()}</b>"
        if extra:
            message += f"\n{extra}"
        return self.send(message)

    def send_exit_triggered(
        self,
        symbol: str,
        reason: str,
        pnl_pct: float,
    ) -> bool:
        """Send notification for auto exit."""
        emoji = "✅" if pnl_pct >= 0 else "⛔"
        reason_map = {
            "stop_loss": "Stop Loss",
            "take_profit": "Take Profit",
            "trailing_stop": "Trailing Stop",
        }
        reason_display = reason_map.get(reason, reason)
        message = (
            f"{emoji} <b>Exit: {symbol}</b>\n"
            f"Reason: {reason_display}\n"
            f"P&L: {pnl_pct:+.2f}%"
        )
        return self.send(message)
