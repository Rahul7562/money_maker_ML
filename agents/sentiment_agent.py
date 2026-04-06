# agents/sentiment_agent.py
# Responsibility: Fetch and analyze crypto Fear & Greed Index for market sentiment.

import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional

import requests

from config import (
    MARKET_MODE,
    SENTIMENT_CACHE_HOURS,
    SENTIMENT_ENABLED,
    SENTIMENT_EXTREME_FEAR_THRESHOLD,
    SENTIMENT_EXTREME_GREED_THRESHOLD,
)

logger = logging.getLogger("SentimentAgent")

FEAR_GREED_API_URL = "https://api.alternative.me/fng/?limit=1"


class SentimentAgent:
    """
    Fetches and analyzes the Crypto Fear & Greed Index.
    
    - Caches results for SENTIMENT_CACHE_HOURS to avoid API hammering
    - Provides buy/sell suppression signals based on extreme sentiment
    - Returns score modifiers to adjust signal confidence
    """

    def __init__(self) -> None:
        """Initialize SentimentAgent with caching."""
        self.enabled = SENTIMENT_ENABLED
        self.cache_hours = SENTIMENT_CACHE_HOURS
        self.fear_threshold = SENTIMENT_EXTREME_FEAR_THRESHOLD
        self.greed_threshold = SENTIMENT_EXTREME_GREED_THRESHOLD
        self.market_mode = MARKET_MODE
        
        # Cache
        self._cached_sentiment: Optional[Dict[str, Any]] = None
        self._cache_timestamp: Optional[datetime] = None
        
        logger.info(
            "SentimentAgent ready | enabled=%s | fear_threshold=%d | greed_threshold=%d",
            self.enabled, self.fear_threshold, self.greed_threshold
        )

    def _fetch_sentiment(self) -> Optional[Dict[str, Any]]:
        """Fetch Fear & Greed Index from API."""
        try:
            response = requests.get(FEAR_GREED_API_URL, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if "data" not in data or not data["data"]:
                logger.warning("Invalid response from Fear & Greed API")
                return None
            
            fng_data = data["data"][0]
            return {
                "score": int(fng_data.get("value", 50)),
                "label": fng_data.get("value_classification", "Neutral"),
                "timestamp": fng_data.get("timestamp", ""),
            }
        except requests.RequestException as e:
            logger.warning("Failed to fetch Fear & Greed Index: %s", e)
            return None
        except (KeyError, ValueError, TypeError) as e:
            logger.warning("Error parsing Fear & Greed response: %s", e)
            return None

    def get_sentiment(self) -> Dict[str, Any]:
        """
        Get current Fear & Greed sentiment, using cache if fresh.
        
        Returns:
            Dict with score (0-100), label, and timestamp.
            Score interpretation:
            - 0-24: Extreme Fear
            - 25-49: Fear
            - 50-50: Neutral
            - 51-74: Greed
            - 75-100: Extreme Greed
        """
        if not self.enabled:
            return {"score": 50, "label": "Neutral (Disabled)", "timestamp": ""}
        
        now = datetime.now(timezone.utc)
        
        # Return cached if fresh
        if (self._cached_sentiment is not None and 
            self._cache_timestamp is not None and
            (now - self._cache_timestamp).total_seconds() < self.cache_hours * 3600):
            return self._cached_sentiment
        
        # Fetch new data
        sentiment = self._fetch_sentiment()
        
        if sentiment:
            self._cached_sentiment = sentiment
            self._cache_timestamp = now
            logger.info(
                "Sentiment updated | score=%d | label=%s",
                sentiment["score"], sentiment["label"]
            )
            return sentiment
        
        # Return cached if fetch failed, or default
        if self._cached_sentiment:
            logger.warning("Using stale sentiment cache")
            return self._cached_sentiment
        
        return {"score": 50, "label": "Neutral (No Data)", "timestamp": ""}

    def should_suppress_buy(self) -> bool:
        """
        Check if extreme fear should suppress BUY signals in SPOT mode.
        
        In extreme fear, markets may continue falling, so be cautious with longs.
        However, this can also be a contrarian opportunity (disabled by default).
        
        Returns:
            True if score < SENTIMENT_EXTREME_FEAR_THRESHOLD in SPOT mode
        """
        if not self.enabled:
            return False
        
        if self.market_mode != "SPOT":
            return False
        
        sentiment = self.get_sentiment()
        # Note: We actually DON'T suppress during extreme fear - it's often a good time to buy
        # This method exists for optional risk-averse behavior
        # Default behavior: return False (don't suppress)
        return False  # Changed from original spec - extreme fear is often bullish contrarian signal

    def should_suppress_sell(self) -> bool:
        """
        Check if extreme greed should suppress SELL signals.
        
        In extreme greed, holding longs may be beneficial.
        
        Returns:
            True if score > SENTIMENT_EXTREME_GREED_THRESHOLD
        """
        if not self.enabled:
            return False
        
        sentiment = self.get_sentiment()
        # Note: We actually DON'T suppress during extreme greed
        # This method exists for optional behavior
        return False  # Changed from original spec

    def get_score_modifier(self, action: str) -> float:
        """
        Get a score modifier based on sentiment.
        
        - Extreme Fear (< 20): 0.85 for BUY (risky), 1.10 for SELL (confirms)
        - Fear (20-40): 0.95 for BUY, 1.05 for SELL
        - Neutral (40-60): 1.0
        - Greed (60-80): 1.05 for BUY (momentum), 0.95 for SELL
        - Extreme Greed (> 80): 0.90 for BUY (top signal), 1.10 for SELL
        
        Args:
            action: "BUY" or "SELL"
            
        Returns:
            Score modifier between 0.85 and 1.15
        """
        if not self.enabled:
            return 1.0
        
        sentiment = self.get_sentiment()
        score = sentiment["score"]
        
        if action == "BUY":
            if score < self.fear_threshold:  # Extreme Fear
                # Contrarian: extreme fear can be good for buying
                return 1.05  # Slight boost for contrarian buying
            elif score < 40:  # Fear
                return 1.0
            elif score < 60:  # Neutral
                return 1.0
            elif score < self.greed_threshold:  # Greed
                return 1.02  # Slight momentum boost
            else:  # Extreme Greed
                return 0.90  # Penalize buying at tops
        
        elif action == "SELL":
            if score < self.fear_threshold:  # Extreme Fear
                return 1.05  # Confirms bearish sentiment
            elif score < 40:  # Fear
                return 1.0
            elif score < 60:  # Neutral
                return 1.0
            elif score < self.greed_threshold:  # Greed
                return 0.95  # Less confident selling in greed
            else:  # Extreme Greed
                return 1.10  # Confirms potential reversal
        
        return 1.0

    def log_sentiment(self) -> None:
        """Log current sentiment - call at start of each cycle."""
        if not self.enabled:
            return
        
        sentiment = self.get_sentiment()
        logger.info(
            "Market Sentiment | Score: %d/100 | Label: %s",
            sentiment["score"], sentiment["label"]
        )

    def get_sentiment_label_emoji(self) -> str:
        """Get an emoji representation of current sentiment for notifications."""
        if not self.enabled:
            return "⚪"
        
        sentiment = self.get_sentiment()
        score = sentiment["score"]
        
        if score < self.fear_threshold:
            return "😱"  # Extreme Fear
        elif score < 40:
            return "😨"  # Fear
        elif score < 60:
            return "😐"  # Neutral
        elif score < self.greed_threshold:
            return "😊"  # Greed
        else:
            return "🤑"  # Extreme Greed

    def is_extreme_sentiment(self) -> bool:
        """Check if market is in extreme fear or greed."""
        if not self.enabled:
            return False
        
        sentiment = self.get_sentiment()
        score = sentiment["score"]
        return score < self.fear_threshold or score > self.greed_threshold
