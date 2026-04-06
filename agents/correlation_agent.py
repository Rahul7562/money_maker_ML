# agents/correlation_agent.py
# Responsibility: Compute correlation matrix and filter highly correlated signals.

import logging
from typing import Any, Dict, List

import pandas as pd

from config import (
    CORRELATION_FILTER_ENABLED,
    CORRELATION_LOOKBACK,
    CORRELATION_THRESHOLD,
)

logger = logging.getLogger("CorrelationAgent")


class CorrelationAgent:
    """
    Computes correlation matrix and filters highly correlated trading signals.
    
    - Computes pairwise correlation using close price returns
    - Filters signals where correlation > threshold, keeping higher-scored signal
    - Prevents concentrated positions in correlated assets
    """

    def __init__(self) -> None:
        """Initialize CorrelationAgent."""
        self.enabled = CORRELATION_FILTER_ENABLED
        self.threshold = CORRELATION_THRESHOLD
        self.lookback = CORRELATION_LOOKBACK
        
        # Cache
        self._correlation_matrix: pd.DataFrame | None = None
        
        logger.info(
            "CorrelationAgent ready | enabled=%s | threshold=%.2f | lookback=%d",
            self.enabled, self.threshold, self.lookback
        )

    def compute_correlation_matrix(
        self, 
        market_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Compute correlation matrix from market data.
        
        Args:
            market_data: Dict mapping symbols to DataFrames with 'close' column.
            
        Returns:
            DataFrame with pairwise correlation coefficients.
        """
        if not market_data:
            return pd.DataFrame()
        
        # Extract returns for last N candles
        returns_dict: Dict[str, pd.Series] = {}
        
        for symbol, df in market_data.items():
            if len(df) < self.lookback + 1:
                continue
            
            # Get last lookback candles and compute returns
            close_prices = df["close"].tail(self.lookback + 1)
            returns = close_prices.pct_change().dropna()
            
            if len(returns) >= self.lookback - 5:  # Allow small tolerance
                returns_dict[symbol] = returns.reset_index(drop=True)
        
        if len(returns_dict) < 2:
            return pd.DataFrame()
        
        # Build returns DataFrame
        returns_df = pd.DataFrame(returns_dict)
        
        # Compute correlation matrix
        self._correlation_matrix = returns_df.corr()
        
        return self._correlation_matrix

    def filter_correlated(
        self, 
        signals: List[Any], 
        market_data: Dict[str, pd.DataFrame] | None = None,
        threshold: float | None = None
    ) -> List[Any]:
        """
        Filter out lower-scored signals that are highly correlated with higher-scored ones.
        
        For each pair with |correlation| > threshold, drops the lower-scored signal.
        
        Args:
            signals: List of Signal objects with 'symbol' and 'score' attributes.
            market_data: Optional market data for computing fresh correlation matrix.
            threshold: Optional threshold override.
            
        Returns:
            Filtered list of signals.
        """
        if not self.enabled:
            return signals
        
        if not signals:
            return signals
        
        threshold = threshold or self.threshold
        
        # Compute correlation if market_data provided
        if market_data:
            self.compute_correlation_matrix(market_data)
        
        if self._correlation_matrix is None or self._correlation_matrix.empty:
            logger.debug("No correlation matrix available, skipping filter")
            return signals
        
        # Sort signals by score descending
        sorted_signals = sorted(signals, key=lambda s: s.score, reverse=True)
        
        # Track which symbols to keep
        kept_symbols: set = set()
        filtered_signals: List[Any] = []
        dropped_count = 0
        
        for signal in sorted_signals:
            symbol = signal.symbol
            
            # Skip if symbol not in correlation matrix
            if symbol not in self._correlation_matrix.columns:
                filtered_signals.append(signal)
                kept_symbols.add(symbol)
                continue
            
            # Check correlation with already-kept symbols
            should_drop = False
            correlated_with = None
            
            for kept_symbol in kept_symbols:
                if kept_symbol not in self._correlation_matrix.columns:
                    continue
                
                corr = abs(self._correlation_matrix.loc[symbol, kept_symbol])
                
                if corr > threshold:
                    should_drop = True
                    correlated_with = kept_symbol
                    break
            
            if should_drop:
                logger.debug(
                    "Dropping %s (score=%.3f) - correlated %.3f with %s",
                    symbol, signal.score, corr, correlated_with
                )
                dropped_count += 1
            else:
                filtered_signals.append(signal)
                kept_symbols.add(symbol)
        
        if dropped_count > 0:
            logger.info(
                "Correlation filter: kept %d, dropped %d signals (threshold=%.2f)",
                len(filtered_signals), dropped_count, threshold
            )
        
        return filtered_signals

    def get_correlation(self, symbol1: str, symbol2: str) -> float | None:
        """
        Get correlation between two symbols.
        
        Args:
            symbol1: First symbol
            symbol2: Second symbol
            
        Returns:
            Correlation coefficient or None if not available.
        """
        if self._correlation_matrix is None or self._correlation_matrix.empty:
            return None
        
        if symbol1 not in self._correlation_matrix.columns:
            return None
        if symbol2 not in self._correlation_matrix.columns:
            return None
        
        return float(self._correlation_matrix.loc[symbol1, symbol2])

    def get_most_correlated_pairs(
        self, 
        top_n: int = 10
    ) -> List[tuple]:
        """
        Get the most correlated pairs of symbols.
        
        Args:
            top_n: Number of pairs to return.
            
        Returns:
            List of tuples (symbol1, symbol2, correlation).
        """
        if self._correlation_matrix is None or self._correlation_matrix.empty:
            return []
        
        pairs = []
        columns = self._correlation_matrix.columns.tolist()
        
        for i, sym1 in enumerate(columns):
            for sym2 in columns[i + 1:]:
                corr = abs(self._correlation_matrix.loc[sym1, sym2])
                if corr < 1.0:  # Exclude self-correlation
                    pairs.append((sym1, sym2, corr))
        
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs[:top_n]

    def check_position_correlation(
        self,
        new_symbol: str,
        existing_positions: List[str],
        threshold: float | None = None
    ) -> tuple:
        """
        Check if a new position would be highly correlated with existing positions.
        
        Args:
            new_symbol: Symbol to check
            existing_positions: List of symbols already in portfolio
            threshold: Optional threshold override
            
        Returns:
            Tuple of (is_correlated, most_correlated_with, correlation_value)
        """
        if not self.enabled:
            return (False, None, 0.0)
        
        threshold = threshold or self.threshold
        
        if self._correlation_matrix is None or self._correlation_matrix.empty:
            return (False, None, 0.0)
        
        if new_symbol not in self._correlation_matrix.columns:
            return (False, None, 0.0)
        
        max_corr = 0.0
        max_corr_symbol = None
        
        for existing in existing_positions:
            if existing not in self._correlation_matrix.columns:
                continue
            
            corr = abs(self._correlation_matrix.loc[new_symbol, existing])
            if corr > max_corr:
                max_corr = corr
                max_corr_symbol = existing
        
        is_correlated = max_corr > threshold
        return (is_correlated, max_corr_symbol, max_corr)

    def log_correlation_summary(self) -> None:
        """Log summary of correlation analysis."""
        if self._correlation_matrix is None or self._correlation_matrix.empty:
            logger.info("Correlation matrix not computed yet")
            return
        
        top_pairs = self.get_most_correlated_pairs(5)
        if top_pairs:
            pairs_str = ", ".join([f"{p[0]}-{p[1]}:{p[2]:.2f}" for p in top_pairs])
            logger.info("Top correlated pairs: %s", pairs_str)
