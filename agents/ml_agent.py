# agents/ml_agent.py
# Responsibility: PyTorch LSTM + Attention model for price prediction.

import json
import logging
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    class _TorchStub:
        class Tensor:
            pass

        @staticmethod
        def device(_name: str):
            return None

        @staticmethod
        def log_softmax(*_args, **_kwargs):
            raise RuntimeError("PyTorch is not installed")

    class _NNStub:
        class Module:
            pass

    torch = _TorchStub()  # type: ignore[assignment]
    nn = _NNStub()  # type: ignore[assignment]
    DataLoader = TensorDataset = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False

from config import (
    ML_EARLY_STOPPING_PATIENCE,
    ML_ENABLED,
    ML_MIN_CONFIDENCE,
    ML_SEQUENCE_LENGTH,
    STATE_DIR,
)

logger = logging.getLogger("MLAgent")

# Feature columns for the model
FEATURE_COLUMNS = [
    "open", "high", "low", "close", "volume",
    "rsi", "macd", "macd_signal", "ema_fast", "ema_slow",
    "atr", "bb_upper", "bb_lower", "bb_mid", "vwap",
    "volume_ratio", "returns_1", "returns_3", "returns_6"
]
NUM_FEATURES = len(FEATURE_COLUMNS)  # 19 features


class TradingLSTM(nn.Module):
    """
    Hybrid LSTM + Self-Attention model for trading signals.
    
    Architecture:
    - Input BatchNorm
    - LSTM(19, 128, 2 layers, dropout=0.3)
    - MultiheadAttention(embed_dim=128, num_heads=4)
    - Dropout(0.3)
    - FC(128 -> 64 -> 3)
    - Softmax output (BUY/HOLD/SELL probabilities)
    """

    def __init__(
        self,
        input_size: int = NUM_FEATURES,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_heads: int = 4,
        num_classes: int = 3,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Input normalization
        self.batch_norm = nn.BatchNorm1d(input_size)
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        
        # Self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)
        
        # Softmax for probabilities
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, features)
            
        Returns:
            Output tensor of shape (batch, num_classes) - probabilities
        """
        batch_size, seq_len, num_features = x.shape
        
        # BatchNorm expects (batch, features, seq) for 1d
        # We need to apply it per-feature across the sequence
        x = x.permute(0, 2, 1)  # (batch, features, seq)
        x = self.batch_norm(x)
        x = x.permute(0, 2, 1)  # (batch, seq, features)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)  # (batch, seq, hidden)
        
        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)  # (batch, seq, hidden)
        
        # Take the last time step
        out = attn_out[:, -1, :]  # (batch, hidden)
        
        # Dropout
        out = self.dropout(out)
        
        # FC layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        # Softmax
        probs = self.softmax(out)
        
        return probs


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy loss with label smoothing."""
    
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(pred, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class MLAgent:
    """
    PyTorch-based ML agent for price prediction.
    
    - Uses LSTM + Attention architecture
    - Trains on OHLCV + technical indicators
    - Labels: BUY(2), HOLD(1), SELL(0) based on forward returns
    - CPU-only inference for Linux VM compatibility
    """

    def __init__(self) -> None:
        """Initialize MLAgent."""
        self.enabled = ML_ENABLED and TORCH_AVAILABLE
        self.sequence_length = ML_SEQUENCE_LENGTH
        self.min_confidence = ML_MIN_CONFIDENCE
        self.patience = ML_EARLY_STOPPING_PATIENCE
        
        self.state_dir = Path(STATE_DIR)
        self.model_file = self.state_dir / "lstm_model.pt"
        self.model_meta_file = self.state_dir / "lstm_model_meta.json"
        
        # Ensure state dir exists
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        # Device - CPU only for VM compatibility
        self.device = torch.device("cpu") if TORCH_AVAILABLE else None
        
        # Model
        self.model: Optional[TradingLSTM] = None
        self._model_loaded = False
        self._last_train_time: Optional[datetime] = None
        
        # Try to load existing model
        if self.enabled:
            self._load_model_if_exists()
        
        status = "ready" if self.enabled else "disabled (torch not available)"
        logger.info("MLAgent %s | sequence_length=%d", status, self.sequence_length)

    def is_model_loaded(self) -> bool:
        """Check if a trained model is loaded and ready for predictions."""
        return self._model_loaded

    def _load_model_if_exists(self) -> bool:
        """Load model from disk if it exists and is fresh."""
        if not self.model_file.exists():
            return False
        
        try:
            # Check if model is fresh (< 24h old)
            if self.model_meta_file.exists():
                with open(self.model_meta_file, "r") as f:
                    meta = json.load(f)
                last_train_str = meta.get("last_train", "2000-01-01")
                last_train = datetime.fromisoformat(last_train_str.replace("Z", "+00:00"))
                # Ensure timezone-aware comparison
                if last_train.tzinfo is None:
                    last_train = last_train.replace(tzinfo=timezone.utc)
                if datetime.now(timezone.utc) - last_train < timedelta(hours=24):
                    self._last_train_time = last_train
            
            # Load model
            self.model = TradingLSTM()
            self.model.load_state_dict(torch.load(self.model_file, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            self._model_loaded = True
            logger.info("Loaded existing model from %s", self.model_file)
            return True
            
        except Exception as e:
            logger.warning("Failed to load model: %s", e)
            self.model = None
            return False

    def _save_model(self) -> None:
        """Save model to disk."""
        if self.model is None:
            return
        
        try:
            torch.save(self.model.state_dict(), self.model_file)
            
            # Save metadata
            meta = {
                "last_train": datetime.now(timezone.utc).isoformat(),
                "sequence_length": self.sequence_length,
                "num_features": NUM_FEATURES,
            }
            with open(self.model_meta_file, "w") as f:
                json.dump(meta, f, indent=2)
            
            logger.info("Model saved to %s", self.model_file)
            
        except Exception as e:
            logger.error("Failed to save model: %s", e)

    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all features from raw OHLCV data.
        
        Features: open, high, low, close, volume, RSI, MACD, MACD_signal,
        EMA_fast, EMA_slow, ATR, BB_upper, BB_lower, BB_mid, VWAP,
        volume_ratio, returns_1, returns_3, returns_6
        """
        data = df.copy()
        
        close = data["close"]
        high = data["high"]
        low = data["low"]
        volume = data["volume"]
        
        # RSI (14)
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1e-12)
        data["rsi"] = (100 - (100 / (1 + rs))).clip(0, 100).fillna(50)
        
        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        data["macd"] = ema12 - ema26
        data["macd_signal"] = data["macd"].ewm(span=9, adjust=False).mean()
        
        # EMAs
        data["ema_fast"] = close.ewm(span=20, adjust=False).mean()
        data["ema_slow"] = close.ewm(span=50, adjust=False).mean()
        
        # ATR
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        data["atr"] = tr.rolling(14).mean().bfill()
        
        # Bollinger Bands
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        data["bb_mid"] = bb_mid.bfill()
        data["bb_upper"] = (bb_mid + 2 * bb_std).bfill()
        data["bb_lower"] = (bb_mid - 2 * bb_std).bfill()
        
        # VWAP
        typical_price = (high + low + close) / 3
        data["vwap"] = (typical_price * volume).rolling(20).sum() / volume.rolling(20).sum()
        data["vwap"] = data["vwap"].bfill()
        
        # Volume ratio
        vol_ma = volume.rolling(20).mean().bfill()
        data["volume_ratio"] = (volume / vol_ma.replace(0, 1)).fillna(1)
        
        # Returns
        data["returns_1"] = close.pct_change(1).fillna(0)
        data["returns_3"] = close.pct_change(3).fillna(0)
        data["returns_6"] = close.pct_change(6).fillna(0)
        
        return data

    def _generate_labels(self, df: pd.DataFrame, threshold: float = 0.005) -> pd.Series:
        """
        Generate labels based on forward returns.
        
        - forward_return > 0.5%: BUY (2)
        - forward_return < -0.5%: SELL (0)
        - else: HOLD (1)
        """
        forward_return = df["close"].shift(-1) / df["close"] - 1
        
        labels = pd.Series(1, index=df.index)  # Default HOLD
        labels[forward_return > threshold] = 2  # BUY
        labels[forward_return < -threshold] = 0  # SELL
        
        return labels

    def _create_sequences(
        self, 
        data: pd.DataFrame, 
        labels: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for training."""
        features = data[FEATURE_COLUMNS].values
        labels_arr = labels.values
        
        X, y = [], []
        for i in range(self.sequence_length, len(features) - 1):
            X.append(features[i - self.sequence_length:i])
            y.append(labels_arr[i])
        
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

    def _normalize_features(self, X: np.ndarray) -> np.ndarray:
        """Normalize features per-sequence."""
        # Z-score normalization per feature per sequence
        X_norm = np.zeros_like(X)
        for i in range(X.shape[0]):
            seq = X[i]
            mean = seq.mean(axis=0, keepdims=True)
            std = seq.std(axis=0, keepdims=True)
            std[std == 0] = 1  # Avoid division by zero
            X_norm[i] = (seq - mean) / std
        return X_norm

    def train(self, market_data: Dict[str, pd.DataFrame], epochs: int = 50) -> Dict[str, Any]:
        """
        Train the model on market data.
        
        Args:
            market_data: Dict mapping symbols to OHLCV DataFrames
            epochs: Max number of training epochs
            
        Returns:
            Training summary dict
        """
        if not self.enabled:
            return {"status": "disabled"}
        
        logger.info("Starting ML model training...")
        
        # Combine data from all symbols
        all_X, all_y = [], []
        
        for symbol, df in market_data.items():
            if len(df) < self.sequence_length + 50:
                continue
            
            try:
                # Compute features
                data = self._compute_features(df)
                labels = self._generate_labels(data)
                
                # Drop NaN rows
                valid_idx = data[FEATURE_COLUMNS].notna().all(axis=1)
                data = data[valid_idx]
                labels = labels[valid_idx]
                
                if len(data) < self.sequence_length + 20:
                    continue
                
                # Create sequences
                X, y = self._create_sequences(data, labels)
                
                if len(X) > 0:
                    all_X.append(X)
                    all_y.append(y)
                    
            except Exception as e:
                logger.warning("Failed to process %s for training: %s", symbol, e)
                continue
        
        if not all_X:
            logger.warning("No valid training data")
            return {"status": "no_data"}
        
        # Combine all data
        X = np.concatenate(all_X, axis=0)
        y = np.concatenate(all_y, axis=0)
        
        # Normalize
        X = self._normalize_features(X)
        
        # Replace any remaining NaN/inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Train/val split (80/20, no shuffle for time series)
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        logger.info(
            "Training data: %d samples | Validation: %d samples | Classes: %s",
            len(X_train), len(X_val), dict(zip(*np.unique(y_train, return_counts=True)))
        )
        
        # Convert to tensors
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.long)
        
        # DataLoaders
        train_dataset = TensorDataset(X_train_t, y_train_t)
        val_dataset = TensorDataset(X_val_t, y_val_t)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        
        # Initialize model
        self.model = TradingLSTM()
        self.model.to(self.device)
        
        # Loss, optimizer, scheduler
        criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # Early stopping
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    outputs = self.model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs, 1)
                    total += y_batch.size(0)
                    correct += (predicted == y_batch).sum().item()
            
            val_loss /= len(val_loader)
            val_acc = correct / total if total > 0 else 0
            
            scheduler.step()
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                logger.info(
                    "Epoch %d/%d | Train Loss: %.4f | Val Loss: %.4f | Val Acc: %.2f%%",
                    epoch + 1, epochs, train_loss, val_loss, val_acc * 100
                )
            
            if patience_counter >= self.patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break
        
        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        self.model.eval()
        self._model_loaded = True
        self._last_train_time = datetime.now(timezone.utc)
        
        # Save model
        self._save_model()
        
        return {
            "status": "trained",
            "epochs": epoch + 1,
            "best_val_loss": best_val_loss,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
        }

    def predict(self, symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Make a prediction for a symbol.
        
        Args:
            symbol: Trading pair symbol
            df: OHLCV DataFrame
            
        Returns:
            Dict with action, confidence, and probabilities
        """
        if not self.enabled or not self._model_loaded or self.model is None:
            return {
                "action": "HOLD",
                "confidence": 0.0,
                "probabilities": {"SELL": 0.33, "HOLD": 0.34, "BUY": 0.33},
                "reason": "Model not available",
            }
        
        try:
            if len(df) < self.sequence_length + 30:
                return {
                    "action": "HOLD",
                    "confidence": 0.0,
                    "probabilities": {"SELL": 0.33, "HOLD": 0.34, "BUY": 0.33},
                    "reason": "Insufficient data",
                }
            
            # Compute features
            data = self._compute_features(df)
            
            # Get last sequence
            features = data[FEATURE_COLUMNS].tail(self.sequence_length).values
            
            # Check for NaN
            if np.isnan(features).any():
                features = np.nan_to_num(features, nan=0.0)
            
            # Normalize
            mean = features.mean(axis=0, keepdims=True)
            std = features.std(axis=0, keepdims=True)
            std[std == 0] = 1
            features_norm = (features - mean) / std
            
            # Convert to tensor
            X = torch.tensor(features_norm, dtype=torch.float32).unsqueeze(0)  # (1, seq, features)
            X = X.to(self.device)
            
            # Predict
            self.model.eval()
            with torch.no_grad():
                probs = self.model(X)
            
            probs_np = probs.cpu().numpy()[0]
            
            # Action mapping: 0=SELL, 1=HOLD, 2=BUY
            action_idx = np.argmax(probs_np)
            confidence = float(probs_np[action_idx])
            
            actions = ["SELL", "HOLD", "BUY"]
            action = actions[action_idx]
            
            # Apply minimum confidence threshold
            if confidence < self.min_confidence:
                action = "HOLD"
            
            return {
                "action": action,
                "confidence": confidence,
                "probabilities": {
                    "SELL": float(probs_np[0]),
                    "HOLD": float(probs_np[1]),
                    "BUY": float(probs_np[2]),
                },
                "reason": f"ML prediction (conf={confidence:.2f})",
            }
            
        except Exception as e:
            logger.warning("ML prediction failed for %s: %s", symbol, e)
            return {
                "action": "HOLD",
                "confidence": 0.0,
                "probabilities": {"SELL": 0.33, "HOLD": 0.34, "BUY": 0.33},
                "reason": f"Prediction error: {e}",
            }

    def should_retrain(self, cycle_count: int, tuning_cycle_multiple: int) -> bool:
        """
        Check if model should be retrained.
        
        Args:
            cycle_count: Current cycle number
            tuning_cycle_multiple: WFO reoptimize frequency
            
        Returns:
            True if model should be retrained
        """
        if not self.enabled:
            return False
        
        # Retrain if no model loaded
        if not self._model_loaded:
            return True
        
        # Retrain every N tuning cycles
        from config import ML_RETRAIN_EVERY_N_TUNING_CYCLES, WFO_REOPTIMIZE_EVERY_CYCLES
        
        retrain_every = WFO_REOPTIMIZE_EVERY_CYCLES * ML_RETRAIN_EVERY_N_TUNING_CYCLES
        return cycle_count > 0 and cycle_count % retrain_every == 0

    def is_model_fresh(self, max_age_hours: int = 24) -> bool:
        """Check if model was trained recently."""
        if self._last_train_time is None:
            return False
        
        age = datetime.now(timezone.utc) - self._last_train_time.replace(tzinfo=timezone.utc)
        return age.total_seconds() < max_age_hours * 3600

    def blend_score(
        self,
        technical_score: float,
        technical_action: str,
        ml_prediction: Dict[str, Any],
        ml_weight: float = 0.4,
    ) -> float:
        """
        Blend technical score with ML confidence.
        
        - If ML agrees with technical: boost score
        - If ML disagrees: penalize by 0.15
        
        Args:
            technical_score: Technical analysis score
            technical_action: Technical signal action (BUY/SELL/HOLD)
            ml_prediction: Dict from predict()
            ml_weight: Weight for ML component (default 0.4)
            
        Returns:
            Blended score
        """
        ml_action = ml_prediction.get("action", "HOLD")
        ml_confidence = ml_prediction.get("confidence", 0.0)
        
        # Base blend
        technical_weight = 1.0 - ml_weight
        
        # Check agreement
        if ml_action == technical_action:
            # Agreement - standard blend with slight boost
            blended = technical_weight * technical_score + ml_weight * ml_confidence
        elif ml_action == "HOLD" or technical_action == "HOLD":
            # One is HOLD - standard blend
            blended = technical_weight * technical_score + ml_weight * ml_confidence
        else:
            # Disagreement (BUY vs SELL) - penalize
            blended = technical_score - 0.15
        
        return max(0.0, min(1.0, blended))
