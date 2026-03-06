#!/usr/bin/env python3
"""
Machine Learning Model for Congestion Window Prediction
Trains a model to predict cwnd updates using custom objective function.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path
import pickle
import logging

logger = logging.getLogger(__name__)


class CongestionWindowPredictor:
    """
    ML model to predict congestion window updates.
    Uses custom objective function: η(t-1) = goodput(t) - α·RTT(t) - β·loss(t)
    """

    def __init__(self, alpha: float = 0.1, beta: float = 1.0,
                 model_type: str = 'gradient_boosting'):
        """
        Initialize the predictor.

        Args:
            alpha: Weight for RTT penalty in objective function
            beta: Weight for loss penalty in objective function
            model_type: Type of model ('gradient_boosting', 'random_forest', 'ridge')
        """
        self.alpha = alpha
        self.beta = beta
        self.model_type = model_type

        # Initialize model
        if model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=10,
                min_samples_leaf=4,
                subsample=0.8,
                random_state=42
            )
        elif model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=4,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'ridge':
            self.model = Ridge(alpha=1.0)
        elif model_type == 'linear':
            self.model = LinearRegression()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.scaler = StandardScaler()
        self.feature_columns = None
        self.is_fitted = False

    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """
        Prepare features and labels from dataframe.

        Args:
            df: DataFrame with TCP statistics

        Returns:
            Tuple of (X, y, sample_weights)
        """
        # Define feature columns
        feature_cols = [
            'goodput_mbps',
            'rtt_ms',
            'loss',
            'snd_cwnd',
            'goodput_lag1',
            'rtt_lag1',
            'cwnd_lag1',
            'goodput_ma3',
            'rtt_ma3',
            'snd_ssthresh',
            'unacked',
            'retransmits'
        ]

        # Make sure all columns exist
        for col in feature_cols:
            if col not in df.columns:
                if col == 'rtt_ms':
                    df['rtt_ms'] = df['rtt_us'] / 1000.0
                elif col == 'loss':
                    df['loss'] = df['retransmits'] + df['lost'] + df['retrans']
                else:
                    df[col] = 0

        self.feature_columns = feature_cols
        X = df[feature_cols].values

        # Label is the actual delta_cwnd
        y = df['delta_cwnd'].values

        # Compute sample weights based on the objective function
        # We want to upweight samples where the objective was high
        # η(t-1) = goodput(t) - α·RTT(t) - β·loss(t)
        goodput = df['goodput_mbps'].values
        rtt = df['rtt_ms'].values
        loss = df['loss'].values

        objective = goodput - self.alpha * rtt - self.beta * loss

        # Normalize objective to positive weights
        # Higher objective = better outcome = higher weight
        weights = objective - objective.min() + 1.0  # Shift to positive
        weights = weights / weights.mean()  # Normalize

        return X, y, weights

    def train(self, train_df: pd.DataFrame) -> dict:
        """
        Train the model on training data.

        Args:
            train_df: Training DataFrame

        Returns:
            Training metrics dictionary
        """
        logger.info(f"Training {self.model_type} model...")

        # Prepare features
        X_train, y_train, weights = self.prepare_features(train_df)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Train model with sample weights
        self.model.fit(X_train_scaled, y_train, sample_weight=weights)
        self.is_fitted = True

        # Evaluate on training data
        y_pred_train = self.model.predict(X_train_scaled)

        metrics = {
            'mse': mean_squared_error(y_train, y_pred_train),
            'mae': mean_absolute_error(y_train, y_pred_train),
            'r2': r2_score(y_train, y_pred_train),
            'samples': len(y_train)
        }

        logger.info(f"Training completed. MSE: {metrics['mse']:.4f}, "
                    f"MAE: {metrics['mae']:.4f}, R2: {metrics['r2']:.4f}")

        return metrics

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        """
        Predict delta_cwnd for test data.

        Args:
            test_df: Test DataFrame

        Returns:
            Array of predicted delta_cwnd values
        """
        if not self.is_fitted:
            raise ValueError("Model is not trained yet. Call train() first.")

        X_test, _, _ = self.prepare_features(test_df)
        X_test_scaled = self.scaler.transform(X_test)

        return self.model.predict(X_test_scaled)

    def predict_cwnd_sequence(self, test_df: pd.DataFrame) -> np.ndarray:
        """
        Predict cwnd values by applying predicted deltas sequentially.

        Args:
            test_df: Test DataFrame

        Returns:
            Array of predicted cwnd values
        """
        if not self.is_fitted:
            raise ValueError("Model is not trained yet. Call train() first.")

        # Start with initial cwnd value
        initial_cwnd = test_df.iloc[0]['snd_cwnd']
        predicted_cwnd = [initial_cwnd]

        # Predict deltas one by one and accumulate
        for i in range(len(test_df)):
            sample = test_df.iloc[i:i+1].copy()

            # Use predicted cwnd from previous step
            if i > 0:
                sample['snd_cwnd'] = predicted_cwnd[-1]
                sample['cwnd_lag1'] = predicted_cwnd[-1] if i == 1 else predicted_cwnd[-2]

            X, _, _ = self.prepare_features(sample)
            X_scaled = self.scaler.transform(X)

            delta_pred = self.model.predict(X_scaled)[0]

            # Apply delta to get next cwnd
            next_cwnd = max(1, predicted_cwnd[-1] + delta_pred)  # Cwnd must be >= 1
            predicted_cwnd.append(next_cwnd)

        return np.array(predicted_cwnd[1:])  # Remove initial value

    def evaluate(self, test_df: pd.DataFrame) -> dict:
        """
        Evaluate model on test data.

        Args:
            test_df: Test DataFrame

        Returns:
            Evaluation metrics dictionary
        """
        # Predict deltas
        y_pred = self.predict(test_df)
        y_true = test_df['delta_cwnd'].values

        # Also predict full cwnd sequence
        cwnd_pred = self.predict_cwnd_sequence(test_df)
        cwnd_true = test_df['snd_cwnd'].values

        metrics = {
            'delta_cwnd': {
                'mse': mean_squared_error(y_true, y_pred),
                'mae': mean_absolute_error(y_true, y_pred),
                'r2': r2_score(y_true, y_pred),
            },
            'cwnd': {
                'mse': mean_squared_error(cwnd_true, cwnd_pred),
                'mae': mean_absolute_error(cwnd_true, cwnd_pred),
                'r2': r2_score(cwnd_true, cwnd_pred),
            },
            'samples': len(y_true)
        }

        # Compute objective function performance
        goodput = test_df['goodput_mbps'].values
        rtt = test_df['rtt_ms'].values
        loss = test_df['loss'].values

        objective_true = goodput - self.alpha * rtt - self.beta * loss
        metrics['objective'] = {
            'mean': objective_true.mean(),
            'std': objective_true.std(),
            'min': objective_true.min(),
            'max': objective_true.max()
        }

        logger.info(f"Test evaluation - Delta MSE: {metrics['delta_cwnd']['mse']:.4f}, "
                    f"Cwnd MSE: {metrics['cwnd']['mse']:.4f}")

        return metrics

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance (for tree-based models).

        Returns:
            DataFrame with feature names and importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model is not trained yet.")

        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_)
        else:
            logger.warning(f"Model type {self.model_type} does not support feature importance")
            return pd.DataFrame()

        df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
        return df

    def save(self, path: str):
        """Save model to disk."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'alpha': self.alpha,
            'beta': self.beta,
            'model_type': self.model_type,
            'is_fitted': self.is_fitted
        }

        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load model from disk."""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.alpha = model_data['alpha']
        self.beta = model_data['beta']
        self.model_type = model_data['model_type']
        self.is_fitted = model_data['is_fitted']

        logger.info(f"Model loaded from {path}")

    def extract_algorithm(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> str:
        """
        Extract a hand-written algorithm based on model behavior and principles.

        Args:
            train_df: Training data
            test_df: Test data

        Returns:
            String describing the algorithm
        """
        # Analyze feature importance
        feature_importance = self.get_feature_importance()

        # Analyze correlations
        df_combined = pd.concat([train_df, test_df])
        correlations = df_combined[['snd_cwnd', 'goodput_mbps', 'rtt_ms', 'loss']].corr()

        # Compute statistics
        avg_cwnd_when_good = df_combined[df_combined['goodput_mbps'] > df_combined['goodput_mbps'].quantile(0.75)]['snd_cwnd'].mean()
        avg_cwnd_when_bad = df_combined[df_combined['goodput_mbps'] < df_combined['goodput_mbps'].quantile(0.25)]['snd_cwnd'].mean()

        avg_rtt_when_good = df_combined[df_combined['goodput_mbps'] > df_combined['goodput_mbps'].quantile(0.75)]['rtt_ms'].mean()
        avg_rtt_when_bad = df_combined[df_combined['goodput_mbps'] < df_combined['goodput_mbps'].quantile(0.25)]['rtt_ms'].mean()

        algorithm = f"""
CONGESTION WINDOW UPDATE ALGORITHM
(Extracted from ML Model Observations)

Parameters:
- α (RTT penalty weight): {self.alpha}
- β (Loss penalty weight): {self.beta}

Key Observations from Data:
1. Feature Importance (Top 3):
{feature_importance.head(3).to_string(index=False) if len(feature_importance) > 0 else "N/A"}

2. Performance Characteristics:
   - Average cwnd when goodput is high (>75th percentile): {avg_cwnd_when_good:.1f} packets
   - Average cwnd when goodput is low (<25th percentile): {avg_cwnd_when_bad:.1f} packets
   - Average RTT when goodput is high: {avg_rtt_when_good:.2f} ms
   - Average RTT when goodput is low: {avg_rtt_when_bad:.2f} ms

3. Correlation with Goodput:
   - Cwnd correlation: {correlations.loc['snd_cwnd', 'goodput_mbps']:.3f}
   - RTT correlation: {correlations.loc['rtt_ms', 'goodput_mbps']:.3f}
   - Loss correlation: {correlations.loc['loss', 'goodput_mbps']:.3f}

PROPOSED HAND-WRITTEN ALGORITHM:

Algorithm: Adaptive Congestion Window Control
-------------------------------------------

State:
  cwnd: current congestion window (packets)
  ssthresh: slow start threshold
  rtt: smoothed round-trip time (ms)
  rtt_min: minimum observed RTT (baseline)
  loss_detected: boolean indicating packet loss
  goodput: measured application throughput (Mbps)

Initialization:
  cwnd = 10  # initial window
  ssthresh = 65535  # large value
  rtt_min = ∞

On each RTT:
  1. Update RTT measurements:
     rtt_min = min(rtt_min, rtt)
     queue_delay = rtt - rtt_min

  2. Compute objective function:
     η = goodput - α * rtt - β * loss_count

  3. Detect congestion signals:
     loss_congestion = (loss_detected == true)
     delay_congestion = (queue_delay > 2 * rtt_min)  # significant queueing

  4. Update cwnd based on state:

     If loss_congestion:
       # Multiplicative decrease on loss
       ssthresh = max(cwnd / 2, 2)
       cwnd = ssthresh

     Else if delay_congestion AND cwnd > ssthresh:
       # Gentle decrease on delay increase (Vegas-like)
       cwnd = cwnd - 1

     Else if cwnd < ssthresh:
       # Slow start: exponential increase
       cwnd = cwnd + 1  (per ACK: cwnd += 1/cwnd)

     Else:
       # Congestion avoidance: linear increase
       # But modulate based on objective function
       if η > η_prev AND goodput_increasing:
         cwnd = cwnd + 1/cwnd  # standard AIMD increase
       else if η < η_prev * 0.9:  # objective degrading
         cwnd = cwnd - 0.5/cwnd  # slight decrease
       else:
         cwnd = cwnd + 0.5/cwnd  # cautious increase

  5. Bounds:
     cwnd = max(2, min(cwnd, 65535))

Key Principles:
- Use AIMD (Additive Increase Multiplicative Decrease) as baseline
- React to both loss (explicit congestion) and delay (implicit congestion)
- Modulate increase rate based on objective function η
- Balance between goodput maximization and latency/loss minimization
- Maintain minimum cwnd to avoid stalling

This algorithm combines:
1. Classic TCP congestion control (AIMD, slow start)
2. Delay-based congestion detection (TCP Vegas-like)
3. Objective-driven adaptation (ML-inspired)
"""

        return algorithm


if __name__ == "__main__":
    # Example usage
    predictor = CongestionWindowPredictor(alpha=0.1, beta=1.0)
    print("Congestion Window Predictor initialized")
