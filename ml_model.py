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

        # η(t-1) = goodput(t) - α·RTT(t) - β·loss(t)
        # The weight for sample at t-1 is the outcome observed at t (one step ahead).
        # Shift objective forward by 1 so weight[i] = η evaluated at i+1.
        objective = goodput - self.alpha * rtt - self.beta * loss
        shifted = np.empty_like(objective)
        shifted[:-1] = objective[1:]   # weight[t-1] ← outcome at t
        shifted[-1]  = objective[-1]   # last sample has no future; use own value

        # Normalize to positive weights
        weights = shifted - shifted.min() + 1.0
        weights = weights / weights.mean()

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
        Extract a hand-written algorithm derived from the actual fitted model.
        For linear models: reads real coefficients and builds the update rule.
        For tree models: uses feature importances + data quantiles to derive thresholds.

        Args:
            train_df: Training data
            test_df: Test data

        Returns:
            String describing the algorithm grounded in actual model observations.
        """
        df = pd.concat([train_df, test_df], ignore_index=True)

        # ── Data statistics ──────────────────────────────────────────────
        rtt_min    = df['rtt_ms'].quantile(0.05)   # near-min RTT ≈ propagation delay
        rtt_med    = df['rtt_ms'].median()
        rtt_p75    = df['rtt_ms'].quantile(0.75)
        cwnd_med   = df['snd_cwnd'].median()
        cwnd_p75   = df['snd_cwnd'].quantile(0.75)
        gp_med     = df['goodput_mbps'].median()
        loss_p75   = df['loss'].quantile(0.75)
        mss_bytes  = 1448  # typical MSS

        # BDP estimate: how many packets should fill the pipe
        bdp_packets = max(1.0, (gp_med * 1e6 / 8) * (rtt_med / 1000.0) / mss_bytes)

        # Correlation of each feature with goodput
        corr_cwnd = df['snd_cwnd'].corr(df['goodput_mbps'])
        corr_rtt  = df['rtt_ms'].corr(df['goodput_mbps'])
        corr_loss = df['loss'].corr(df['goodput_mbps'])

        # ── Model-derived update rule ─────────────────────────────────────
        if hasattr(self.model, 'coef_') and self.feature_columns:
            # Linear / Ridge: the learned equation IS the algorithm
            coefs    = self.model.coef_
            intercept = float(self.model.intercept_)

            # Build the exact learned equation string
            terms = [f"({coefs[i]:+.5f} × {col})"
                     for i, col in enumerate(self.feature_columns)]
            learned_eq = f"  delta_cwnd = {intercept:+.5f}\n"
            learned_eq += "\n".join(f"            {t}" for t in terms)

            # Rank features by absolute coefficient magnitude
            ranked = sorted(zip(self.feature_columns, coefs),
                            key=lambda x: abs(x[1]), reverse=True)
            top3 = ranked[:3]

            # Interpret dominant features in network terms
            interpretations = []
            for feat, w in top3:
                sign = "increases" if w > 0 else "decreases"
                if 'cwnd' in feat:
                    interpretations.append(
                        f"  • {feat} (coef={w:+.5f}): cwnd has strong momentum — "
                        f"the model is autoregressive. A {feat} ↑ {sign} delta_cwnd. "
                        f"This reflects the BDP principle: if the pipe was full last interval "
                        f"it likely remains full.")
                elif 'rtt' in feat:
                    interpretations.append(
                        f"  • {feat} (coef={w:+.5f}): rising RTT {sign} delta_cwnd. "
                        f"RTT increase signals queue build-up (queueing delay = RTT − RTT_min). "
                        f"The model learned to {'grow' if w > 0 else 'shrink'} cwnd when RTT rises.")
                elif 'goodput' in feat:
                    interpretations.append(
                        f"  • {feat} (coef={w:+.5f}): higher goodput {sign} delta_cwnd. "
                        f"Goodput ≈ cwnd × MSS / RTT (BDP relation), so goodput rising "
                        f"implies the network can absorb more and cwnd should "
                        f"{'grow' if w > 0 else 'shrink'}.")
                elif 'loss' in feat:
                    interpretations.append(
                        f"  • {feat} (coef={w:+.5f}): loss {sign} delta_cwnd. "
                        f"Loss signals buffer overflow; the model {'increases' if w > 0 else 'decreases'} "
                        f"cwnd in response — {'counter-intuitive, may reflect recovery bursts' if w > 0 else 'consistent with AIMD decrease'}.")
                else:
                    interpretations.append(
                        f"  • {feat} (coef={w:+.5f}): {sign} delta_cwnd.")

            interp_str = "\n".join(interpretations)

            # Simplified hand-written rule derived from top coefficients
            top_feat, top_w = top3[0]
            second_feat, second_w = top3[1] if len(top3) > 1 else (None, 0)
            third_feat, third_w  = top3[2] if len(top3) > 2 else (None, 0)

            # RTT-based queueing threshold from data
            queue_thresh = rtt_p75  # if RTT exceeds 75th pctile, assume queueing

            handwritten = f"""
Simplified Hand-Written Algorithm (derived from model observations)
-------------------------------------------------------------------
State:
  cwnd      — current congestion window (packets)
  cwnd_prev — cwnd from previous interval
  rtt       — current smoothed RTT (ms)
  rtt_min   — minimum observed RTT = {rtt_min:.2f} ms  (propagation baseline)
  goodput   — measured throughput (Mbps)
  loss      — cumulative retransmit/loss counter

Thresholds (from data percentiles this run):
  QUEUE_THRESH = {queue_thresh:.2f} ms   (RTT 75th pctile — queueing likely above this)
  BDP_TARGET   = {bdp_packets:.1f} packets (goodput={gp_med:.2f} Mbps × RTT={rtt_med:.2f} ms / MSS)
  LOSS_THRESH  = {loss_p75:.1f}          (loss 75th pctile)

Per-interval update rule:

  queue_delay = rtt - rtt_min          # isolate queueing component

  # Step 1 — loss event: hard multiplicative decrease (AIMD)
  if loss > LOSS_THRESH:
    cwnd = max(2, cwnd × 0.5)

  # Step 2 — queue building: gentle delay-based decrease (Vegas-like)
  elif queue_delay > (rtt_min × {(queue_thresh / max(rtt_min, 0.001)):.2f}):
    cwnd = max(2, cwnd - 1)

  # Step 3 — pipe under-utilized: probe upward
  elif cwnd < BDP_TARGET × 0.9:
    cwnd = cwnd + 1                    # additive increase toward BDP

  # Step 4 — near BDP capacity: modulate by model's dominant signal
  else:
    # Primary driver: {top_feat} (coef={top_w:+.5f})
    delta = {top_w:.5f} × {top_feat}"""

            if second_feat:
                handwritten += f"\n          # Secondary: {second_feat} (coef={second_w:+.5f})"
                handwritten += f"\n          delta += {second_w:.5f} × {second_feat}"
            if third_feat:
                handwritten += f"\n          # Tertiary: {third_feat} (coef={third_w:+.5f})"
                handwritten += f"\n          delta += {third_w:.5f} × {third_feat}"

            handwritten += f"""
    cwnd = max(2, cwnd + delta)

  # Step 5 — bounds
  cwnd = max(2, min(cwnd, 65535))
"""

        else:
            # Tree-based: derive top3 from feature importances
            fi = self.get_feature_importance()
            if len(fi) >= 3:
                top3 = list(zip(fi['feature'].tolist(), fi['importance'].tolist()))[:3]
            elif len(fi) > 0:
                top3 = list(zip(fi['feature'].tolist(), fi['importance'].tolist()))
            else:
                top3 = [('unknown', 0.0)]

            top_feat, top_imp = top3[0]
            interp_lines = []
            for feat, imp in top3:
                interp_lines.append(f"  • {feat} (importance={imp:.5f}): drives cwnd decisions most.")
            interp_str = "\n".join(interp_lines)
            learned_eq = "  (tree-based model — no closed-form equation; see feature importances above)"
            handwritten = f"""
Simplified Hand-Written Algorithm (tree model, importance-derived)
-------------------------------------------------------------------
  queue_delay = rtt - rtt_min

  if loss > {loss_p75:.1f}:                  cwnd = max(2, cwnd × 0.5)
  elif queue_delay > rtt_min × {(rtt_p75 / max(rtt_min, 0.001)):.2f}:  cwnd = max(2, cwnd - 1)
  elif cwnd < {bdp_packets:.1f}:              cwnd = cwnd + 1
  else:                                   cwnd = cwnd + 0.5 / cwnd
  cwnd = max(2, min(cwnd, 65535))

  # Top signal: {top_feat} (importance={top_imp:.5f})
"""

        # ── Always-safe reference to top3[0] for final summary ───────────
        top_label   = top3[0][0]
        top_val     = top3[0][1]

        # ── Final report ─────────────────────────────────────────────────
        algorithm = f"""
CONGESTION WINDOW UPDATE ALGORITHM
(Dynamically extracted from fitted {self.model_type} model — this run)

Objective function used during training:
  η(t-1) = goodput(t) − {self.alpha}·RTT(t) − {self.beta}·loss(t)
  Weight for sample at t-1 is the observed outcome at t (one-step-ahead evaluation).
  Samples are upweighted when the NEXT interval has high goodput and low RTT/loss.

═══════════════════════════════════════════════════════════════════
SECTION 1 — LEARNED MODEL EQUATION
═══════════════════════════════════════════════════════════════════
{learned_eq}

═══════════════════════════════════════════════════════════════════
SECTION 2 — DATA OBSERVATIONS (this run)
═══════════════════════════════════════════════════════════════════
  RTT:     min={rtt_min:.2f} ms  median={rtt_med:.2f} ms  p75={rtt_p75:.2f} ms
  cwnd:    median={cwnd_med:.1f} pkts  p75={cwnd_p75:.1f} pkts
  goodput: median={gp_med:.2f} Mbps
  BDP estimate: {bdp_packets:.1f} packets
    (= goodput × RTT / MSS — the ideal cwnd to saturate the pipe without queueing)

  Pearson correlation with goodput:
    cwnd ↔ goodput: {corr_cwnd:+.3f}   (positive → larger cwnd → more throughput, confirms BDP)
    RTT  ↔ goodput: {corr_rtt:+.3f}   (negative expected → RTT rise signals queue build-up)
    loss ↔ goodput: {corr_loss:+.3f}   (negative expected → loss → buffer overflow, throughput drop)

═══════════════════════════════════════════════════════════════════
SECTION 3 — DOMINANT FEATURES & NETWORK INTERPRETATION
═══════════════════════════════════════════════════════════════════
{interp_str}

═══════════════════════════════════════════════════════════════════
SECTION 4 — HAND-WRITTEN ALGORITHM
(suitable for kernel/eBPF implementation — all thresholds from this run's data)
═══════════════════════════════════════════════════════════════════
{handwritten}
Key principles grounded in this run's observations:
  1. BDP target = {bdp_packets:.1f} pkts (bandwidth={gp_med:.2f} Mbps × RTT={rtt_med:.2f} ms / MSS=1448B).
     Cwnd below BDP leaves bandwidth unused. Cwnd above BDP fills router queues → RTT rises.
  2. RTT_min = {rtt_min:.2f} ms is the pure propagation delay (5th pctile this run).
     queue_delay = RTT − RTT_min isolates the queueing component without loss signals.
  3. LOSS_THRESH = {loss_p75:.1f} (75th pctile of loss counter). Above this the buffer is
     overflowing → multiplicative halving (AIMD) is the correct response.
  4. Dominant learned signal: {top_label} = {top_val:+.5f}.
     This is what the model weighted most heavily when deciding how to update cwnd.
     Implement this term first in any eBPF/kernel hook.
"""
        return algorithm


if __name__ == "__main__":
    # Example usage
    predictor = CongestionWindowPredictor(alpha=0.1, beta=1.0)
    print("Congestion Window Predictor initialized")
