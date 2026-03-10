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
        self.alpha = alpha
        self.beta = beta
        self.model_type = model_type

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

        FIX: snd_cwnd removed from features — it is the variable whose delta
        we are predicting. Including it causes data leakage and produces
        artificially perfect R² scores. cwnd_lag1 (t-1 value) is kept because
        it is a legitimate historical input available at prediction time.
        """
        feature_cols = [
            'goodput_mbps',
            'rtt_ms',
            'loss',
            # snd_cwnd intentionally excluded — leakage (predicting its delta)
            'goodput_lag1',
            'rtt_lag1',
            'cwnd_lag1',    # t-1 value: fine, available at prediction time
            'goodput_ma3',
            'rtt_ma3',
            'snd_ssthresh',
            'unacked',
            'retransmits',
        ]

        # Ensure all columns exist, derive if missing
        for col in feature_cols:
            if col not in df.columns:
                if col == 'rtt_ms':
                    df['rtt_ms'] = df['rtt_us'] / 1000.0
                elif col == 'loss':
                    df['loss'] = (
                        df['retransmits'].fillna(0)
                        + df['lost'].fillna(0)
                        + df['retrans'].fillna(0)
                    ).diff().fillna(0).clip(lower=0)
                else:
                    df[col] = 0

        self.feature_columns = feature_cols
        X = df[feature_cols].values
        y = df['delta_cwnd'].values

        # Sample weights from objective function η(t-1) = goodput(t) - α·RTT(t) - β·loss(t)
        # weight[t-1] = outcome observed at t (one-step-ahead evaluation)
        goodput  = df['goodput_mbps'].values
        rtt      = df['rtt_ms'].values
        loss     = df['loss'].values

        objective      = goodput - self.alpha * rtt - self.beta * loss
        shifted        = np.empty_like(objective)
        shifted[:-1]   = objective[1:]   # weight[t-1] ← outcome at t
        shifted[-1]    = objective[-1]   # last sample: use own value

        weights = shifted - shifted.min() + 1.0
        weights = weights / weights.mean()

        return X, y, weights

    def train(self, train_df: pd.DataFrame) -> dict:
        logger.info(f"Training {self.model_type} model...")

        X_train, y_train, weights = self.prepare_features(train_df)
        X_train_scaled = self.scaler.fit_transform(X_train)

        self.model.fit(X_train_scaled, y_train, sample_weight=weights)
        self.is_fitted = True

        y_pred_train = self.model.predict(X_train_scaled)
        metrics = {
            'mse':     mean_squared_error(y_train, y_pred_train),
            'mae':     mean_absolute_error(y_train, y_pred_train),
            'r2':      r2_score(y_train, y_pred_train),
            'samples': len(y_train),
        }

        logger.info(f"Training completed. MSE: {metrics['mse']:.4f}, "
                    f"MAE: {metrics['mae']:.4f}, R2: {metrics['r2']:.4f}")
        return metrics

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model is not trained yet. Call train() first.")
        X_test, _, _ = self.prepare_features(test_df)
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_test_scaled)

    def predict_cwnd_sequence(self, test_df: pd.DataFrame) -> np.ndarray:
        """
        Predict cwnd values by applying predicted deltas sequentially.

        FIX: index bug when i=0 — predicted_cwnd has only one element so
        predicted_cwnd[-2] would silently wrap to index 0 (initial_cwnd),
        which happens to be correct by accident at i=1 but is wrong logic.
        Now explicitly handled for all three cases: i=0, i=1, i>=2.
        """
        if not self.is_fitted:
            raise ValueError("Model is not trained yet. Call train() first.")

        initial_cwnd  = test_df.iloc[0]['snd_cwnd']
        predicted_cwnd = [initial_cwnd]

        for i in range(len(test_df)):
            sample = test_df.iloc[i:i+1].copy()

            if i > 0:
                sample['snd_cwnd'] = predicted_cwnd[-1]

                # FIX: explicit three-way branch — no wraparound indexing
                if i == 1:
                    sample['cwnd_lag1'] = predicted_cwnd[-1]   # only one pred so far
                elif i >= 2:
                    sample['cwnd_lag1'] = predicted_cwnd[-2]   # previous predicted
                # i == 0 branch: use whatever is already in the dataframe row

            X, _, _   = self.prepare_features(sample)
            X_scaled  = self.scaler.transform(X)
            delta_pred = self.model.predict(X_scaled)[0]

            next_cwnd = max(1, predicted_cwnd[-1] + delta_pred)
            predicted_cwnd.append(next_cwnd)

        return np.array(predicted_cwnd[1:])  # drop seed value

    def evaluate(self, test_df: pd.DataFrame) -> dict:
        y_pred    = self.predict(test_df)
        y_true    = test_df['delta_cwnd'].values

        cwnd_pred = self.predict_cwnd_sequence(test_df)
        cwnd_true = test_df['snd_cwnd'].values

        metrics = {
            'delta_cwnd': {
                'mse': mean_squared_error(y_true, y_pred),
                'mae': mean_absolute_error(y_true, y_pred),
                'r2':  r2_score(y_true, y_pred),
            },
            'cwnd': {
                'mse': mean_squared_error(cwnd_true, cwnd_pred),
                'mae': mean_absolute_error(cwnd_true, cwnd_pred),
                'r2':  r2_score(cwnd_true, cwnd_pred),
            },
            'samples': len(y_true),
        }

        goodput        = test_df['goodput_mbps'].values
        rtt            = test_df['rtt_ms'].values
        loss           = test_df['loss'].values
        objective_true = goodput - self.alpha * rtt - self.beta * loss

        metrics['objective'] = {
            'mean': objective_true.mean(),
            'std':  objective_true.std(),
            'min':  objective_true.min(),
            'max':  objective_true.max(),
        }

        logger.info(f"Test evaluation — Delta R²: {metrics['delta_cwnd']['r2']:.4f}, "
                    f"Cwnd R²: {metrics['cwnd']['r2']:.4f}")
        return metrics

    def get_feature_importance(self) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Model is not trained yet.")

        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_)
        else:
            logger.warning(f"Model type {self.model_type} does not support feature importance")
            return pd.DataFrame()

        return pd.DataFrame({
            'feature':    self.feature_columns,
            'importance': importance,
        }).sort_values('importance', ascending=False)

    def save(self, path: str):
        model_data = {
            'model':           self.model,
            'scaler':          self.scaler,
            'feature_columns': self.feature_columns,
            'alpha':           self.alpha,
            'beta':            self.beta,
            'model_type':      self.model_type,
            'is_fitted':       self.is_fitted,
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        self.model           = model_data['model']
        self.scaler          = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.alpha           = model_data['alpha']
        self.beta            = model_data['beta']
        self.model_type      = model_data['model_type']
        self.is_fitted       = model_data['is_fitted']
        logger.info(f"Model loaded from {path}")

    def extract_algorithm(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> str:
        """
        Extract a hand-written algorithm derived from the actual fitted model.
        For linear models: reads real coefficients and builds the update rule.
        For tree models: uses feature importances + data quantiles to derive thresholds.
        """
        df = pd.concat([train_df, test_df], ignore_index=True)

        rtt_min    = df['rtt_ms'].quantile(0.05)
        rtt_med    = df['rtt_ms'].median()
        rtt_p75    = df['rtt_ms'].quantile(0.75)
        cwnd_med   = df['snd_cwnd'].median()
        cwnd_p75   = df['snd_cwnd'].quantile(0.75)
        gp_med     = df['goodput_mbps'].median()
        loss_p75   = df['loss'].quantile(0.75)
        mss_bytes  = 1448

        bdp_packets = max(1.0, (gp_med * 1e6 / 8) * (rtt_med / 1000.0) / mss_bytes)

        corr_cwnd = df['snd_cwnd'].corr(df['goodput_mbps'])
        corr_rtt  = df['rtt_ms'].corr(df['goodput_mbps'])
        corr_loss = df['loss'].corr(df['goodput_mbps'])

        if hasattr(self.model, 'coef_') and self.feature_columns:
            coefs     = self.model.coef_
            intercept = float(self.model.intercept_)

            terms      = [f"({coefs[i]:+.5f} × {col})"
                          for i, col in enumerate(self.feature_columns)]
            learned_eq = f"  delta_cwnd = {intercept:+.5f}\n"
            learned_eq += "\n".join(f"            {t}" for t in terms)

            ranked = sorted(zip(self.feature_columns, coefs),
                            key=lambda x: abs(x[1]), reverse=True)
            top3 = ranked[:3]

            interpretations = []
            for feat, w in top3:
                sign = "increases" if w > 0 else "decreases"
                if 'cwnd' in feat:
                    interpretations.append(
                        f"  • {feat} (coef={w:+.5f}): cwnd has strong momentum — "
                        f"autoregressive. A {feat} ↑ {sign} delta_cwnd.")
                elif 'rtt' in feat:
                    interpretations.append(
                        f"  • {feat} (coef={w:+.5f}): rising RTT {sign} delta_cwnd. "
                        f"RTT increase signals queue build-up.")
                elif 'goodput' in feat:
                    interpretations.append(
                        f"  • {feat} (coef={w:+.5f}): higher goodput {sign} delta_cwnd. "
                        f"Goodput ≈ cwnd × MSS / RTT (BDP relation).")
                elif 'loss' in feat:
                    interpretations.append(
                        f"  • {feat} (coef={w:+.5f}): loss {sign} delta_cwnd. "
                        f"Loss signals buffer overflow.")
                else:
                    interpretations.append(f"  • {feat} (coef={w:+.5f}): {sign} delta_cwnd.")

            interp_str = "\n".join(interpretations)
            top_feat, top_w     = top3[0]
            second_feat, second_w = top3[1] if len(top3) > 1 else (None, 0)
            third_feat,  third_w  = top3[2] if len(top3) > 2 else (None, 0)
            queue_thresh = rtt_p75

            handwritten = f"""
Simplified Hand-Written Algorithm (derived from model observations)
-------------------------------------------------------------------
  queue_delay = rtt - rtt_min    # isolate queueing component

  if loss > {loss_p75:.1f}:
    cwnd = max(2, cwnd × 0.5)                # AIMD multiplicative decrease

  elif queue_delay > (rtt_min × {(queue_thresh / max(rtt_min, 0.001)):.2f}):
    cwnd = max(2, cwnd - 1)                  # Vegas-like delay decrease

  elif cwnd < {bdp_packets:.1f}:
    cwnd = cwnd + 1                          # probe toward BDP target

  else:
    delta = {top_w:.5f} × {top_feat}"""

            if second_feat:
                handwritten += f"\n          delta += {second_w:.5f} × {second_feat}"
            if third_feat:
                handwritten += f"\n          delta += {third_w:.5f} × {third_feat}"
            handwritten += f"\n    cwnd = max(2, cwnd + delta)\n\n  cwnd = max(2, min(cwnd, 65535))\n"

        else:
            fi = self.get_feature_importance()
            top3 = list(zip(fi['feature'].tolist(), fi['importance'].tolist()))[:3] if len(fi) >= 3 \
                   else list(zip(fi['feature'].tolist(), fi['importance'].tolist()))
            if not top3:
                top3 = [('unknown', 0.0)]

            top_feat, top_imp = top3[0]
            interp_str  = "\n".join(
                f"  • {f} (importance={v:.5f})" for f, v in top3
            )
            learned_eq  = "  (tree model — no closed-form equation; see feature importances)"
            handwritten = f"""
Simplified Hand-Written Algorithm (tree model, importance-derived)
-------------------------------------------------------------------
  queue_delay = rtt - rtt_min

  if loss > {loss_p75:.1f}:                  cwnd = max(2, cwnd × 0.5)
  elif queue_delay > rtt_min × {(rtt_p75 / max(rtt_min, 0.001)):.2f}:  cwnd = max(2, cwnd - 1)
  elif cwnd < {bdp_packets:.1f}:              cwnd = cwnd + 1
  else:                                       cwnd = cwnd + 0.5 / cwnd
  cwnd = max(2, min(cwnd, 65535))

  # Top signal: {top_feat} (importance={top_imp:.5f})
"""

        top_label = top3[0][0]
        top_val   = top3[0][1]

        algorithm = f"""
CONGESTION WINDOW UPDATE ALGORITHM
(Extracted from fitted {self.model_type} model)

Objective function used during training:
  η(t-1) = goodput(t) − {self.alpha}·RTT(t) − {self.beta}·loss(t)

═══════════════════════════════════════════════════════════════════
SECTION 1 — LEARNED MODEL EQUATION
═══════════════════════════════════════════════════════════════════
{learned_eq}

═══════════════════════════════════════════════════════════════════
SECTION 2 — DATA OBSERVATIONS
═══════════════════════════════════════════════════════════════════
  RTT:     min={rtt_min:.2f} ms  median={rtt_med:.2f} ms  p75={rtt_p75:.2f} ms
  cwnd:    median={cwnd_med:.1f} pkts  p75={cwnd_p75:.1f} pkts
  goodput: median={gp_med:.2f} Mbps
  BDP:     {bdp_packets:.1f} packets  (goodput × RTT / MSS=1448B)

  Pearson correlations with goodput:
    cwnd ↔ goodput : {corr_cwnd:+.3f}
    RTT  ↔ goodput : {corr_rtt:+.3f}
    loss ↔ goodput : {corr_loss:+.3f}

═══════════════════════════════════════════════════════════════════
SECTION 3 — DOMINANT FEATURES
═══════════════════════════════════════════════════════════════════
{interp_str}

═══════════════════════════════════════════════════════════════════
SECTION 4 — HAND-WRITTEN ALGORITHM
═══════════════════════════════════════════════════════════════════
{handwritten}
Grounding:
  BDP = {bdp_packets:.1f} pkts — cwnd below this wastes bandwidth, above fills queues.
  RTT_min = {rtt_min:.2f} ms — propagation baseline (5th pctile).
  loss threshold = {loss_p75:.1f} (75th pctile of per-interval events).
  Dominant signal: {top_label} = {top_val:+.5f}
"""
        return algorithm


if __name__ == "__main__":
    predictor = CongestionWindowPredictor(alpha=0.1, beta=1.0)
    print("Congestion Window Predictor initialized")