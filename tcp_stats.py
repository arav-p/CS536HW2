#!/usr/bin/env python3
"""
TCP Statistics Collection and Analysis Module
Processes TCP socket statistics from iperf tests and prepares data for analysis.
"""

import json
import csv
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class TCPStatsProcessor:
    """Process and store TCP statistics from iperf tests."""

    def __init__(self, output_dir: str = "data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def process_test_results(self, results: Dict, server_id: str):
        """Process results from a single test and save to files."""
        if not results or not results.get('success'):
            logger.warning(f"No valid results for {server_id}")
            return

        samples = results['samples']
        data_records = []

        for sample in samples:
            record = {
                'timestamp':      sample['timestamp'],
                'bytes_sent':     sample['bytes_sent'],
                'interval_bytes': sample['interval_bytes'],
                'bytes_acked':    sample.get('bytes_acked', sample['interval_bytes']),
                'goodput_mbps':   sample['goodput_mbps'],
            }

            tcp_info = sample.get('tcp_info')
            if tcp_info:
                record['snd_cwnd']    = tcp_info.get('snd_cwnd', 0)
                record['rtt_us']      = tcp_info.get('rtt', 0)
                record['rttvar_us']   = tcp_info.get('rttvar', 0)
                record['retransmits'] = tcp_info.get('retransmits', 0)
                record['lost']        = tcp_info.get('lost', 0)
                record['retrans']     = tcp_info.get('retrans', 0)
                record['snd_ssthresh']= tcp_info.get('snd_ssthresh', 0)
                record['unacked']     = tcp_info.get('unacked', 0)
                record['sacked']      = tcp_info.get('sacked', 0)
                record['pmtu']        = tcp_info.get('pmtu', 0)
                record['rcv_ssthresh']= tcp_info.get('rcv_ssthresh', 0)
            else:
                for field in ['snd_cwnd', 'rtt_us', 'rttvar_us', 'retransmits',
                              'lost', 'retrans', 'snd_ssthresh', 'unacked',
                              'sacked', 'pmtu', 'rcv_ssthresh']:
                    record[field] = 0

            data_records.append(record)

        csv_path = self.output_dir / f"{server_id}_stats.csv"
        self._save_csv(data_records, csv_path)

        json_path = self.output_dir / f"{server_id}_stats.json"
        self._save_json({
            'server':      results['host'],
            'duration':    results['duration'],
            'total_bytes': results['total_bytes'],
            'summary':     results['summary'],
            'samples':     data_records
        }, json_path)

        logger.info(f"Saved stats for {server_id} to {csv_path}")

    def _save_csv(self, records: List[Dict], path: Path):
        if not records:
            return
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)

    def _save_json(self, data: Dict, path: Path):
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load_stats(self, server_id: str) -> pd.DataFrame:
        csv_path = self.output_dir / f"{server_id}_stats.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Stats file not found: {csv_path}")
        return pd.read_csv(csv_path)

    def compute_loss_signal(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute per-interval loss signal from TCP stats.

        FIX: retransmits/lost/retrans are cumulative counters from TCP_INFO.
        Taking diff() gives events per interval. clip(lower=0) prevents
        negative values from occasional counter resets.
        """
        cumulative = (
            df['retransmits'].fillna(0)
            + df['lost'].fillna(0)
            + df['retrans'].fillna(0)
        )
        return cumulative.diff().fillna(0).clip(lower=0)

    def compute_rtt_ms(self, df: pd.DataFrame) -> pd.Series:
        """Convert RTT from microseconds to milliseconds."""
        return df['rtt_us'] / 1000.0

    def get_summary_statistics(self, server_id: str) -> Dict:
        df = self.load_stats(server_id)

        return {
            'num_samples':  len(df),
            'duration':     df['timestamp'].max(),
            'total_bytes':  df['bytes_sent'].max(),
            'goodput': {
                'min':    df['goodput_mbps'].min(),
                'max':    df['goodput_mbps'].max(),
                'mean':   df['goodput_mbps'].mean(),
                'median': df['goodput_mbps'].median(),
                'p95':    df['goodput_mbps'].quantile(0.95),
                'std':    df['goodput_mbps'].std(),
            },
            'cwnd': {
                'min':    df['snd_cwnd'].min(),
                'max':    df['snd_cwnd'].max(),
                'mean':   df['snd_cwnd'].mean(),
                'median': df['snd_cwnd'].median(),
            },
            'rtt': {
                'min':    df['rtt_us'].min()    / 1000.0,
                'max':    df['rtt_us'].max()    / 1000.0,
                'mean':   df['rtt_us'].mean()   / 1000.0,
                'median': df['rtt_us'].median() / 1000.0,
            },
            'loss': {
                'total_retransmits': df['retransmits'].sum(),
                'total_lost':        df['lost'].sum(),
                'total_retrans':     df['retrans'].sum(),
                'max_retransmits':   df['retransmits'].max(),
            }
        }

    def prepare_ml_dataset(self, server_ids: List[str]) -> pd.DataFrame:
        """Prepare dataset for ML model from multiple server tests."""
        all_data = []

        for server_id in server_ids:
            try:
                df = self.load_stats(server_id)
                df['server_id'] = server_id

                df['rtt_ms'] = self.compute_rtt_ms(df)

                # FIX: per-interval loss, not cumulative sum
                df['loss'] = self.compute_loss_signal(df)

                # Label: delta of congestion window
                df['delta_cwnd'] = df['snd_cwnd'].diff().fillna(0)

                # Lag features
                df['goodput_lag1'] = df['goodput_mbps'].shift(1).fillna(df['goodput_mbps'])
                df['rtt_lag1']     = df['rtt_ms'].shift(1).fillna(df['rtt_ms'])
                df['cwnd_lag1']    = df['snd_cwnd'].shift(1).fillna(df['snd_cwnd'])

                # Moving averages
                df['goodput_ma3'] = df['goodput_mbps'].rolling(window=3, min_periods=1).mean()
                df['rtt_ma3']     = df['rtt_ms'].rolling(window=3, min_periods=1).mean()

                all_data.append(df)

            except Exception as e:
                logger.error(f"Error loading data for {server_id}: {e}")
                continue

        if not all_data:
            raise ValueError("No valid data loaded")

        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.replace([np.inf, -np.inf], np.nan)
        combined_df = combined_df.fillna(0)

        return combined_df

    def split_train_test(self, df: pd.DataFrame, test_split: float = 0.3) -> tuple:
        """
        Split dataset into train and test sets chronologically per server.
        Time-series data must never be shuffled — future data cannot train on past.
        """
        train_dfs = []
        test_dfs  = []

        for server_id in df['server_id'].unique():
            server_df = df[df['server_id'] == server_id].copy()
            split_idx = int(len(server_df) * (1 - test_split))
            train_dfs.append(server_df.iloc[:split_idx])
            test_dfs.append(server_df.iloc[split_idx:])

        return (
            pd.concat(train_dfs, ignore_index=True),
            pd.concat(test_dfs,  ignore_index=True),
        )


if __name__ == "__main__":
    processor = TCPStatsProcessor()
    print("TCP Stats Processor initialized")