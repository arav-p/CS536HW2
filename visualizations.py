#!/usr/bin/env python3
"""
Visualization Module for TCP Statistics and ML Model Results
Generates PDF plots for throughput, TCP metrics, and ML predictions.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class Visualizer:

    def __init__(self, output_dir: str = "plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def plot_throughput_timeseries(self, stats_dfs: Dict[str, pd.DataFrame],
                                   output_name: str = "throughput_timeseries.pdf"):
        """Plot throughput time series for all destinations on one figure."""
        plt.figure(figsize=(14, 8))

        for server_id, df in stats_dfs.items():
            plt.plot(df['timestamp'], df['goodput_mbps'],
                     label=server_id, alpha=0.7, linewidth=1.5)

        plt.xlabel('Time (seconds)', fontsize=12)
        plt.ylabel('Goodput (Mbps)', fontsize=12)
        plt.title('Throughput Time Series — All Destinations', fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output_path = self.output_dir / output_name
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved throughput timeseries to {output_path}")

    def plot_tcp_metrics_timeseries(self, df: pd.DataFrame, server_id: str):
        """
        Plot cwnd, RTT, loss, and goodput over time for a single destination.

        FIX: RTT zeros filtered before plotting (early samples before TCP
        warms up report rtt=0 and distort the y-axis).
        FIX: loss computed as per-interval diff of cumulative counter.
        """
        fig, axes = plt.subplots(4, 1, figsize=(14, 12))

        df = df.copy()
        df['rtt_ms'] = df['rtt_us'] / 1000.0

        # FIX: per-interval loss
        df['loss'] = (
            df['retransmits'] + df['lost'] + df['retrans']
        ).diff().fillna(0).clip(lower=0)

        # Plot 1: Congestion Window
        axes[0].plot(df['timestamp'], df['snd_cwnd'], color='blue', linewidth=1.5)
        axes[0].set_ylabel('Congestion Window\n(packets)', fontsize=11)
        axes[0].set_title(f'TCP Metrics Time Series — {server_id}',
                          fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # Plot 2: RTT — FIX: filter zeros so warm-up doesn't distort axis
        df_rtt = df[df['rtt_ms'] > 0]
        axes[1].plot(df_rtt['timestamp'], df_rtt['rtt_ms'], color='green', linewidth=1.5)
        axes[1].set_ylabel('RTT (ms)', fontsize=11)
        axes[1].grid(True, alpha=0.3)

        # Plot 3: Per-interval loss events
        axes[2].plot(df['timestamp'], df['loss'], color='red', linewidth=1.5)
        axes[2].set_ylabel('Loss Events\n(per interval)', fontsize=11)
        axes[2].grid(True, alpha=0.3)

        # Plot 4: Goodput
        axes[3].plot(df['timestamp'], df['goodput_mbps'], color='purple', linewidth=1.5)
        axes[3].set_ylabel('Goodput (Mbps)', fontsize=11)
        axes[3].set_xlabel('Time (seconds)', fontsize=12)
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / f"{server_id}_tcp_timeseries.pdf"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved TCP metrics timeseries to {output_path}")

    def plot_scatter_relationships(self, df: pd.DataFrame, server_id: str):
        """
        Scatter plots: cwnd vs goodput, RTT vs goodput, loss vs goodput.

        FIX: loss is per-interval (diff of cumulative), consistent with
        time series plot. RTT zeros filtered here too.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        df = df.copy()
        df['rtt_ms'] = df['rtt_us'] / 1000.0

        # FIX: per-interval loss
        df['loss'] = (
            df['retransmits'] + df['lost'] + df['retrans']
        ).diff().fillna(0).clip(lower=0)

        df_plot = df[df['snd_cwnd'] > 0].copy()

        # cwnd vs goodput
        axes[0].scatter(df_plot['snd_cwnd'], df_plot['goodput_mbps'],
                        alpha=0.5, s=20, c='blue')
        axes[0].set_xlabel('Congestion Window (packets)', fontsize=11)
        axes[0].set_ylabel('Goodput (Mbps)', fontsize=11)
        axes[0].set_title('Cwnd vs Goodput', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # RTT vs goodput — FIX: filter zeros
        df_rtt = df_plot[df_plot['rtt_ms'] > 0]
        axes[1].scatter(df_rtt['rtt_ms'], df_rtt['goodput_mbps'],
                        alpha=0.5, s=20, c='green')
        axes[1].set_xlabel('RTT (ms)', fontsize=11)
        axes[1].set_ylabel('Goodput (Mbps)', fontsize=11)
        axes[1].set_title('RTT vs Goodput', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        # Loss vs goodput (per-interval events)
        axes[2].scatter(df_plot['loss'], df_plot['goodput_mbps'],
                        alpha=0.5, s=20, c='red')
        axes[2].set_xlabel('Loss Events (per interval)', fontsize=11)
        axes[2].set_ylabel('Goodput (Mbps)', fontsize=11)
        axes[2].set_title('Loss vs Goodput', fontsize=12, fontweight='bold')
        axes[2].grid(True, alpha=0.3)

        plt.suptitle(f'TCP Metrics vs Goodput — {server_id}',
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        output_path = self.output_dir / f"{server_id}_scatter_plots.pdf"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved scatter plots to {output_path}")

    def plot_summary_table(self, summaries: Dict[str, Dict],
                           output_name: str = "summary_table.pdf"):
        """Summary table of min/median/avg/p95/max goodput per destination."""
        table_data = []
        for server_id, summary in summaries.items():
            g = summary['goodput']
            table_data.append([
                server_id,
                f"{g['min']:.2f}",
                f"{g['median']:.2f}",
                f"{g['mean']:.2f}",
                f"{g['p95']:.2f}",
                f"{g['max']:.2f}",
            ])

        fig, ax = plt.subplots(figsize=(12, len(table_data) * 0.5 + 2))
        ax.axis('tight')
        ax.axis('off')

        headers = ['Server', 'Min (Mbps)', 'Median (Mbps)', 'Avg (Mbps)',
                   'P95 (Mbps)', 'Max (Mbps)']
        table = ax.table(cellText=table_data, colLabels=headers,
                         cellLoc='center', loc='center',
                         colWidths=[0.25, 0.15, 0.15, 0.15, 0.15, 0.15])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        for i in range(len(headers)):
            cell = table[(0, i)]
            cell.set_facecolor('#4CAF50')
            cell.set_text_props(weight='bold', color='white')

        for i in range(1, len(table_data) + 1):
            for j in range(len(headers)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')

        plt.title('Throughput Summary Statistics', fontsize=14,
                  fontweight='bold', pad=20)
        output_path = self.output_dir / output_name
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved summary table to {output_path}")

    def plot_ml_predictions(self, df: pd.DataFrame, predictions: np.ndarray,
                            server_id: str, split_idx: int):
        """Actual vs predicted cwnd for a single server (two-panel: full + test zoom)."""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        timestamps  = df['timestamp'].values
        actual_cwnd = df['snd_cwnd'].values

        # Full timeseries with train/test split marker
        axes[0].plot(timestamps[:split_idx], actual_cwnd[:split_idx],
                     label='Train (Actual)', color='blue', linewidth=1.5, alpha=0.7)
        axes[0].plot(timestamps[split_idx:], actual_cwnd[split_idx:],
                     label='Test (Actual)', color='green', linewidth=1.5, alpha=0.7)
        axes[0].plot(timestamps[split_idx:], predictions,
                     label='Test (Predicted)', color='red',
                     linewidth=1.5, linestyle='--', alpha=0.8)
        axes[0].axvline(x=timestamps[split_idx], color='black',
                        linestyle=':', linewidth=2, label='Train/Test Split')
        axes[0].set_ylabel('Congestion Window (packets)', fontsize=11)
        axes[0].set_title(f'ML Model: Actual vs Predicted Cwnd — {server_id}',
                          fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # Test period zoomed
        axes[1].plot(timestamps[split_idx:], actual_cwnd[split_idx:],
                     label='Actual', color='green', linewidth=2, alpha=0.7)
        axes[1].plot(timestamps[split_idx:], predictions,
                     label='Predicted', color='red', linewidth=2,
                     linestyle='--', alpha=0.8)
        axes[1].set_xlabel('Time (seconds)', fontsize=12)
        axes[1].set_ylabel('Congestion Window (packets)', fontsize=11)
        axes[1].set_title('Test Period (Zoomed)', fontsize=12, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / f"{server_id}_ml_predictions.pdf"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved ML predictions plot to {output_path}")

    def plot_ml_predictions_multiple(self, results_list: List[Dict],
                                     output_name: str = "ml_predictions_all.pdf"):
        """ML predictions for multiple servers stacked in one PDF."""
        n_plots = len(results_list)
        fig, axes = plt.subplots(n_plots, 1, figsize=(14, 5 * n_plots))

        if n_plots == 1:
            axes = [axes]

        for idx, result in enumerate(results_list):
            df          = result['df']
            predictions = result['predictions']
            server_id   = result['server_id']
            split_idx   = result['split_idx']

            timestamps  = df['timestamp'].values
            actual_cwnd = df['snd_cwnd'].values

            axes[idx].plot(timestamps[:split_idx], actual_cwnd[:split_idx],
                           label='Train', color='blue', linewidth=1.0, alpha=0.6)
            axes[idx].plot(timestamps[split_idx:], actual_cwnd[split_idx:],
                           label='Test (Actual)', color='green', linewidth=1.5)
            axes[idx].plot(timestamps[split_idx:], predictions,
                           label='Test (Predicted)', color='red',
                           linewidth=1.5, linestyle='--', alpha=0.8)
            axes[idx].axvline(x=timestamps[split_idx], color='black',
                              linestyle=':', linewidth=1.5, alpha=0.7)
            axes[idx].set_ylabel('Cwnd (packets)', fontsize=10)
            axes[idx].set_title(server_id, fontsize=11, fontweight='bold')
            axes[idx].legend(fontsize=9, loc='best')
            axes[idx].grid(True, alpha=0.3)

        axes[-1].set_xlabel('Time (seconds)', fontsize=11)
        plt.suptitle('ML Model Predictions — Multiple Destinations',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        output_path = self.output_dir / output_name
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved multi-server ML predictions to {output_path}")


if __name__ == "__main__":
    viz = Visualizer()
    print("Visualizer initialized")