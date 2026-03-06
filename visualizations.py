#!/usr/bin/env python3
"""
Visualization Module for TCP Statistics and ML Model Results
Generates PDF plots for throughput, TCP metrics, and ML predictions.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class Visualizer:
    """Create visualizations for TCP statistics and ML results."""

    def __init__(self, output_dir: str = "plots"):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory to save plot files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def plot_throughput_timeseries(self, stats_dfs: Dict[str, pd.DataFrame],
                                    output_name: str = "throughput_timeseries.pdf"):
        """
        Plot throughput time series for multiple destinations.

        Args:
            stats_dfs: Dictionary mapping server_id to DataFrame
            output_name: Output filename
        """
        plt.figure(figsize=(14, 8))

        for server_id, df in stats_dfs.items():
            plt.plot(df['timestamp'], df['goodput_mbps'],
                     label=server_id, alpha=0.7, linewidth=1.5)

        plt.xlabel('Time (seconds)', fontsize=12)
        plt.ylabel('Goodput (Mbps)', fontsize=12)
        plt.title('Throughput Time Series - All Destinations', fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output_path = self.output_dir / output_name
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved throughput timeseries plot to {output_path}")

    def plot_tcp_metrics_timeseries(self, df: pd.DataFrame, server_id: str):
        """
        Plot TCP metrics (cwnd, RTT, loss, throughput) over time for a single destination.

        Args:
            df: DataFrame with TCP statistics
            server_id: Server identifier for the plot
        """
        fig, axes = plt.subplots(4, 1, figsize=(14, 12))

        # Prepare data
        df['rtt_ms'] = df['rtt_us'] / 1000.0
        df['loss'] = df['retransmits'] + df['lost'] + df['retrans']

        # Plot 1: Congestion Window
        axes[0].plot(df['timestamp'], df['snd_cwnd'], color='blue', linewidth=1.5)
        axes[0].set_ylabel('Congestion Window\n(packets)', fontsize=11)
        axes[0].set_title(f'TCP Metrics Time Series - {server_id}', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # Plot 2: RTT
        axes[1].plot(df['timestamp'], df['rtt_ms'], color='green', linewidth=1.5)
        axes[1].set_ylabel('RTT (ms)', fontsize=11)
        axes[1].grid(True, alpha=0.3)

        # Plot 3: Loss Signal
        axes[2].plot(df['timestamp'], df['loss'], color='red', linewidth=1.5)
        axes[2].set_ylabel('Loss Signal\n(cumulative)', fontsize=11)
        axes[2].grid(True, alpha=0.3)

        # Plot 4: Throughput
        axes[3].plot(df['timestamp'], df['goodput_mbps'], color='purple', linewidth=1.5)
        axes[3].set_ylabel('Goodput (Mbps)', fontsize=11)
        axes[3].set_xlabel('Time (seconds)', fontsize=12)
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()

        output_path = self.output_dir / f"{server_id}_tcp_timeseries.pdf"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved TCP metrics timeseries plot to {output_path}")

    def plot_scatter_relationships(self, df: pd.DataFrame, server_id: str):
        """
        Plot scatter plots showing relationships between TCP metrics and throughput.

        Args:
            df: DataFrame with TCP statistics
            server_id: Server identifier
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Prepare data
        df['rtt_ms'] = df['rtt_us'] / 1000.0
        df['loss'] = df['retransmits'] + df['lost'] + df['retrans']

        # Remove zeros for better visualization
        df_plot = df[df['snd_cwnd'] > 0].copy()

        # Plot 1: cwnd vs goodput
        axes[0].scatter(df_plot['snd_cwnd'], df_plot['goodput_mbps'],
                        alpha=0.5, s=20, c='blue')
        axes[0].set_xlabel('Congestion Window (packets)', fontsize=11)
        axes[0].set_ylabel('Goodput (Mbps)', fontsize=11)
        axes[0].set_title('Cwnd vs Goodput', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # Plot 2: RTT vs goodput
        df_rtt = df_plot[df_plot['rtt_ms'] > 0]
        axes[1].scatter(df_rtt['rtt_ms'], df_rtt['goodput_mbps'],
                        alpha=0.5, s=20, c='green')
        axes[1].set_xlabel('RTT (ms)', fontsize=11)
        axes[1].set_ylabel('Goodput (Mbps)', fontsize=11)
        axes[1].set_title('RTT vs Goodput', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        # Plot 3: Loss vs goodput
        axes[2].scatter(df_plot['loss'], df_plot['goodput_mbps'],
                        alpha=0.5, s=20, c='red')
        axes[2].set_xlabel('Loss Signal (cumulative)', fontsize=11)
        axes[2].set_ylabel('Goodput (Mbps)', fontsize=11)
        axes[2].set_title('Loss vs Goodput', fontsize=12, fontweight='bold')
        axes[2].grid(True, alpha=0.3)

        plt.suptitle(f'TCP Metrics vs Goodput Relationships - {server_id}',
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        output_path = self.output_dir / f"{server_id}_scatter_plots.pdf"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved scatter plots to {output_path}")

    def plot_summary_table(self, summaries: Dict[str, Dict],
                           output_name: str = "summary_table.pdf"):
        """
        Create a summary table showing statistics for all destinations.

        Args:
            summaries: Dictionary mapping server_id to summary statistics
            output_name: Output filename
        """
        # Prepare data for table
        table_data = []
        for server_id, summary in summaries.items():
            goodput = summary['goodput']
            row = [
                server_id,
                f"{goodput['min']:.2f}",
                f"{goodput['median']:.2f}",
                f"{goodput['mean']:.2f}",
                f"{goodput['p95']:.2f}",
                f"{goodput['max']:.2f}"
            ]
            table_data.append(row)

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

        # Style header
        for i in range(len(headers)):
            cell = table[(0, i)]
            cell.set_facecolor('#4CAF50')
            cell.set_text_props(weight='bold', color='white')

        # Alternate row colors
        for i in range(1, len(table_data) + 1):
            for j in range(len(headers)):
                cell = table[(i, j)]
                if i % 2 == 0:
                    cell.set_facecolor('#f0f0f0')

        plt.title('Throughput Summary Statistics', fontsize=14,
                  fontweight='bold', pad=20)

        output_path = self.output_dir / output_name
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved summary table to {output_path}")

    def plot_ml_predictions(self, df: pd.DataFrame, predictions: np.ndarray,
                            server_id: str, split_idx: int):
        """
        Plot actual vs predicted cwnd for ML model evaluation.

        Args:
            df: DataFrame with actual data
            predictions: Array of predicted cwnd values
            server_id: Server identifier
            split_idx: Index where test split begins
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Prepare data
        timestamps = df['timestamp'].values
        actual_cwnd = df['snd_cwnd'].values

        # Plot 1: Full timeseries with train/test split
        axes[0].plot(timestamps[:split_idx], actual_cwnd[:split_idx],
                     label='Train (Actual)', color='blue', linewidth=1.5, alpha=0.7)
        axes[0].plot(timestamps[split_idx:], actual_cwnd[split_idx:],
                     label='Test (Actual)', color='green', linewidth=1.5, alpha=0.7)

        # Only plot predictions for test period
        test_timestamps = timestamps[split_idx:]
        axes[0].plot(test_timestamps, predictions,
                     label='Test (Predicted)', color='red',
                     linewidth=1.5, linestyle='--', alpha=0.8)

        axes[0].axvline(x=timestamps[split_idx], color='black',
                        linestyle=':', linewidth=2, label='Train/Test Split')
        axes[0].set_ylabel('Congestion Window (packets)', fontsize=11)
        axes[0].set_title(f'ML Model: Actual vs Predicted Cwnd - {server_id}',
                          fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Zoomed in on test period only
        axes[1].plot(test_timestamps, actual_cwnd[split_idx:],
                     label='Actual', color='green', linewidth=2, alpha=0.7)
        axes[1].plot(test_timestamps, predictions,
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
        """
        Plot ML predictions for multiple servers in a grid.

        Args:
            results_list: List of dicts with keys: 'server_id', 'df', 'predictions', 'split_idx'
            output_name: Output filename
        """
        n_plots = len(results_list)
        fig, axes = plt.subplots(n_plots, 1, figsize=(14, 5 * n_plots))

        if n_plots == 1:
            axes = [axes]

        for idx, result in enumerate(results_list):
            df = result['df']
            predictions = result['predictions']
            server_id = result['server_id']
            split_idx = result['split_idx']

            timestamps = df['timestamp'].values
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
            axes[idx].set_title(f'{server_id}', fontsize=11, fontweight='bold')
            axes[idx].legend(fontsize=9, loc='best')
            axes[idx].grid(True, alpha=0.3)

        axes[-1].set_xlabel('Time (seconds)', fontsize=11)
        plt.suptitle('ML Model Predictions - Multiple Destinations',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        output_path = self.output_dir / output_name
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved multi-server ML predictions plot to {output_path}")


if __name__ == "__main__":
    # Example usage
    viz = Visualizer()
    print("Visualizer initialized")
