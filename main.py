#!/usr/bin/env python3
"""
Main Orchestrator Script for CS536 HW2
Fully automated pipeline: iperf tests -> TCP stats -> ML model -> visualizations
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List
import pandas as pd

from iperf_client import test_server
from tcp_stats import TCPStatsProcessor
from visualizations import Visualizer
from ml_model import CongestionWindowPredictor
from server_discovery import select_servers

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_iperf_tests(num_servers: int, duration: int,
                    sample_interval: float, csv_path: str) -> List[tuple]:
    """
    Test servers from the CSV until num_servers succeed.
    Automatically replaces servers that fail the probe or the iperf handshake.
    """
    from server_discovery import load_from_csv, _is_reachable
    import random

    logger.info(f"Loading server pool from {csv_path} ...")
    pool = load_from_csv(csv_path)
    if not pool:
        logger.error("No servers found in CSV.")
        return []

    random.shuffle(pool)
    pool_iter    = iter(pool)
    results_list = []
    attempted    = 0
    server_index = 0

    logger.info(f"Running iperf tests — need {num_servers} successful, will retry on failure")

    while len(results_list) < num_servers:
        try:
            host, port = next(pool_iter)
        except StopIteration:
            logger.warning("Exhausted all servers in the pool.")
            break

        attempted += 1
        label = f"{host}:{port}"

        logger.info(f"Probing {label} ...")
        if not _is_reachable(host, port, timeout=5):
            logger.info(f"  unreachable, trying next server")
            continue

        server_id = f"server_{server_index:02d}_{host.replace('.', '_').replace('-', '_')}"
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing ({len(results_list)+1}/{num_servers} needed): {label}")
        logger.info(f"{'='*60}")

        result = test_server(host, port, duration, sample_interval)

        if result and result.get('success'):
            results_list.append((server_id, result))
            server_index += 1
            logger.info(f"✓ Success: {label} ({len(results_list)}/{num_servers})")
        else:
            logger.warning(f"✗ Failed: {label}, trying replacement...")

    logger.info(f"\nCompleted: {len(results_list)}/{num_servers} successful "
                f"out of {attempted} attempted")
    return results_list


def save_and_visualize_part1(results_list: List[tuple], processor: TCPStatsProcessor,
                              viz: Visualizer):
    """Part 1: Save results and create throughput visualizations."""
    logger.info("\n" + "="*60)
    logger.info("PART 1: Throughput Analysis")
    logger.info("="*60)

    for server_id, result in results_list:
        processor.process_test_results(result, server_id)

    stats_dfs = {}
    summaries  = {}

    for server_id, _ in results_list:
        try:
            df = processor.load_stats(server_id)
            stats_dfs[server_id] = df
            summaries[server_id] = processor.get_summary_statistics(server_id)
        except Exception as e:
            logger.error(f"Error loading stats for {server_id}: {e}")

    if stats_dfs:
        viz.plot_throughput_timeseries(stats_dfs, "part1_throughput_timeseries.pdf")
        viz.plot_summary_table(summaries, "part1_summary_table.pdf")

    logger.info("Part 1 complete.")


def analyze_tcp_stats_part2(results_list: List[tuple], processor: TCPStatsProcessor,
                             viz: Visualizer):
    """Part 2: TCP statistics analysis and visualization."""
    logger.info("\n" + "="*60)
    logger.info("PART 2: TCP Statistics Analysis")
    logger.info("="*60)

    if not results_list:
        logger.error("No successful tests for Part 2 analysis")
        return

    server_id, _ = results_list[0]
    logger.info(f"Using {server_id} as representative destination")

    df = processor.load_stats(server_id)

    if 'rtt_ms' not in df.columns:
        df['rtt_ms'] = df['rtt_us'] / 1000.0
    if 'loss' not in df.columns:
        df['loss'] = (
            df['retransmits'] + df['lost'] + df['retrans']
        ).diff().fillna(0).clip(lower=0)

    viz.plot_tcp_metrics_timeseries(df, server_id)
    viz.plot_scatter_relationships(df, server_id)

    logger.info("\nTCP Metrics Observations:")
    logger.info(f"  Avg Cwnd    : {df['snd_cwnd'].mean():.1f} packets")
    logger.info(f"  Avg RTT     : {df['rtt_ms'].mean():.2f} ms")
    logger.info(f"  Total losses: {df['loss'].sum():.0f} events")
    logger.info(f"  Avg Goodput : {df['goodput_mbps'].mean():.2f} Mbps")
    logger.info("Part 2 complete.")


def train_ml_model_part3(results_list: List[tuple], processor: TCPStatsProcessor,
                         viz: Visualizer, alpha: float, beta: float):
    """Part 3: ML model training and evaluation."""
    logger.info("\n" + "="*60)
    logger.info("PART 3: ML Model Training")
    logger.info("="*60)

    server_ids = [sid for sid, _ in results_list]
    logger.info(f"Preparing ML dataset from {len(server_ids)} servers...")

    dataset = processor.prepare_ml_dataset(server_ids)
    logger.info(f"Dataset shape: {dataset.shape}")

    train_df, test_df = processor.split_train_test(dataset, test_split=0.3)
    logger.info(f"Train: {len(train_df)} samples  |  Test: {len(test_df)} samples")

    predictor = CongestionWindowPredictor(alpha=alpha, beta=beta,
                                          model_type='linear')

    train_metrics = predictor.train(train_df)
    logger.info(f"Training metrics: {train_metrics}")

    test_metrics = predictor.evaluate(test_df)
    logger.info(f"Test metrics: {test_metrics}")

    importance = predictor.get_feature_importance()
    logger.info("\nFeature Importance:")
    logger.info(importance.to_string(index=False))

    model_path = Path("data") / "cwnd_predictor.pkl"
    predictor.save(str(model_path))

    # Generate predictions for up to 5 servers
    plot_servers    = server_ids[:5]
    results_for_plot = []

    for server_id in plot_servers:
        server_df = dataset[dataset['server_id'] == server_id].copy()
        split_idx = int(len(server_df) * 0.7)

        test_portion = server_df.iloc[split_idx:].copy()
        predictions  = predictor.predict_cwnd_sequence(test_portion)

        results_for_plot.append({
            'server_id':   server_id,
            'df':          server_df,
            'predictions': predictions,
            'split_idx':   split_idx,
        })

        viz.plot_ml_predictions(server_df, predictions, server_id, split_idx)

    viz.plot_ml_predictions_multiple(results_for_plot, "part3_ml_predictions_all.pdf")

    algorithm      = predictor.extract_algorithm(train_df, test_df)
    algorithm_path = Path("data") / "extracted_algorithm.txt"
    with open(algorithm_path, 'w') as f:
        f.write(algorithm)

    logger.info(f"Extracted algorithm saved to {algorithm_path}")
    logger.info("Part 3 complete.")
    return algorithm


def main():
    parser = argparse.ArgumentParser(
        description='CS536 HW2: Automated iperf, TCP stats, and ML pipeline'
    )
    parser.add_argument('--csv', type=str, default='listed_iperf3_servers-2.csv')
    parser.add_argument('--num-servers', type=int, default=10)
    parser.add_argument('--duration',    type=int, default=60)
    parser.add_argument('--sample-interval', type=float, default=0.2)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--beta',  type=float, default=1.0)
    parser.add_argument('--quick-test', action='store_true',
                        help='Quick test mode: 3 servers, 10s duration')
    args = parser.parse_args()

    if args.quick_test:
        args.num_servers = 3
        args.duration    = 10
        logger.info("QUICK TEST MODE: 3 servers, 10s duration")

    processor = TCPStatsProcessor(output_dir="data")
    viz       = Visualizer(output_dir="plots")

    results_list = run_iperf_tests(args.num_servers, args.duration,
                                   args.sample_interval, args.csv)

    if not results_list:
        logger.error("No successful tests completed. Exiting.")
        sys.exit(1)

    save_and_visualize_part1(results_list, processor, viz)
    analyze_tcp_stats_part2(results_list, processor, viz)
    algorithm = train_ml_model_part3(results_list, processor, viz,
                                     args.alpha, args.beta)

    logger.info("\n" + "="*60)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*60)
    logger.info("  TCP stats (CSV/JSON) : data/")
    logger.info("  Plots (PDF)          : plots/")
    logger.info("  ML model             : data/cwnd_predictor.pkl")
    logger.info("  Algorithm            : data/extracted_algorithm.txt")
    logger.info("\n" + algorithm)


if __name__ == "__main__":
    main()