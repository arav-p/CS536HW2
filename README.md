# CS536 HW2: iPerf, TCP Statistics, and Congestion Control

Complete implementation of Assignment 2 for CS 536 Spring 2026.

## Project Structure

```
CS536HW2/
├── main.py                         # Main orchestrator script
├── iperf_client.py                 # iPerf3 protocol client implementation
├── tcp_stats.py                    # TCP statistics collection and processing
├── visualizations.py               # Plotting utilities
├── ml_model.py                     # ML model for cwnd prediction
├── server_discovery.py             # Server loading and reachability probing
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Docker containerization (ubuntu:24.04)
├── run_experiment.sh               # One-command build + run script
├── listed_iperf3_servers-2.csv     # iPerf3 server list
├── data/                           # Output: CSV/JSON data (created at runtime)
└── plots/                          # Output: PDF plots (created at runtime)
```

## Requirements

- **Linux** (Ubuntu 22.04+ recommended) or **Windows with WSL2**
- Docker installed
- Internet connection for accessing public iPerf3 servers

**Note:** macOS is NOT supported due to TCP_INFO limitations.

## Quick Start

### Using Docker (Recommended)

Run the entire experiment with a single command:
```bash
./run_experiment.sh
```

This builds the image, creates output directories, and runs the pipeline. Results are written directly to `./data` and `./plots` on the host via volume mounts.

Or manually:
```bash
docker build -t cs536hw2 .
docker run --rm \
  --network host \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/plots:/app/plots \
  cs536hw2
```

### Running Directly on Linux

1. Install dependencies:
```bash
pip3 install -r requirements.txt
```

2. Run the pipeline:
```bash
python3 main.py --csv listed_iperf3_servers-2.csv --num-servers 10 --duration 60
```

## Usage

### Full Pipeline
```bash
python3 main.py --csv listed_iperf3_servers-2.csv \
                --num-servers 10 \
                --duration 60 \
                --sample-interval 0.2 \
                --alpha 0.1 \
                --beta 1.0
```

### Command Line Arguments

- `--csv FILE`: Path to server list CSV (default: listed_iperf3_servers-2.csv)
- `--num-servers N`: Number of random servers to test (default: 10)
- `--duration SEC`: Test duration per server (default: 60)
- `--sample-interval SEC`: Sampling interval (default: 0.2)
- `--alpha FLOAT`: RTT penalty weight in objective function (default: 0.1)
- `--beta FLOAT`: Loss penalty weight in objective function (default: 1.0)
- `--quick-test`: Quick test mode (3 servers, 10s duration)

## What the Script Does

### Part 1: iPerf Throughput Tests
1. Selects N random servers from the list
2. Implements full iPerf3 protocol (control connection, parameter exchange, data transfer)
3. Measures goodput at regular intervals
4. Handles server failures gracefully
5. Outputs:
   - `data/server_XX_stats.csv` - Raw statistics for each server
   - `plots/part1_throughput_timeseries.pdf` - Time series of all servers
   - `plots/part1_summary_table.pdf` - Summary statistics table

### Part 2: TCP Statistics Analysis
1. Collects TCP socket statistics (cwnd, RTT, loss, etc.) via `TCP_INFO`
2. Analyzes relationship between TCP metrics and throughput
3. Outputs:
   - `plots/server_XX_tcp_timeseries.pdf` - Time series of TCP metrics
   - `plots/server_XX_scatter_plots.pdf` - Scatter plots (cwnd/RTT/loss vs goodput)

### Part 3: ML Model
1. Prepares dataset from all collected traces
2. Trains linear regression model to predict delta cwnd (cwnd updates)
3. Uses custom objective function: η(t-1) = goodput(t) - α·RTT(t) - β·loss(t), with weight at t-1 derived from outcome at t
4. Evaluates on test set (30% split)
5. Derives hand-written algorithm dynamically from learned coefficients, data percentiles, and BDP
6. Outputs:
   - `data/cwnd_predictor.pkl` - Trained model
   - `plots/part3_ml_predictions_all.pdf` - Predictions for multiple servers
   - `plots/server_XX_ml_predictions.pdf` - Individual prediction plots
   - `data/extracted_algorithm.txt` - Hand-written algorithm

## Output Files

All outputs are generated automatically:

### Data Files (data/)
- `server_XX_stats.csv` - Per-server TCP statistics
- `server_XX_stats.json` - Per-server results with metadata
- `cwnd_predictor.pkl` - Trained ML model
- `extracted_algorithm.txt` - Hand-written congestion control algorithm

### Plots (plots/)
- `part1_throughput_timeseries.pdf` - Throughput for all destinations
- `part1_summary_table.pdf` - Summary statistics table
- `server_XX_tcp_timeseries.pdf` - TCP metrics time series
- `server_XX_scatter_plots.pdf` - Relationship scatter plots
- `server_XX_ml_predictions.pdf` - ML predictions per server
- `part3_ml_predictions_all.pdf` - Combined ML predictions

## Implementation Details

### iPerf3 Protocol
The client implements the complete iPerf3 protocol:
1. Control connection on port 5201
2. Cookie exchange
3. JSON parameter negotiation
4. Data connection establishment
5. Continuous data transfer with statistics collection
6. Graceful termination

### TCP Statistics Collection
Uses Linux `TCP_INFO` socket option to extract:
- `snd_cwnd`: Congestion window
- `rtt`, `rttvar`: Round-trip time and variance
- `retransmits`, `lost`, `retrans`: Loss indicators
- `snd_ssthresh`: Slow start threshold
- Other TCP state variables

### ML Model
- Algorithm: Linear Regression (sklearn)
- Features: goodput_mbps, rtt_ms, loss, snd_cwnd, lagged cwnd values, moving averages
- Target: delta_cwnd (change in congestion window)
- Training: Weighted by objective function η with correct off-by-one shift (weight[t-1] ← outcome at t)
- Evaluation: MSE, MAE, R² on both delta and absolute cwnd
- Algorithm extraction: dynamically derived from learned coefficients, RTT/loss percentiles, and BDP

### Objective Function
η(t-1) = goodput(t) - α·RTT(t) - β·loss(t)

Where:
- goodput(t): Measured throughput
- RTT(t): Round-trip time (penalizes latency)
- loss(t): Packet loss count (penalizes loss)
- α, β: Tunable weights (default: α=0.1, β=1.0)

## Troubleshooting

### "Connection refused" errors
- Some servers may be offline or rate-limiting
- Script automatically skips failed servers and tries others

### "TCP_INFO not available"
- You're not running on Linux/WSL2
- Run in Docker with `--network=host` flag

### Insufficient successful tests
- Increase `--num-servers` to test more servers
- Some public servers are unreliable; script will retry others

### Out of memory
- Reduce `--duration` or `--num-servers`
- Increase Docker memory limit

## Testing

Minimal test to verify everything works:
```bash
python3 main.py --csv listed_iperf3_servers-2.csv --num-servers 3 --duration 10
```

This runs 3 servers for 10 seconds each and generates all outputs.

## Notes

- Assignment requires Linux kernel for TCP statistics
- Docker container uses host network stack (`--network=host`)
- All experiments are fully reproducible via Docker
- Random server selection ensures variety across runs
- Robust error handling for unreliable public servers

## Assignment Completion

This implementation satisfies all requirements:

✅ **Part 1 (20 pts)**: Complete iPerf3 client with protocol implementation
✅ **Part 2 (40 pts)**: TCP statistics collection and visualization
✅ **Part 3 (40 pts)**: ML model with custom objective and extracted algorithm
✅ **Automation**: Single script runs entire pipeline
✅ **Docker**: Containerized with ubuntu:24.04, one-command execution via `run_experiment.sh`
✅ **Outputs**: All CSV, JSON, and PDF files generated automatically

## Author

CS536 Spring 2026 - Assignment 2 Implementation
