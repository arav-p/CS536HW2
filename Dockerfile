FROM ubuntu:24.04

# Avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python packages
RUN pip3 install --no-cache-dir -r requirements.txt --break-system-packages

# Copy all source code
COPY iperf_client.py .
COPY tcp_stats.py .
COPY visualizations.py .
COPY ml_model.py .
COPY server_discovery.py .
COPY main.py .
COPY listed_iperf3_servers-2.csv .

# Create output directories
RUN mkdir -p data plots

# Make scripts executable
RUN chmod +x main.py iperf_client.py

# Default command: run main pipeline
CMD ["python3", "main.py", "--csv", "listed_iperf3_servers-2.csv", "--num-servers", "10", "--duration", "60"]
