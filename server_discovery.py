#!/usr/bin/env python3
"""
Loads the iperf3 server list from the provided CSV file and selects n random
reachable servers.
"""

import csv
import random
import socket
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

DEFAULT_PORT = 5201
DEFAULT_CSV = "listed_iperf3_servers-2.csv"


def _parse_port(port_str: str) -> int:
    """Parse a port or port range (e.g. '5201' or '5201-5209') -> first port."""
    port_str = port_str.strip()
    if not port_str:
        return DEFAULT_PORT
    try:
        return int(port_str.split("-")[0])
    except ValueError:
        return DEFAULT_PORT


def load_from_csv(path: str = DEFAULT_CSV) -> List[Tuple[str, int]]:
    """
    Load servers from the iperf3 server list CSV.
    Expected columns: IP/HOST, PORT, ...
    """
    servers = []
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                host = row.get("IP/HOST", "").strip()
                port = _parse_port(row.get("PORT", ""))
                if host:
                    servers.append((host, port))
        logger.info(f"Loaded {len(servers)} servers from {path}")
    except FileNotFoundError:
        logger.error(f"Server list CSV not found: {path}")
    return servers


def _is_reachable(host: str, port: int, timeout: int = 5) -> bool:
    """Quick TCP connect check."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (socket.timeout, socket.error, OSError):
        return False


def select_servers(n: int, csv_path: str = DEFAULT_CSV,
                   probe: bool = True, probe_timeout: int = 5) -> List[Tuple[str, int]]:
    """
    Pick n random servers from the CSV, probing reachability before selecting.

    Args:
        n: Number of servers to return.
        csv_path: Path to the server list CSV.
        probe: If True, skip servers that don't accept a TCP connection.
        probe_timeout: Timeout in seconds for each probe.

    Returns:
        List of (host, port) tuples, length <= n.
    """
    pool = load_from_csv(csv_path)
    if not pool:
        return []

    random.shuffle(pool)

    if not probe:
        selected = pool[:n]
        logger.info(f"Selected {len(selected)} servers (no probing).")
        return selected

    selected = []
    for host, port in pool:
        if len(selected) >= n:
            break
        logger.info(f"Probing {host}:{port} ...")
        if _is_reachable(host, port, probe_timeout):
            selected.append((host, port))
            logger.info(f"  reachable ({len(selected)}/{n})")
        else:
            logger.info(f"  unreachable, skipping")

    logger.info(f"Selected {len(selected)} reachable servers.")
    return selected
