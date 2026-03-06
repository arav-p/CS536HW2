#!/usr/bin/env python3
"""
iPerf3 Client Implementation
Implements the iperf3 protocol to communicate with standard iperf3 servers.
"""

import socket
import json
import time
import struct
import threading
import logging
import random
import string
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ThroughputSample:
    """Single throughput measurement sample"""
    timestamp: float
    bytes_sent: int
    interval_bytes: int
    goodput_mbps: float
    tcp_info: Optional[Dict] = None


class IPerf3Client:
    """
    iPerf3 protocol client that connects to standard iperf3 servers.
    Implements control connection, parameter negotiation, and data transfer.
    """

    IPERF_CONTROL_PORT = 5201
    COOKIE_SIZE = 37          # 36 alphanumeric chars + 1 null terminator

    # iperf3 control channel state bytes
    TEST_START       = 1
    TEST_RUNNING     = 2
    TEST_END         = 4
    PARAM_EXCHANGE   = 9
    CREATE_STREAMS   = 10
    EXCHANGE_RESULTS = 13
    DISPLAY_RESULTS  = 14
    ACCESS_DENIED    = -1
    SERVER_ERROR     = -2

    def __init__(self, host: str, port: int = 5201, duration: int = 60,
                 sample_interval: float = 0.2, timeout: int = 10):
        self.host = host
        self.port = port
        self.duration = duration
        self.sample_interval = sample_interval
        self.timeout = timeout

        self.control_sock = None
        self.data_sock = None
        self.cookie = None

        self.samples: List[ThroughputSample] = []
        self.total_bytes_sent = 0
        self.last_sample_bytes = 0
        self.start_time = None
        self.running = False
        self._bytes_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Establish control connection and perform the full iperf3 handshake."""
        try:
            self.control_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.control_sock.settimeout(self.timeout)

            logger.info(f"Connecting to {self.host}:{self.port}")
            self.control_sock.connect((self.host, self.port))

            # Step 1: Client generates and sends the cookie (37 bytes)
            self.cookie = self._generate_cookie()
            cookie_bytes = (self.cookie + '\0').encode('ascii')  # exactly 37 bytes
            self.control_sock.sendall(cookie_bytes)
            logger.info(f"Sent cookie: {self.cookie}")

            # Step 2: Server sends PARAM_EXCHANGE (byte 9)
            state = self._recv_state()
            if state is None:
                logger.error("No state byte received after cookie")
                return False
            if state in (self.ACCESS_DENIED, self.SERVER_ERROR):
                logger.error(f"Server rejected connection with state {state}")
                return False
            if state != self.PARAM_EXCHANGE:
                logger.error(f"Expected PARAM_EXCHANGE ({self.PARAM_EXCHANGE}), got {state}")
                return False

            # Step 3: Client sends JSON parameters
            if not self._send_parameters():
                return False

            # Step 4: Server sends CREATE_STREAMS (byte 10)
            state = self._recv_state()
            if state is None:
                logger.error("No state byte received after parameter exchange")
                return False
            if state in (self.ACCESS_DENIED, self.SERVER_ERROR):
                logger.error(f"Server rejected parameters with state {state}")
                return False
            if state != self.CREATE_STREAMS:
                logger.error(f"Expected CREATE_STREAMS ({self.CREATE_STREAMS}), got {state}")
                return False

            # Step 5: Open data connection and send cookie on it
            if not self._create_streams():
                return False

            logger.info("Successfully connected and negotiated parameters")
            return True

        except socket.timeout:
            logger.error(f"Connection timeout to {self.host}:{self.port}")
            return False
        except socket.error as e:
            logger.error(f"Socket error: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during connection: {e}")
            return False

    def run_test(self) -> bool:
        """Run the throughput test and perform the iperf3 result exchange."""
        if not self.control_sock or not self.data_sock:
            logger.error("Not connected. Call connect() first.")
            return False

        try:
            self.running = True
            self.start_time = time.time()
            self.total_bytes_sent = 0
            self.last_sample_bytes = 0

            # Step 6: Server sends TEST_START (byte 1)
            state = self._recv_state()
            if state != self.TEST_START:
                logger.error(f"Expected TEST_START ({self.TEST_START}), got {state}")
                self.running = False
                return False

            # Step 7: Server sends TEST_RUNNING (byte 2)
            state = self._recv_state()
            if state != self.TEST_RUNNING:
                logger.error(f"Expected TEST_RUNNING ({self.TEST_RUNNING}), got {state}")
                self.running = False
                return False

            # Start sampling thread
            sample_thread = threading.Thread(target=self._sampling_loop, daemon=True)
            sample_thread.start()

            # Data transfer loop
            data_buffer = b'0' * (128 * 1024)  # 128KB send buffer
            end_time = self.start_time + self.duration
            logger.info(f"Starting test for {self.duration} seconds")

            self.data_sock.settimeout(self.timeout)
            while time.time() < end_time and self.running:
                try:
                    sent = self.data_sock.send(data_buffer)
                    if sent == 0:
                        logger.error("Connection closed by server during data transfer")
                        self.running = False
                        break
                    with self._bytes_lock:
                        self.total_bytes_sent += sent
                except socket.timeout:
                    logger.error("Data send timed out — server may be non-responsive")
                    self.running = False
                    break
                except socket.error as e:
                    logger.error(f"Error sending data: {e}")
                    self.running = False
                    break

            self.running = False
            sample_thread.join(timeout=2)

            logger.info(f"Test completed. Total bytes sent: {self.total_bytes_sent}")

            # Step 8: Termination sequence
            self._terminate_test()

            return True

        except Exception as e:
            logger.error(f"Error during test: {e}")
            self.running = False
            return False

    def get_results(self) -> Dict:
        """Return test results. Samples are returned as plain dicts (JSON-serializable)."""
        if not self.samples:
            return {'success': False, 'error': 'No samples collected'}

        goodput_values = [s.goodput_mbps for s in self.samples]
        sorted_vals = sorted(goodput_values)

        return {
            'success': True,
            'host': self.host,
            'duration': self.duration,
            'total_bytes': self.total_bytes_sent,
            'num_samples': len(self.samples),
            'samples': [asdict(s) for s in self.samples],
            'summary': {
                'min_goodput_mbps': min(goodput_values),
                'max_goodput_mbps': max(goodput_values),
                'avg_goodput_mbps': sum(goodput_values) / len(goodput_values),
                'median_goodput_mbps': sorted_vals[len(sorted_vals) // 2],
                'p95_goodput_mbps': sorted_vals[int(len(sorted_vals) * 0.95)],
            }
        }

    def close(self):
        """Close all connections."""
        for sock_attr in ('data_sock', 'control_sock'):
            sock = getattr(self, sock_attr)
            if sock:
                try:
                    sock.close()
                except Exception as e:
                    logger.debug(f"Error closing {sock_attr}: {e}")
                setattr(self, sock_attr, None)

    # ------------------------------------------------------------------
    # Protocol helpers
    # ------------------------------------------------------------------

    def _generate_cookie(self) -> str:
        """Generate a 36-character alphanumeric cookie per iperf3 protocol."""
        chars = string.ascii_lowercase + string.digits
        return ''.join(random.choices(chars, k=36))

    def _send_state(self, state: int) -> bool:
        """Send a single signed-byte state token on the control connection."""
        try:
            self.control_sock.sendall(struct.pack('!b', state))
            return True
        except Exception as e:
            logger.error(f"Failed to send state {state}: {e}")
            return False

    def _recv_state(self) -> Optional[int]:
        """Read a single signed-byte state token from the control connection."""
        data = self._recv_exact(self.control_sock, 1)
        if data is None:
            return None
        return struct.unpack('!b', data)[0]

    def _send_parameters(self) -> bool:
        """Send test parameters as a length-prefixed JSON message."""
        params = {
            "tcp": True,
            "omit": 0,
            "time": self.duration,
            "parallel": 1,
            "len": 128 * 1024,
            "client_version": "3.9",
        }
        return self._send_json(self.control_sock, params)

    def _create_streams(self) -> bool:
        """Open the data connection and send the cookie on it (called after CREATE_STREAMS)."""
        try:
            self.data_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.data_sock.settimeout(self.timeout)
            self.data_sock.connect((self.host, self.port))

            # Send the same 37-byte cookie (36 chars + null) on the data connection
            cookie_bytes = (self.cookie + '\0').encode('ascii')
            self.data_sock.sendall(cookie_bytes)

            logger.info("Data connection established")
            return True
        except Exception as e:
            logger.error(f"Failed to create data connection: {e}")
            return False

    def _terminate_test(self):
        """
        Perform the iperf3 test termination sequence:
          Client  -> TEST_END (4)
          Server  -> EXCHANGE_RESULTS (13) + JSON server results
          Client  -> EXCHANGE_RESULTS (13) + JSON client results
          Server  -> DISPLAY_RESULTS (14)
        """
        try:
            # Close write side of data socket to signal end-of-stream
            try:
                self.data_sock.shutdown(socket.SHUT_WR)
            except Exception as e:
                logger.debug(f"data_sock shutdown: {e}")

            # Send TEST_END on control channel
            if not self._send_state(self.TEST_END):
                return

            # Receive EXCHANGE_RESULTS from server
            state = self._recv_state()
            if state != self.EXCHANGE_RESULTS:
                logger.warning(f"Expected EXCHANGE_RESULTS ({self.EXCHANGE_RESULTS}), got {state}")
                return

            # Receive server's JSON results
            server_results = self._recv_json(self.control_sock)
            if server_results:
                logger.info(f"Server results: {json.dumps(server_results, indent=2)}")

            # Send EXCHANGE_RESULTS + our own JSON results
            if not self._send_state(self.EXCHANGE_RESULTS):
                return

            client_results = {
                "bytes_sent": self.total_bytes_sent,
                "seconds": self.duration,
            }
            self._send_json(self.control_sock, client_results)

            # Receive DISPLAY_RESULTS
            state = self._recv_state()
            if state == self.DISPLAY_RESULTS:
                logger.info("Test termination sequence complete")
            else:
                logger.debug(f"Final state byte: {state}")

        except Exception as e:
            logger.debug(f"Error during test termination: {e}")

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sampling_loop(self):
        """Background thread to collect throughput and TCP stats samples."""
        last_sample_time = self.start_time

        while self.running:
            time.sleep(self.sample_interval)

            current_time = time.time()
            elapsed = current_time - last_sample_time

            if elapsed < self.sample_interval * 0.9:
                continue

            with self._bytes_lock:
                current_total = self.total_bytes_sent
            interval_bytes = current_total - self.last_sample_bytes
            goodput_mbps = (interval_bytes * 8) / (elapsed * 1_000_000)

            tcp_info = self._get_tcp_info()

            sample = ThroughputSample(
                timestamp=current_time - self.start_time,
                bytes_sent=current_total,
                interval_bytes=interval_bytes,
                goodput_mbps=goodput_mbps,
                tcp_info=tcp_info,
            )
            self.samples.append(sample)

            self.last_sample_bytes = current_total
            last_sample_time = current_time

    def _get_tcp_info(self) -> Optional[Dict]:
        """Extract TCP socket statistics (Linux TCP_INFO)."""
        if not self.data_sock:
            return None
        try:
            if hasattr(socket, 'TCP_INFO'):
                raw = self.data_sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_INFO, 256)
                if len(raw) < 104:
                    return {'note': f'TCP_INFO too short: {len(raw)} bytes'}
                fields = struct.unpack('B' * 8 + 'I' * 24, raw[:104])
                return {
                    'state': fields[0],
                    'ca_state': fields[1],
                    'retransmits': fields[2],
                    'probes': fields[3],
                    'backoff': fields[4],
                    'rto': fields[8],
                    'ato': fields[9],
                    'snd_mss': fields[10],
                    'rcv_mss': fields[11],
                    'unacked': fields[12],
                    'sacked': fields[13],
                    'lost': fields[14],
                    'retrans': fields[15],
                    'fackets': fields[16],
                    'last_data_sent': fields[17],
                    'last_ack_sent': fields[18],
                    'last_data_recv': fields[19],
                    'last_ack_recv': fields[20],
                    'pmtu': fields[21],
                    'rcv_ssthresh': fields[22],
                    'rtt': fields[23],
                    'rttvar': fields[24],
                    'snd_ssthresh': fields[25],
                    'snd_cwnd': fields[26],
                    'advmss': fields[27],
                    'reordering': fields[28],
                }
            else:
                return {'platform': 'non-linux', 'note': 'TCP_INFO not available on this platform'}
        except Exception as e:
            logger.debug(f"Could not get TCP_INFO: {e}")
            return None

    # ------------------------------------------------------------------
    # Low-level socket I/O
    # ------------------------------------------------------------------

    def _send_json(self, sock: socket.socket, data: dict) -> bool:
        """Send a length-prefixed JSON message."""
        try:
            json_bytes = json.dumps(data).encode('utf-8')
            sock.sendall(struct.pack('!I', len(json_bytes)))
            sock.sendall(json_bytes)
            return True
        except Exception as e:
            logger.error(f"Error sending JSON: {e}")
            return False

    def _recv_json(self, sock: socket.socket) -> Optional[dict]:
        """Receive a length-prefixed JSON message."""
        try:
            length_bytes = self._recv_exact(sock, 4)
            if not length_bytes:
                return None
            length = struct.unpack('!I', length_bytes)[0]
            if length > 1024 * 1024:
                logger.error(f"JSON message too large: {length}")
                return None
            json_bytes = self._recv_exact(sock, length)
            if not json_bytes:
                return None
            return json.loads(json_bytes.decode('utf-8'))
        except Exception as e:
            logger.error(f"Error receiving JSON: {e}")
            return None

    def _recv_exact(self, sock: socket.socket, length: int) -> Optional[bytes]:
        """Receive exactly 'length' bytes from socket."""
        data = b''
        while len(data) < length:
            try:
                chunk = sock.recv(length - len(data))
                if not chunk:
                    return None
                data += chunk
            except socket.timeout:
                logger.debug("Timeout waiting for data")
                return None
            except socket.error as e:
                logger.debug(f"Socket error in recv_exact: {e}")
                return None
        return data


# ----------------------------------------------------------------------
# Module-level helper
# ----------------------------------------------------------------------

def test_server(host: str, port: int = 5201, duration: int = 60,
                sample_interval: float = 0.2) -> Optional[Dict]:
    """
    Test a single iperf3 server and return results.

    Returns:
        Results dictionary or None if test failed
    """
    client = IPerf3Client(host, port, duration, sample_interval)
    try:
        if not client.connect():
            logger.warning(f"Failed to connect to {host}:{port}")
            return None
        if not client.run_test():
            logger.warning(f"Test failed for {host}:{port}")
            return None
        return client.get_results()
    except Exception as e:
        logger.error(f"Exception testing {host}: {e}")
        return None
    finally:
        client.close()


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="iPerf3 Python client")
    parser.add_argument("host", help="iperf3 server hostname or IP")
    parser.add_argument("-p", "--port", type=int, default=5201, help="Server port (default 5201)")
    parser.add_argument("-t", "--time", type=int, default=10, dest="duration",
                        help="Test duration in seconds (default 10)")
    parser.add_argument("-i", "--interval", type=float, default=0.2,
                        help="Sampling interval in seconds (default 0.2)")
    args = parser.parse_args()

    result = test_server(args.host, args.port, args.duration, args.interval)

    if result and result['success']:
        print(f"\nTest Results for {args.host}:")
        print(f"  Total bytes sent : {result['total_bytes']:,}")
        print(f"  Samples collected: {result['num_samples']}")
        print(f"\nThroughput Summary:")
        for key, value in result['summary'].items():
            print(f"  {key}: {value:.2f} Mbps")
    else:
        print(f"Test failed for {args.host}")
        sys.exit(1)
