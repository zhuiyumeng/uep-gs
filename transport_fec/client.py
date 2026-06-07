"""
RTP Client with optional FEC.

Usage:
    python -m transport_fec.client \\
        --host 127.0.0.1 --port 5005 \\
        --output received.ply --loss 0.05
"""

import os
import sys
import socket
import argparse
import logging
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transport_fec.rtp_packet import RTPPacket
from transport_fec.decoder import RTPDecoder, write_ply
from common.cli_utils import add_shared_client_args

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('client')


def main():
    parser = argparse.ArgumentParser(description='3DGS RTP Client (FEC)')
    add_shared_client_args(parser)
    parser.add_argument(
        '--no-fec', action='store_true', default=False,
        dest='no_fec',
        help='Disable FEC decoding (ignore FECParity packets)',
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
    sock.settimeout(args.timeout)
    sock.connect((args.host, args.port))

    decoder = RTPDecoder(use_fec=not args.no_fec)
    frame_data: bytes | None = None
    frame_meta = None

    def on_complete():
        nonlocal frame_data, frame_meta
        frame_data = decoder.get_complete_frame()
        frame_meta = decoder.meta

    decoder._on_frame_complete = on_complete

    sock.send(b'hello')
    log.info(
        'Connected to %s:%s%s',
        args.host, args.port,
        ' [FEC ON]' if not args.no_fec else ' [FEC OFF]',
    )

    total_packets = 0
    lost_packets = 0
    try:
        while True:
            data = sock.recv(65535)
            total_packets += 1

            if args.loss > 0 and random.random() < args.loss:
                lost_packets += 1
                continue

            pkt = RTPPacket.parse(data)
            decoder.feed_packet(pkt)

            if frame_data is not None and frame_meta is not None:
                output_dir = os.path.dirname(args.output) or '.'
                os.makedirs(output_dir, exist_ok=True)
                write_ply(args.output, frame_data, frame_meta)
                size_mb = len(frame_data) / 1024 / 1024
                log.info(
                    'Frame complete | %d packets received | %.2f MB → %s',
                    total_packets, size_mb, args.output,
                )
                frame_data = None
                frame_meta = None
                break

    except socket.timeout:
        log.warning(
            'Timeout after %.0f s (received %d packets, lost %d)',
            args.timeout, total_packets, lost_packets,
        )
        if args.flush_on_timeout:
            decoder.flush()
            if frame_data is not None and frame_meta is not None:
                output_dir = os.path.dirname(args.output) or '.'
                os.makedirs(output_dir, exist_ok=True)
                write_ply(args.output, frame_data, frame_meta)
                size_mb = len(frame_data) / 1024 / 1024
                log.info(
                    'Flushed incomplete frame | %d packets | %.2f MB → %s',
                    total_packets, size_mb, args.output,
                )
            else:
                log.error('Failed to flush: no frame data available')
    except KeyboardInterrupt:
        log.info('Interrupted (received %d packets)', total_packets)

    sock.close()
    log.info('Done.')


if __name__ == '__main__':
    main()
