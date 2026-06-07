"""
RTP Server with optional FEC.

Usage:
    python -m transport_fec.server \\
        --lod_dir ./transport_data --prefix model \\
        --sh_degree 3 --port 5005
"""

import os
import sys
import socket
import argparse
import time
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transport_fec.encoder import RTPEncoder
from common.cli_utils import find_lod_plys, _fmt_size, add_shared_server_args

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('server')


def main():
    parser = argparse.ArgumentParser(description='3DGS RTP Server (FEC)')
    add_shared_server_args(parser)
    parser.add_argument(
        '--no-fec', action='store_true', default=False,
        dest='no_fec',
        help='Disable FEC forward error correction',
    )
    args = parser.parse_args()

    lod_files = find_lod_plys(args.lod_dir, args.prefix)
    if not lod_files:
        log.error(
            'No LOD PLY files found in %s with prefix "%s"',
            args.lod_dir, args.prefix,
        )
        sys.exit(1)

    log.info('Found %d LOD files:', len(lod_files))
    for f in lod_files:
        sz = os.path.getsize(f)
        log.info('  %s  (%s)', os.path.basename(f), _fmt_size(sz))

    encoder = RTPEncoder()
    encoder.load_lod_plys(lod_files, args.sh_degree)
    meta = encoder.meta
    log.info(
        'Loaded: %d gaussians, %d LODs, %.2f MB, SH degree %d%s',
        meta.total_gaussians, meta.num_lods,
        meta.total_bytes / 1024 / 1024,
        meta.sh_degree,
        ' [FEC ON]' if not args.no_fec else ' [FEC OFF]',
    )

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
    sock.bind((args.host, args.port))
    sock.settimeout(1.0)

    log.info('Server ready on %s:%d', args.host, args.port)
    log.info('Waiting for client handshake ...')

    frame_id = 0
    seq = 0

    try:
        while True:
            try:
                data, addr = sock.recvfrom(1024)
                log.info('Client connected: %s:%s', *addr)
            except socket.timeout:
                continue

            while True:
                try:
                    ts = frame_id * (90000 // args.fps)
                    packets = encoder.encode_frame(
                        seq, ts, args.mtu, fec=not args.no_fec,
                    )
                    total = len(packets)
                    for i, pkt in enumerate(packets):
                        sock.sendto(pkt.serialize(), addr)
                        if args.pace_ms > 0 and i % args.pace_every == (args.pace_every - 1):
                            time.sleep(args.pace_ms / 1000.0)
                    log.info('Frame %d | %d packets sent', frame_id, total)
                    seq += total

                    mbps = (
                        meta.total_bytes * args.fps * 8 / 1024 / 1024
                    )
                    log.info(
                        'Frame %d | %d packets | %.2f MB | ~%.0f Mbps',
                        frame_id, total, meta.total_bytes / 1024 / 1024,
                        mbps,
                    )

                    frame_id += 1
                    if not args.loop:
                        break
                    if args.oneshot:
                        log.info('Oneshot: sent 1 frame, waiting for next client')
                        break
                    time.sleep(1.0 / args.fps)

                except socket.timeout:
                    log.info('Client disconnected.')
                    break
                except OSError as e:
                    log.warning('Send error: %s', e)
                    break

            if not args.loop:
                break

    except KeyboardInterrupt:
        log.info('Server stopped.')

    sock.close()


if __name__ == '__main__':
    main()
