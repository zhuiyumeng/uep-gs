"""
transport_fec_ml RTP Client — 配合 ML 自适应 FEC 服务端

接收 ML 优化的 RTP 流并解码输出 PLY。

Usage:
    python -m transport_fec_ml.client \\
        --host 127.0.0.1 --port 5005 \\
        --output received.ply --loss 0.05 --ml-policy 0.03
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

from .adaptive_fec import AnalyticalOptimizer, AdaptiveFECPolicy

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('ml-client')


def main():
    parser = argparse.ArgumentParser(description='3DGS RTP Client (ML-FEC)')
    add_shared_client_args(parser)
    parser.add_argument(
        '--no-fec', action='store_true', default=False,
        dest='no_fec',
        help='Disable FEC decoding (ignore FECParity packets)',
    )
    parser.add_argument(
        '--ml-policy', type=float, default=None, dest='ml_policy_loss',
        metavar='LOSS_RATE',
        help='ML DP policy loss rate (must match server side)',
    )
    parser.add_argument(
        '--ml-budget', type=float, default=0.20, dest='ml_budget',
        help='Bandwidth budget (must match server side)',
    )
    parser.add_argument(
        '--lod-info', type=str, default=None, dest='lod_info',
        metavar='GAUSSIANS_PER_LOD',
        help='Comma-separated gaussian counts per LOD for DP computation',
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    # ── ML policy ──
    fec_policy = None
    if args.ml_policy_loss is not None:
        if not args.lod_info:
            log.warning(
                '[ML Policy] --lod-info required for client-side DP; '
                'using default UEP_POLICY instead'
            )
        else:
            lod_g_counts = [int(x) for x in args.lod_info.split(',')]
            # 从高斯数估算包数（与 server 端一致）
            gaussian_stride = 17 + 3 * ((3 + 1) ** 2 - 1)  # 62 for sh_degree=3
            data_per_packet = 1392
            lod_packet_counts = []
            for n_g in lod_g_counts:
                lod_bytes = n_g * gaussian_stride * 4
                lod_packet_counts.append(
                    (lod_bytes + data_per_packet - 1) // data_per_packet
                )

            log.info(
                '[ML Policy] Computing DP allocation: loss=%.1f%%, budget=%.0f%%',
                args.ml_policy_loss * 100, args.ml_budget * 100,
            )
            allocation = AnalyticalOptimizer.optimize(
                lod_sizes=lod_packet_counts,
                loss_rate=args.ml_policy_loss,
                bandwidth_budget=args.ml_budget,
                lod_gaussian_counts=lod_g_counts,
            )
            fec_policy = AdaptiveFECPolicy(allocation)
            log.info('[ML Policy] %s', allocation.summary())

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
    sock.settimeout(args.timeout)
    sock.connect((args.host, args.port))

    decoder = RTPDecoder(use_fec=not args.no_fec, fec_policy=fec_policy)
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
        ' [FEC ON + ML]' if (fec_policy and not args.no_fec)
        else (' [FEC ON]' if not args.no_fec else ' [FEC OFF]'),
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
