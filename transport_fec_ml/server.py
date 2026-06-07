"""
transport_fec_ml RTP Server — 使用 ML 自适应 FEC 策略

基于 DP 优化器动态计算最优 RS(k,n) 分配，替代硬编码 UEP_POLICY。

Usage:
    python -m transport_fec_ml.server \\
        --lod_dir ./data --prefix model \\
        --sh_degree 3 --port 5005 --ml-policy 0.03
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

from .adaptive_fec import AnalyticalOptimizer, AdaptiveFECPolicy

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('ml-server')


def main():
    parser = argparse.ArgumentParser(description='3DGS RTP Server (ML-FEC)')
    add_shared_server_args(parser)
    parser.add_argument(
        '--no-fec', action='store_true', default=False,
        dest='no_fec',
        help='Disable FEC forward error correction',
    )
    parser.add_argument(
        '--ml-policy', type=float, default=None, dest='ml_policy_loss',
        metavar='LOSS_RATE',
        help='Use ML DP optimizer with target loss rate (e.g. 0.03)',
    )
    parser.add_argument(
        '--ml-budget', type=float, default=0.20, dest='ml_budget',
        help='Bandwidth budget for DP optimizer (default: 0.20)',
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

    # ── ML policy: 预加载 PLY 获取 LOD 分布 ──
    fec_policy = None
    if args.ml_policy_loss is not None:
        # 先加载无 FEC 的临时 encoder 只是为了获取 meta
        tmp_encoder = RTPEncoder()
        tmp_encoder.load_lod_plys(lod_files, args.sh_degree)
        meta = tmp_encoder.meta

        # 从 meta 推算每 LOD 的数据包数
        # gaussian_stride * 4 bytes per gaussian, DATA_PER_PACKET = 1392
        data_per_packet = 1392
        lod_packet_counts = []
        for n_g in meta.lod_sizes:
            lod_bytes = n_g * meta.gaussian_stride * 4
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
            lod_gaussian_counts=meta.lod_sizes,
        )
        fec_policy = AdaptiveFECPolicy(allocation)
        log.info('[ML Policy] %s', allocation.summary())
    else:
        log.info('[ML Policy] Using default UEP_POLICY (no --ml-policy)')

    # ── 正式 encoder ──
    encoder = RTPEncoder(fec_policy=fec_policy)
    encoder.load_lod_plys(lod_files, args.sh_degree)
    meta = encoder.meta
    log.info(
        'Loaded: %d gaussians, %d LODs, %.2f MB, SH degree %d%s',
        meta.total_gaussians, meta.num_lods,
        meta.total_bytes / 1024 / 1024,
        meta.sh_degree,
        ' [FEC ON + ML]' if (fec_policy and not args.no_fec)
        else (' [FEC ON]' if not args.no_fec else ' [FEC OFF]'),
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
