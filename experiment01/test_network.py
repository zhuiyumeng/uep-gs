#!/usr/bin/env python
"""
experiment01/test_network.py
实验 2: NetworkEstimator 测试

测试项目：
1. EMA 平滑响应：丢包率序列 0% → 5% → 10% → 3%
2. 冷启动不退化：首个报告 p=0 时不锁定在 0
3. Per-LOD EMA 平滑
4. RTT 估计计算
"""

import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transport_fec_ml.network_estimator import NetworkEstimator, NetworkState
from transport_fec_ml.rtcp import (
    RTCPReceiverReport, RTCPReportBlock, FECStatsReport,
)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(name)s | %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('test_network')


def make_rr(fraction_lost: float, jitter: int = 0,
            last_sr: int = 0, dlsr: int = 0) -> RTCPReceiverReport:
    """构造一个模拟 RTCP Receiver Report"""
    rb = RTCPReportBlock(
        ssrc=1,
        fraction_lost=int(fraction_lost * 256),
        cumulative_lost=0,
        ext_highest_seq=0,
        interarrival_jitter=jitter,
        last_sr=last_sr,
        delay_since_last_sr=dlsr,
    )
    return RTCPReceiverReport(ssrc=2, report_blocks=[rb])


def make_fec_stats(per_lod_data: list[tuple[int, int, int, int]]
                   ) -> FECStatsReport:
    """构造 FECStats

    Args:
        per_lod_data: [(total, lost, recovered, unrecovered), ...]
    """
    total = [d[0] for d in per_lod_data]
    lost = [d[1] for d in per_lod_data]
    recovered = [d[2] for d in per_lod_data]
    unrecovered = [d[3] for d in per_lod_data]
    return FECStatsReport(
        per_lod_total=total,
        per_lod_lost=lost,
        per_lod_recovered=recovered,
        per_lod_unrecovered=unrecovered,
    )


def main():
    print("=" * 70)
    print("  Experiment 2: NetworkEstimator Test")
    print("=" * 70)

    # --- Test 1: EMA 平滑 + 冷启动 ---
    print("\n" + "-" * 60)
    print("  Test 2.1: EMA Smoothing + Cold Start")
    print("  Loss sequence: 0% → 5% → 10% → 3%")
    print("-" * 60)

    estimator = NetworkEstimator(alpha=0.3, alpha_jitter=0.3)

    test_sequence = [
        # (fraction_lost_raw, expected_behavior)
        (0.00, "cold start: should NOT be 0"),
        (0.05, "rising"),
        (0.10, "peak"),
        (0.10, "sustained peak"),
        (0.03, "falling"),
        (0.03, "sustained low"),
    ]

    print(f"\n  {'Step':>5s}  {'Raw':>7s}  {'EMA Loss':>10s}  {'Jitter':>8s}  "
          f"{'RTT':>8s}  {'Note'}")
    print(f"  {'-'*5}  {'-'*7}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*20}")

    for i, (p_raw, note) in enumerate(test_sequence):
        rr = make_rr(p_raw)
        estimator.feed_rtcp_report(rr)
        state = estimator.state
        print(f"  {i:5d}  {p_raw:7.3f}  {state.loss_rate:10.4f}  "
              f"{state.jitter:8.1f}  {state.rtt_estimate:8.4f}  {note}")

    # 验证冷启动：首个 p=0 时 EMA 不应为 0
    est2 = NetworkEstimator(alpha=0.3)
    rr_zero = make_rr(0.0)
    est2.feed_rtcp_report(rr_zero)
    cold_start_val = est2.loss_rate
    expected_prior = (0.6 * 0.0 + 0.4 * 0.05)  # = 0.02
    print(f"\n  Cold start check:")
    print(f"    First RR with p=0 → EMA loss_rate = {cold_start_val:.4f}")
    print(f"    Expected ≈ {expected_prior:.4f} (0.6*0 + 0.4*0.05)")
    if abs(cold_start_val - expected_prior) < 0.001:
        print(f"    ✅ Cold start working correctly (not 0!)")
    else:
        print(f"    ❌ Cold start mismatch")

    # 验证收敛
    rr_high = make_rr(0.10)
    for _ in range(5):
        est2.feed_rtcp_report(rr_high)
    print(f"    After 5× p=0.10: loss_rate = {est2.loss_rate:.4f} "
          f"(should approach 0.10)")

    # --- Test 2: Per-LOD EMA ---
    print("\n" + "-" * 60)
    print("  Test 2.2: Per-LOD EMA Smoothing")
    print("-" * 60)

    est3 = NetworkEstimator(alpha=0.3)
    # Feed 3 times with same per-LOD stats
    for t in range(3):
        fs = make_fec_stats([
            (1000, 50 + t * 10, 40, 10 + t * 10),   # LOD 0: loss increasing
            (5000, 200, 150, 50),                     # LOD 1: stable
            (30000, 1000, 800, 200),                  # LOD 2: stable
            (420000, 20000, 15000, 5000),             # LOD 3: stable
        ])
        rr_sim = make_rr(200 / 5000)
        est3.feed_rtcp_report(rr_sim, fs)

    state = est3.state
    print(f"\n  Per-LOD stats after 3 reports (EMA smoothed):")
    print(f"  {'LOD':>5s}  {'Loss Rate':>12s}  {'FEC Recovery':>14s}")
    print(f"  {'-'*5}  {'-'*12}  {'-'*14}")
    for i in range(len(state.per_lod_loss_rate)):
        print(f"  {i:5d}  {state.per_lod_loss_rate[i]:12.4f}  "
              f"{state.per_lod_fec_recovery_rate[i]:14.4f}")

    # --- Test 3: RTT Estimation ---
    print("\n" + "-" * 60)
    print("  Test 2.3: RTT Estimation (from SR/DLSR)")
    print("-" * 60)

    import time
    est4 = NetworkEstimator(alpha=0.3)
    now_ntp = int(time.time() * 65536) & 0xFFFFFFFF
    fake_lsr = now_ntp - 5000   # SR 在 ~76ms 前发出
    fake_dlsr = 1000            # 客户端处理延迟 ~15ms

    rr_with_rtt = make_rr(0.05, last_sr=fake_lsr, dlsr=fake_dlsr)
    est4.feed_rtcp_report(rr_with_rtt)

    expected_rtt = (now_ntp - fake_lsr - fake_dlsr) / 65536.0
    print(f"    fake_last_sr = {fake_lsr}")
    print(f"    fake_dlsr    = {fake_dlsr}")
    print(f"    now_ntp      = {now_ntp}")
    print(f"    expected RTT ≈ {expected_rtt:.4f}s")
    print(f"    estimated RTT = {est4.state.rtt_estimate:.4f}s")
    if abs(est4.state.rtt_estimate - expected_rtt) < 0.001:
        print(f"    ✅ RTT estimation correct")
    else:
        print(f"    ⚠️  RTT estimation needs review")

    # --- Test 4: Reset ---
    print("\n" + "-" * 60)
    print("  Test 2.4: Reset")
    print("-" * 60)

    est4.reset()
    assert est4.loss_rate == 0.0
    assert est4.state.rtt_estimate == 0.0
    assert est4._report_count == 0
    assert est4._per_lod_loss_ema == []
    print("    ✅ Reset works correctly")

    print("\n" + "=" * 70)
    print("  Experiment 2 Complete.")
    print("=" * 70)


if __name__ == '__main__':
    main()
