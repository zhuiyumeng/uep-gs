#!/usr/bin/env python
"""
experiment01/test_pipeline.py
实验 4: AdaptiveFECPipeline 闭环测试

模拟多帧场景，网络丢包率变化，测试 Pipeline 能否：
1. 通过 RTCP RR 跟踪网络变化
2. 每帧重新优化 FEC 策略
3. 输出策略切换历史
"""

import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transport_fec_ml.pipeline import AdaptiveFECPipeline
from transport_fec_ml.adaptive_fec import AnalyticalOptimizer
from transport_fec_ml.rtcp import (
    RTCPReceiverReport, RTCPReportBlock, FECStatsReport,
)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(name)s | %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('test_pipeline')


def make_rr_and_stats(
    fraction_lost: float,
    jitter: int = 0,
    last_sr: int = 0,
    dlsr: int = 0,
    per_lod_data: list[tuple[int, int, int, int]] | None = None,
) -> tuple[RTCPReceiverReport, FECStatsReport | None]:
    """构造模拟 RTCP RR + FECStats"""
    rb = RTCPReportBlock(
        ssrc=1,
        fraction_lost=int(fraction_lost * 256),
        cumulative_lost=0,
        ext_highest_seq=0,
        interarrival_jitter=jitter,
        last_sr=last_sr,
        delay_since_last_sr=dlsr,
    )
    rr = RTCPReceiverReport(ssrc=2, report_blocks=[rb])

    fec_stats = None
    if per_lod_data:
        fec_stats = FECStatsReport(
            per_lod_total=[d[0] for d in per_lod_data],
            per_lod_lost=[d[1] for d in per_lod_data],
            per_lod_recovered=[d[2] for d in per_lod_data],
            per_lod_unrecovered=[d[3] for d in per_lod_data],
        )
    return rr, fec_stats


def main():
    print("=" * 70)
    print("  Experiment 4: AdaptiveFECPipeline Closed-Loop Test")
    print("=" * 70)

    # 真实 LOD 数据
    lod_sizes = [1, 64, 1064, 80284]
    lod_gaussians = [1, 356, 5967, 450621]

    # ---- Scenario 1: 简单闭环 ----
    print("\n" + "-" * 60)
    print("  Scenario 4.1: Simple Closed Loop")
    print("  Loss: 3% → 6% → 10% (3 frames, 3 RRs each)")
    print("-" * 60)

    pipeline = AdaptiveFECPipeline(
        alpha=0.3, bandwidth_budget=0.20,
        loss_change_threshold=0.005,
        min_interval_steps=1,
    )

    # Frame 1: 3% loss (3 RRs)
    for _ in range(3):
        rr, fs = make_rr_and_stats(0.03)
        pipeline.feed_simulated(0.03)
        pipeline.optimize(lod_sizes, lod_gaussians)

    alloc1 = pipeline.get_allocation()
    print(f"\n  After 3× p=0.03:")
    print(f"    State loss_rate: {pipeline.loss_rate:.4f}")
    for lod in sorted(alloc1.lod_configs):
        k, n = alloc1.lod_configs[lod]
        print(f"    LOD {lod}: RS({n},{k}) rec={alloc1.per_lod_recovery_rate[lod]:.4f}")

    # Frame 2: 6% loss (3 RRs)
    for _ in range(3):
        rr, fs = make_rr_and_stats(0.06)
        pipeline.feed_simulated(0.06)
        pipeline.optimize(lod_sizes, lod_gaussians)

    alloc2 = pipeline.get_allocation()
    print(f"\n  After 3× p=0.06:")
    print(f"    State loss_rate: {pipeline.loss_rate:.4f}")
    for lod in sorted(alloc2.lod_configs):
        k, n = alloc2.lod_configs[lod]
        print(f"    LOD {lod}: RS({n},{k}) rec={alloc2.per_lod_recovery_rate[lod]:.4f}")

    # Frame 3: 10% loss (3 RRs)
    for _ in range(3):
        rr, fs = make_rr_and_stats(0.10)
        pipeline.feed_simulated(0.10)
        pipeline.optimize(lod_sizes, lod_gaussians)

    alloc3 = pipeline.get_allocation()
    print(f"\n  After 3× p=0.10:")
    print(f"    State loss_rate: {pipeline.loss_rate:.4f}")
    for lod in sorted(alloc3.lod_configs):
        k, n = alloc3.lod_configs[lod]
        print(f"    LOD {lod}: RS({n},{k}) rec={alloc3.per_lod_recovery_rate[lod]:.4f}")

    # ---- Scenario 2: 策略切换历史 ----
    print("\n" + "-" * 60)
    print("  Scenario 4.2: Strategy History")
    print("-" * 60)

    print(f"\n  History ({len(pipeline.history)} optimizations):")
    print(pipeline.summary())

    # ---- Scenario 3: should_optimize 阈值 ----
    print("\n" + "-" * 60)
    print("  Scenario 4.3: Optimization Threshold")
    print("  threshold=0.01, min_interval=2 steps")
    print("-" * 60)

    pipeline2 = AdaptiveFECPipeline(
        alpha=0.3, bandwidth_budget=0.20,
        loss_change_threshold=0.01,
        min_interval_steps=2,
    )

    # 丢包率微小变化，不应触发重优化
    for p in [0.03, 0.031, 0.032, 0.06, 0.061]:
        pipeline2.feed_simulated(p)
        result = pipeline2.optimize(lod_sizes, lod_gaussians)
        status = "OPTIMIZED" if result else "skipped"
        print(f"    p={p:.3f}, loss_ema={pipeline2.loss_rate:.4f} → {status}")

    print(f"\n  Total optimizations: {len(pipeline2.history)} (expected: 2)")

    # ---- Scenario 4: Force optimize ----
    print("\n" + "-" * 60)
    print("  Scenario 4.4: Force Optimize")
    print("-" * 60)

    pipeline3 = AdaptiveFECPipeline(
        alpha=0.3, bandwidth_budget=0.20,
        loss_change_threshold=0.10,  # very high threshold
        min_interval_steps=100,      # very long interval
    )

    # Normal call should skip
    pipeline3.feed_simulated(0.05)
    r1 = pipeline3.optimize(lod_sizes, lod_gaussians)
    print(f"    Normal optimize: {'OPTIMIZED' if r1 else 'skipped'} (expected: OPTIMIZED, first call)")

    # Second call should skip (min_interval)
    pipeline3.feed_simulated(0.05)
    r2 = pipeline3.optimize(lod_sizes, lod_gaussians)
    print(f"    Second optimize: {'OPTIMIZED' if r2 else 'skipped'} (expected: skipped)")

    # Force should work
    r3 = pipeline3.optimize(lod_sizes, lod_gaussians, force=True)
    print(f"    Force optimize:  {'OPTIMIZED' if r3 else 'skipped'} (expected: OPTIMIZED)")

    # Reset
    pipeline3.reset()
    assert pipeline3.loss_rate == 0.0
    assert len(pipeline3.history) == 0
    print(f"    Reset: OK (history={len(pipeline3.history)}, loss={pipeline3.loss_rate})")

    print("\n" + "=" * 70)
    print("  Experiment 4 Complete.")
    print("=" * 70)


if __name__ == '__main__':
    main()
