#!/usr/bin/env python
"""
experiment01/run_fec_ml_bandwidth.py
transport_fec_ml 带宽开销实验

在三组丢包率 (3%, 6%, 10%) 下运行 AnalyticalOptimizer，
记录 FEC 策略分配的额外带宽开销，并用 FECSimulator 验证恢复效果。

用法:
    python experiment01/run_fec_ml_bandwidth.py
"""

import sys
import os
import json
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transport_fec_ml.adaptive_fec import (
    AnalyticalOptimizer, AdaptiveFECPolicy, FECAllocation,
)
from transport_fec_ml.simulator import FECSimulator
from transport_fec_ml.pipeline import AdaptiveFECPipeline

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%H:%M:%S',
)

# ── 真实 LOD 数据 ──────────────────────────────────────────────
LOD_SIZES        = [1, 64, 1064, 80284]   # 每 LOD 数据包数
LOD_GAUSSIANS    = [1, 356, 5967, 450621]  # 每 LOD 高斯数
LOSS_RATES       = [0.03, 0.06, 0.10]
BANDWIDTH_BUDGET = 0.20                     # 统一 20% 预算

# 硬编码 UEP_POLICY（当前生产策略）
UEP_POLICY = {
    0: (1, 2),    # RS(2,1), +100%
    1: (8, 10),   # RS(10,8), +25%
    2: (8, 9),    # RS(9,8),  +12.5%
    3: (8, 9),    # RS(9,8),  +12.5%
}


def make_uep_allocation(loss_rate: float) -> FECAllocation:
    """从硬编码 UEP_POLICY 构造 FECAllocation"""
    total_packets = sum(LOD_SIZES)
    baseline_rec = AnalyticalOptimizer.no_fec_recovery_rate(loss_rate)
    lod_weights = [1.0 / (i + 1) for i in range(len(LOD_SIZES))]
    per_lod_rec = {}
    per_lod_bw = {}
    total_parity = 0

    for lod in range(len(LOD_SIZES)):
        k, n = UEP_POLICY.get(lod, (8, 8))
        num_blocks = (LOD_SIZES[lod] + k - 1) // k if k > 0 else 0
        parity = num_blocks * (n - k) if n > k else 0
        total_parity += parity
        per_lod_bw[lod] = parity / max(total_packets, 1)
        per_lod_rec[lod] = (
            AnalyticalOptimizer.block_recovery_prob(k, n, loss_rate)
            if k != n else baseline_rec
        )

    total_weighted_size = sum(
        lod_weights[i] * LOD_SIZES[i] for i in range(len(LOD_SIZES))
    )
    wrr = sum(
        lod_weights[i] * LOD_SIZES[i] * per_lod_rec[i]
        for i in range(len(LOD_SIZES))
    ) / max(total_weighted_size, 1)

    return FECAllocation(
        lod_configs=dict(UEP_POLICY),
        total_overhead_packets=total_parity,
        total_overhead_ratio=total_parity / max(total_packets, 1),
        weighted_recovery_rate=wrr,
        per_lod_recovery_rate=per_lod_rec,
        per_lod_bandwidth_cost=per_lod_bw,
        baseline_recovery_rate=loss_rate,
    )


def fmt_pct(v: float) -> str:
    return f"{v * 100:6.2f}%"


def fmt_lod_config(k: int, n: int) -> str:
    return f"RS({n:2d},{k:2d})" if k != n else "  (no-FEC)"


def main():
    print()
    print("=" * 76)
    print("  transport_fec_ml — FEC 带宽开销实验")
    print("=" * 76)
    print(f"  LOD 数据:  {len(LOD_SIZES)} LODs, {sum(LOD_GAUSSIANS):,} gaussians, "
          f"{sum(LOD_SIZES):,} packets")
    print(f"  丢包率:    {[f'{r:.0%}' for r in LOSS_RATES]}")
    print(f"  带宽预算:  {BANDWIDTH_BUDGET:.0%}")
    print()

    sim = FECSimulator(seed=42)
    results = []

    for lr in LOSS_RATES:
        print("-" * 76)
        print(f"  ▸ 丢包率 = {lr:.0%}")
        print("-" * 76)

        # ── 1. AnalyticalOptimizer DP 求解 ──
        dp_alloc = AnalyticalOptimizer.optimize(
            LOD_SIZES, lr, bandwidth_budget=BANDWIDTH_BUDGET,
            lod_gaussian_counts=LOD_GAUSSIANS,
        )
        # ── 2. 硬编码 UEP_POLICY ──
        uep_alloc = make_uep_allocation(lr)

        # ── 3. FECSimulator 模拟验证 ──
        dp_result = sim.simulate_multi(
            LOD_SIZES, LOD_GAUSSIANS, lr, dp_alloc, num_trials=100,
        )
        uep_result = sim.simulate_multi(
            LOD_SIZES, LOD_GAUSSIANS, lr, uep_alloc, num_trials=100,
        )

        # ── 4. 原始数据量 ──
        original_bytes = sum(LOD_GAUSSIANS) * 248   # stride=248 bytes per gaussian
        original_mb = original_bytes / 1024 / 1024

        # ── 5. 打印结果 ──
        print()
        print(f"  {'策略':<14s}  {'额外带宽':>10s}  {'发送量(MB)':>12s}  "
              f"{'WRR':>8s}  {'模拟恢复率':>10s}  {'各 LOD 配置'}")
        print(f"  {'─'*14}  {'─'*10}  {'─'*12}  {'─'*8}  {'─'*10}  {'─'*30}")

        for label, alloc, sim_result in [
            ("UEP_POLICY",   uep_alloc, uep_result),
            ("DP Optimizer", dp_alloc,  dp_result),
        ]:
            total_sent_mb = original_mb * (1 + alloc.total_overhead_ratio)
            config_str = " | ".join(
                f"L{i}:{fmt_lod_config(k,n)}"
                for i, (k, n) in sorted(alloc.lod_configs.items())
            )

            print(f"  {label:<14s}  {fmt_pct(alloc.total_overhead_ratio):>10s}  "
                  f"{total_sent_mb:>10.1f} MB  {alloc.weighted_recovery_rate:>8.4f}  "
                  f"{sim_result.gaussian_recovery_rate:>10.4f}  {config_str}")

        print()
        print(f"  📦 原始数据: {original_mb:.1f} MB ({original_bytes:,} bytes)")
        print(f"  📊 DP 策略额外发送: "
              f"{original_mb * dp_alloc.total_overhead_ratio:.1f} MB "
              f"({fmt_pct(dp_alloc.total_overhead_ratio)})")
        print(f"  📊 UEP 策略额外发送: "
              f"{original_mb * uep_alloc.total_overhead_ratio:.1f} MB "
              f"({fmt_pct(uep_alloc.total_overhead_ratio)})")
        print(f"  🔄 DP 恢复率 vs 无 FEC ({(1-lr)*100:.0f}%): "
              f"Δ={dp_result.gaussian_recovery_rate - (1-lr):+.4f}")
        print(f"  🔄 UEP 恢复率 vs 无 FEC ({(1-lr)*100:.0f}%): "
              f"Δ={uep_result.gaussian_recovery_rate - (1-lr):+.4f}")

        results.append({
            'loss_rate': lr,
            'dp_allocation': {
                'lod_configs': {str(k): list(v) for k, v in dp_alloc.lod_configs.items()},
                'overhead_ratio': dp_alloc.total_overhead_ratio,
                'wrr': dp_alloc.weighted_recovery_rate,
            },
            'uep_allocation': {
                'overhead_ratio': uep_alloc.total_overhead_ratio,
                'wrr': uep_alloc.weighted_recovery_rate,
            },
            'dp_simulation': {
                'gaussian_recovery_rate': dp_result.gaussian_recovery_rate,
                'effective_gaussians': dp_result.effective_gaussians,
            },
            'uep_simulation': {
                'gaussian_recovery_rate': uep_result.gaussian_recovery_rate,
                'effective_gaussians': uep_result.effective_gaussians,
            },
        })

    # ── 汇总表 ──
    print()
    print("=" * 76)
    print("  汇总对比")
    print("=" * 76)
    print()
    print(f"  {'丢包率':>8s}  {'策略':<14s}  {'额外带宽':>10s}  "
          f"{'WRR':>8s}  {'高斯恢复率':>10s}  {'有效高斯数':>12s}")
    print(f"  {'─'*8}  {'─'*14}  {'─'*10}  {'─'*8}  {'─'*10}  {'─'*12}")

    for r in results:
        print(f"  {r['loss_rate']:8.0%}  {'DP Optimizer':<14s}  "
              f"{fmt_pct(r['dp_allocation']['overhead_ratio']):>10s}  "
              f"{r['dp_allocation']['wrr']:>8.4f}  "
              f"{r['dp_simulation']['gaussian_recovery_rate']:>10.4f}  "
              f"{r['dp_simulation']['effective_gaussians']:>12,}")
        print(f"  {'':>8s}  {'UEP_POLICY':<14s}  "
              f"{fmt_pct(r['uep_allocation']['overhead_ratio']):>10s}  "
              f"{r['uep_allocation']['wrr']:>8.4f}  "
              f"{r['uep_simulation']['gaussian_recovery_rate']:>10.4f}  "
              f"{r['uep_simulation']['effective_gaussians']:>12,}")
        print()

    # 保存结果
    out_path = os.path.join(
        os.path.dirname(__file__), 'results', 'fec_ml_bandwidth.json',
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  结果已保存: {out_path}")
    print()


if __name__ == '__main__':
    main()
