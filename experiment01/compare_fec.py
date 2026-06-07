#!/usr/bin/env python
"""
experiment01/compare_fec.py
实验 3: 离线策略对比

使用 FECSimulator 模拟真实 LOD 数据在 3 种策略下的表现：
1. 硬编码 UEP_POLICY（当前生产策略）
2. 旧 DP 优化器（benefit 缺陷）
3. 新 DP 优化器（修复后）

丢包率: 3%, 6%, 10%
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

# 硬编码 UEP_POLICY
UEP_POLICY_CONFIGS = {
    0: (1, 2),   # RS(2,1), +100%
    1: (8, 10),  # RS(10,8), +25%
    2: (8, 9),   # RS(9,8), +12.5%
    3: (8, 9),   # RS(9,8), +12.5%
}

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('compare_fec')


def make_uep_allocation(lod_sizes, loss_rate) -> FECAllocation:
    """从硬编码 UEP_POLICY 构造 FECAllocation"""
    total_packets = sum(lod_sizes)
    baseline_rec = AnalyticalOptimizer.no_fec_recovery_rate(loss_rate)
    per_lod_rec = {}
    per_lod_bw = {}
    total_parity = 0

    for lod in range(len(lod_sizes)):
        k, n = UEP_POLICY_CONFIGS.get(lod, (8, 8))
        num_blocks = (lod_sizes[lod] + k - 1) // k if k > 0 else 0
        parity = num_blocks * (n - k) if n > k else 0
        total_parity += parity
        per_lod_bw[lod] = parity / max(total_packets, 1)
        if k != n:
            per_lod_rec[lod] = AnalyticalOptimizer.block_recovery_prob(
                k, n, loss_rate,
            )
        else:
            per_lod_rec[lod] = baseline_rec

    total_weighted_size = sum(
        (1.0 / (i + 1)) * lod_sizes[i]
        for i in range(len(lod_sizes))
    )
    wrr = sum(
        (1.0 / (i + 1)) * lod_sizes[i] * per_lod_rec[i]
        for i in range(len(lod_sizes))
    ) / max(total_weighted_size, 1)

    return FECAllocation(
        lod_configs=dict(UEP_POLICY_CONFIGS),
        total_overhead_packets=total_parity,
        total_overhead_ratio=total_parity / max(total_packets, 1),
        weighted_recovery_rate=wrr,
        per_lod_recovery_rate=per_lod_rec,
        per_lod_bandwidth_cost=per_lod_bw,
        baseline_recovery_rate=loss_rate,
    )


# 模拟旧公式（与 test_optimizer.py 中的 OldOptimizer 相同）
OLD_CANDIDATE_K = [1, 2, 4, 5, 8, 10]
OLD_CONFIGS = []
for _k in OLD_CANDIDATE_K:
    for _extra in range(0, 5):
        _n = _k + _extra
        if _n <= 12:
            OLD_CONFIGS.append((_k, _n))
OLD_CONFIGS = sorted(set(OLD_CONFIGS))


def old_optimize(lod_sizes, loss_rate, bandwidth_budget) -> FECAllocation:
    """旧版优化器（有 benefit bug）"""
    K = len(lod_sizes)
    lod_weights = [1.0 / (i + 1) for i in range(K)]
    total_data_packets = sum(lod_sizes)
    baseline_rec = AnalyticalOptimizer.no_fec_recovery_rate(loss_rate)

    if bandwidth_budget <= 0.0 or loss_rate <= 0.0:
        configs = {i: (8, 8) for i in range(K)}
        rates = {i: 1.0 for i in range(K)}
        return FECAllocation(
            lod_configs=configs, total_overhead_packets=0,
            total_overhead_ratio=0.0,
            weighted_recovery_rate=1.0 if loss_rate <= 0.0 else baseline_rec,
            per_lod_recovery_rate=rates,
            per_lod_bandwidth_cost={i: 0.0 for i in range(K)},
            baseline_recovery_rate=loss_rate,
        )

    max_parity_budget = int(bandwidth_budget * total_data_packets)

    lod_candidates = []
    for i in range(K):
        candidates = [{
            'k': 8, 'n': 8, 'parity': 0,
            'recovery': baseline_rec, 'benefit': 0.0,
        }]
        for k, n in OLD_CONFIGS:
            if k >= n or k <= 0:
                continue
            rec = AnalyticalOptimizer.block_recovery_prob(k, n, loss_rate)
            delta = rec - baseline_rec
            if delta <= 0:
                continue
            num_blocks = (lod_sizes[i] + k - 1) // k
            parity = num_blocks * (n - k)
            if parity > max_parity_budget:
                continue
            benefit = lod_weights[i] * lod_sizes[i] * delta
            candidates.append({
                'k': k, 'n': n, 'parity': parity,
                'recovery': rec, 'benefit': benefit,
            })
        lod_candidates.append(candidates)

    dp = [{0: (0.0, {})}]
    for i in range(K):
        new_dp = {}
        for parity_used, (benefit, choices) in dp[i].items():
            for cand in lod_candidates[i]:
                new_parity = parity_used + cand['parity']
                if new_parity > max_parity_budget:
                    continue
                new_benefit = benefit + cand['benefit']
                if new_parity not in new_dp or new_benefit > new_dp[new_parity][0]:
                    nc = dict(choices)
                    nc[i] = (cand['k'], cand['n'], cand['recovery'], cand['parity'])
                    new_dp[new_parity] = (new_benefit, nc)
        dp.append(new_dp)

    best_benefit = 0.0
    best_parity = 0
    best_choices = {}
    for parity, (benefit, choices) in dp[K].items():
        if benefit > best_benefit:
            best_benefit = benefit
            best_parity = parity
            best_choices = choices

    lod_configs = {}
    per_lod_rec = {}
    per_lod_bw = {}
    total_weighted_size = sum(lod_weights[i] * lod_sizes[i] for i in range(K))
    for i in range(K):
        if i in best_choices:
            k, n, rec, parity = best_choices[i]
            lod_configs[i] = (k, n)
            per_lod_rec[i] = rec
            per_lod_bw[i] = parity / max(total_data_packets, 1)
        else:
            lod_configs[i] = (8, 8)
            per_lod_rec[i] = baseline_rec
            per_lod_bw[i] = 0.0

    wrr = sum(
        lod_weights[i] * lod_sizes[i] * per_lod_rec[i]
        for i in range(K)
    ) / max(total_weighted_size, 1)

    return FECAllocation(
        lod_configs=lod_configs, total_overhead_packets=best_parity,
        total_overhead_ratio=best_parity / max(total_data_packets, 1),
        weighted_recovery_rate=wrr,
        per_lod_recovery_rate=per_lod_rec,
        per_lod_bandwidth_cost=per_lod_bw,
        baseline_recovery_rate=loss_rate,
    )


def main():
    print("=" * 70)
    print("  Experiment 3: Offline FEC Strategy Comparison")
    print("=" * 70)

    # 真实 LOD 数据
    lod_sizes = [1, 64, 1064, 80284]
    lod_gaussians = [1, 356, 5967, 450621]
    loss_rates = [0.03, 0.06, 0.10]

    print(f"\n  LOD Data:")
    print(f"  {'LOD':>5s}  {'Packets':>10s}  {'Gaussians':>12s}  {'% Gauss':>10s}")
    total_g = sum(lod_gaussians)
    for i in range(len(lod_sizes)):
        print(f"  {i:5d}  {lod_sizes[i]:10d}  {lod_gaussians[i]:12d}  "
              f"{lod_gaussians[i]/total_g*100:9.2f}%")

    # 生成 3 种策略
    strategies = {}

    for lr in loss_rates:
        # 1. UEP_POLICY
        uep = make_uep_allocation(lod_sizes, lr)
        strategies[f"UEP_POLICY (loss={lr:.0%})"] = uep

        # 2. Old DP
        old = old_optimize(lod_sizes, lr, 0.20)
        strategies[f"Old DP (loss={lr:.0%})"] = old

        # 3. New DP
        new = AnalyticalOptimizer.optimize(
            lod_sizes, lr, bandwidth_budget=0.20,
            lod_gaussian_counts=lod_gaussians,
        )
        strategies[f"New DP (loss={lr:.0%})"] = new

    # 打印各策略配置
    print(f"\n  Strategy Configurations:")
    print(f"  {'Strategy':>30s}  {'Configs':>50s}  {'Overhead':>10s}  {'WRR':>8s}")
    print(f"  {'-'*30}  {'-'*50}  {'-'*10}  {'-'*8}")
    for name, alloc in strategies.items():
        config_str = ", ".join(
            f"L{i}:RS({n},{k})" if k != n else f"L{i}:(8,8)"
            for i, (k, n) in sorted(alloc.lod_configs.items())
        )
        print(f"  {name:>30s}  {config_str:>50s}  "
              f"{alloc.total_overhead_ratio:10.2%}  {alloc.weighted_recovery_rate:8.4f}")

    # FECSimulator 模拟
    print(f"\n  Running FECSimulator (100 trials per strategy×loss_rate)...")

    sim = FECSimulator(seed=42)

    # 按丢包率分组对比
    for lr in loss_rates:
        print(f"\n  {'='*60}")
        print(f"  Loss Rate: {lr:.0%}")
        print(f"  {'='*60}")

        lr_strategies = {
            f"UEP_POLICY": strategies[f"UEP_POLICY (loss={lr:.0%})"],
            f"Old DP": strategies[f"Old DP (loss={lr:.0%})"],
            f"New DP": strategies[f"New DP (loss={lr:.0%})"],
        }

        comparison = sim.compare_strategies(
            lod_sizes, lod_gaussians, [lr], lr_strategies, num_trials=100,
        )

        print(f"\n  {'Strategy':>15s}  {'Pkt Rec':>10s}  {'Gauss Rec':>12s}  "
              f"{'Eff Gauss':>12s}  {'Δ vs UEP':>12s}")
        print(f"  {'-'*15}  {'-'*10}  {'-'*12}  {'-'*12}  {'-'*12}")

        uep_result = comparison.results.get("UEP_POLICY", {}).get(lr)
        uep_gauss_rec = uep_result.gaussian_recovery_rate if uep_result else 0

        for name in comparison.strategy_names:
            r = comparison.results.get(name, {}).get(lr)
            if r:
                delta = r.gaussian_recovery_rate - uep_gauss_rec
                print(f"  {name:>15s}  {r.overall_recovery_rate:10.4f}  "
                      f"{r.gaussian_recovery_rate:12.4f}  "
                      f"{r.effective_gaussians:12d}  {delta:+.4f}")

    # 保存结果
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)

    output = {}
    for name, alloc in strategies.items():
        output[name] = {
            'lod_configs': {str(k): list(v) for k, v in alloc.lod_configs.items()},
            'overhead_ratio': alloc.total_overhead_ratio,
            'weighted_recovery_rate': alloc.weighted_recovery_rate,
            'per_lod_recovery_rate': alloc.per_lod_recovery_rate,
        }

    out_path = os.path.join(results_dir, 'allocation_comparison.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {out_path}")

    print("\n" + "=" * 70)
    print("  Experiment 3 Complete.")
    print("=" * 70)


if __name__ == '__main__':
    main()
