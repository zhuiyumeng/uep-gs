#!/usr/bin/env python
"""
experiment01/test_optimizer.py
实验 1: DP 优化器正确性验证

测试场景：
  A: 低丢包 (3%, 预算 20%) — 验证 LOD0 得到强保护
  B: 中丢包 (6%, 预算 20%) — 验证多 LOD 分配
  C: 高丢包 (10%, 预算 30%) — 验证所有 LOD 都获 FEC
  D: 紧预算 (6%, 预算 5%)  — 验证预算集中给高重要性 LOD

每场景同时运行旧公式和新公式，对比分配结果。
"""

import sys
import os
import math
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transport_fec_ml.adaptive_fec import (
    AnalyticalOptimizer, FECAllocation, CANDIDATE_K, CANDIDATE_CONFIGS,
)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(name)s | %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('test_optimizer')

# ============================================================
# 辅助：模拟旧公式（benefit 无归一化，CANDIDATE_K 有缺口）
# ============================================================

OLD_CANDIDATE_K = [1, 2, 4, 5, 8, 10]
OLD_CONFIGS = []
for _k in OLD_CANDIDATE_K:
    for _extra in range(0, 5):
        _n = _k + _extra
        if _n <= 12:
            OLD_CONFIGS.append((_k, _n))
OLD_CONFIGS = sorted(set(OLD_CONFIGS))


class OldOptimizer(AnalyticalOptimizer):
    """旧版本优化器（benefit 使用 lod_sizes，CANDIDATE_K 有缺口）"""

    @staticmethod
    def optimize_old(
        lod_sizes, loss_rate, bandwidth_budget,
        lod_weights=None,
    ) -> FECAllocation:
        K = len(lod_sizes)
        if lod_weights is None:
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
            candidates = []
            candidates.append({
                'k': 8, 'n': 8, 'parity': 0,
                'recovery': baseline_rec, 'benefit': 0.0,
            })
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
                # 旧公式：benefit = w_i * lod_sizes[i] * delta
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
                    key = new_parity
                    if key not in new_dp or new_benefit > new_dp[key][0]:
                        new_choices = dict(choices)
                        new_choices[i] = (cand['k'], cand['n'],
                                          cand['recovery'], cand['parity'])
                        new_dp[key] = (new_benefit, new_choices)
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
        total_weighted_size = sum(
            lod_weights[i] * lod_sizes[i] for i in range(K)
        )
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
            lod_configs=lod_configs,
            total_overhead_packets=best_parity,
            total_overhead_ratio=best_parity / max(total_data_packets, 1),
            weighted_recovery_rate=wrr,
            per_lod_recovery_rate=per_lod_rec,
            per_lod_bandwidth_cost=per_lod_bw,
            baseline_recovery_rate=loss_rate,
        )


# ============================================================
# 测试数据
# ============================================================

# 模拟真实场景: 4 LOD, 高斯分布类似 K-means 结果但更平衡
# 包数 = gaussian_count * 248 / 1392 ≈ gaussian_count / 5.6
# 但为了测试 benefit 公式效果, 用一个平衡的合成数据

# 合成场景：4 LOD，高斯数递增但包数差异大
SCENARIOS = {
    'A: 低丢包 3%, 预算 20%': {
        'lod_sizes': [5, 100, 1500, 80000],        # 包数
        'lod_gaussian_counts': [500, 5000, 30000, 420000],  # 高斯数
        'loss_rate': 0.03,
        'bandwidth_budget': 0.20,
    },
    'B: 中丢包 6%, 预算 20%': {
        'lod_sizes': [5, 100, 1500, 80000],
        'lod_gaussian_counts': [500, 5000, 30000, 420000],
        'loss_rate': 0.06,
        'bandwidth_budget': 0.20,
    },
    'C: 高丢包 10%, 预算 30%': {
        'lod_sizes': [5, 100, 1500, 80000],
        'lod_gaussian_counts': [500, 5000, 30000, 420000],
        'loss_rate': 0.10,
        'bandwidth_budget': 0.30,
    },
    'D: 紧预算 6%, 预算 5%': {
        'lod_sizes': [5, 100, 1500, 80000],
        'lod_gaussian_counts': [500, 5000, 30000, 420000],
        'loss_rate': 0.06,
        'bandwidth_budget': 0.05,
    },
}

# 真实 LOD 数据（从 experiments/multi_loss/lod/ 读取）
REAL_LOD_SIZES = [1, 64, 1064, 80284]        # 包数
REAL_LOD_GAUSSIANS = [1, 356, 5967, 450621]  # 高斯数

SCENARIOS_REAL = {
    'R: 真实数据 低丢包 3%': {
        'lod_sizes': REAL_LOD_SIZES,
        'lod_gaussian_counts': REAL_LOD_GAUSSIANS,
        'loss_rate': 0.03,
        'bandwidth_budget': 0.20,
    },
    'R: 真实数据 中丢包 6%': {
        'lod_sizes': REAL_LOD_SIZES,
        'lod_gaussian_counts': REAL_LOD_GAUSSIANS,
        'loss_rate': 0.06,
        'bandwidth_budget': 0.20,
    },
    'R: 真实数据 高丢包 10%': {
        'lod_sizes': REAL_LOD_SIZES,
        'lod_gaussian_counts': REAL_LOD_GAUSSIANS,
        'loss_rate': 0.10,
        'bandwidth_budget': 0.30,
    },
}


# ============================================================
# 对比函数
# ============================================================

def compare_allocation(name, old: FECAllocation, new: FECAllocation):
    """打印新旧分配对比"""
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")

    print(f"\n  {'':>6s}  {'Old Formula':>45s}  {'New Formula':>45s}")
    print(f"  {'LOD':>6s}  {'Config':>12s}  {'Recovery':>10s}  {'BW':>8s}  "
          f"{'Config':>12s}  {'Recovery':>10s}  {'BW':>8s}")
    print(f"  {'-'*6}  {'-'*12}  {'-'*10}  {'-'*8}  "
          f"{'-'*12}  {'-'*10}  {'-'*8}")

    K = max(
        len(old.lod_configs) if old else 0,
        len(new.lod_configs) if new else 0,
    )
    for i in range(K):
        ok, on = old.lod_configs.get(i, (8, 8))
        nk, nn = new.lod_configs.get(i, (8, 8))
        orr = old.per_lod_recovery_rate.get(i, 0)
        nrr = new.per_lod_recovery_rate.get(i, 0)
        obw = old.per_lod_bandwidth_cost.get(i, 0)
        nbw = new.per_lod_bandwidth_cost.get(i, 0)

        # 标记变化
        flag_old = ""
        flag_new = ""
        if ok != on:  # k != n means FEC allocated
            flag_old = " ← FEC"
        if nk != nn:
            flag_new = " ← FEC"

        print(f"  {i:6d}  RS({on:2d},{ok:2d}){flag_old:>6s}  "
              f"{orr:10.4f}  {obw:8.4f}  "
              f"RS({nn:2d},{nk:2d}){flag_new:>6s}  "
              f"{nrr:10.4f}  {nbw:8.4f}")

    print(f"\n  Summary:")
    print(f"    Old: overhead={old.total_overhead_ratio:.2%}, "
          f"wrr={old.weighted_recovery_rate:.4f}")
    print(f"    New: overhead={new.total_overhead_ratio:.2%}, "
          f"wrr={new.weighted_recovery_rate:.4f}")

    # LOD0 是否得到保护（关键验证）
    old_lod0_fec = old.lod_configs.get(0, (8, 8))
    new_lod0_fec = new.lod_configs.get(0, (8, 8))
    old_lod0_has_fec = old_lod0_fec[0] != old_lod0_fec[1]
    new_lod0_has_fec = new_lod0_fec[0] != new_lod0_fec[1]

    baseline_rec = AnalyticalOptimizer.no_fec_recovery_rate(
        old.baseline_recovery_rate
    )
    old_lod0_rec = old.per_lod_recovery_rate.get(0, 0)
    new_lod0_rec = new.per_lod_recovery_rate.get(0, 0)

    print(f"\n  🔑 Key Check (LOD0):")
    print(f"    Baseline recovery (no-FEC): {baseline_rec:.4f}")
    print(f"    Old LOD0 recovery:          {old_lod0_rec:.4f} "
          f"{'✅ > baseline' if old_lod0_rec > baseline_rec + 0.001 else '❌ = baseline (BUG: LOD0 ignored!)'}")
    print(f"    New LOD0 recovery:          {new_lod0_rec:.4f} "
          f"{'✅ > baseline' if new_lod0_rec > baseline_rec + 0.001 else '❌ = baseline'}")


# ============================================================
# 主流程
# ============================================================

def main():
    print("=" * 70)
    print("  Experiment 1: DP Optimizer Correctness Test")
    print("  CANDIDATE_K old:", OLD_CANDIDATE_K)
    print("  CANDIDATE_K new:", CANDIDATE_K)
    print("  Candidate configs old:", len(OLD_CONFIGS))
    print("  Candidate configs new:", len(CANDIDATE_CONFIGS))
    print("=" * 70)

    # --- Part 1: 合成数据测试 ---
    print("\n\n" + "#" * 70)
    print("#  Part 1: Synthetic Data (Balanced LOD Distribution)")
    print("#" * 70)

    for name, params in SCENARIOS.items():
        lod_sizes = params['lod_sizes']
        g_counts = params['lod_gaussian_counts']
        loss = params['loss_rate']
        budget = params['bandwidth_budget']

        print(f"\n  Input: loss_rate={loss:.0%}, budget={budget:.0%}")
        print(f"  LOD sizes (packets): {lod_sizes}")
        print(f"  LOD gaussian counts: {g_counts}")

        old_alloc = OldOptimizer.optimize_old(lod_sizes, loss, budget)
        new_alloc = AnalyticalOptimizer.optimize(
            lod_sizes, loss, budget, lod_gaussian_counts=g_counts,
        )

        compare_allocation(name, old_alloc, new_alloc)

    # --- Part 2: 真实数据测试 ---
    print("\n\n" + "#" * 70)
    print("#  Part 2: Real LOD Data (experiments/multi_loss/lod/)")
    print(f"#  LOD sizes (packets): {REAL_LOD_SIZES}")
    print(f"#  LOD gaussians:       {REAL_LOD_GAUSSIANS}")
    print("#" * 70)

    for name, params in SCENARIOS_REAL.items():
        lod_sizes = params['lod_sizes']
        g_counts = params['lod_gaussian_counts']
        loss = params['loss_rate']
        budget = params['bandwidth_budget']

        print(f"\n  Input: loss_rate={loss:.0%}, budget={budget:.0%}")

        old_alloc = OldOptimizer.optimize_old(lod_sizes, loss, budget)
        new_alloc = AnalyticalOptimizer.optimize(
            lod_sizes, loss, budget, lod_gaussian_counts=g_counts,
        )

        compare_allocation(name, old_alloc, new_alloc)

    print("\n\n" + "=" * 70)
    print("  Experiment 1 Complete.")
    print("=" * 70)


if __name__ == '__main__':
    main()
