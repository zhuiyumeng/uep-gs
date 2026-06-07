"""
自适应 FEC 策略模块

核心组件：
- AnalyticalOptimizer: 基于 RS(n,k) 块恢复概率模型的 DP 优化器
  给定 LOD 分布、丢包率、带宽预算，计算最优 FEC 分配
- AdaptiveFECPolicy: 策略对象，可注入 RTPEncoder/RTPDecoder 以覆盖硬编码 UEP_POLICY

优化模型说明：
- 无 FEC 时：每个数据包独立存活，期望恢复率 = (1-p)
- RS(n,k) 块 FEC：块恢复为 all-or-nothing，成功概率 = P(≤ n-k 个丢包 in n)
  实际恢复率 = P_recover （整个块的 k 个数据包要么全部恢复，要么全丢）
  注：reedsolo 的 erasure decoding 在已知丢包位置时最多纠正 n-k 个丢包，
  本模型使用 all-or-nothing 近似作为下界估计（保守）。
- 增量 benefit = w_i * g_i * delta / sqrt(parity + 1)
  其中 g_i = lod_gaussian_counts[i]（高斯数量），delta = P_recover_RS - (1-p)
  使用高斯数量而非包数避免 LOD 包数差异主导 benefit；
  除以 sqrt(parity+1) 引入边际收益递减，鼓励预算分散。
"""

import math
import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)


# ---- RS 候选参数 ----

CANDIDATE_K = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
# 生成所有合法 (k, n) 组合: n ∈ {k, k+1, ..., k+4}
# k=n 表示无 FEC; n>k 表示有 n-k 个校验包
# n 上限为 12，控制 RS 编码的计算开销
CANDIDATE_CONFIGS: list[tuple[int, int]] = []
for _k in CANDIDATE_K:
    for _extra in range(0, 5):  # 0~4 个校验包
        _n = _k + _extra
        if _n <= 12:
            CANDIDATE_CONFIGS.append((_k, _n))
CANDIDATE_CONFIGS = sorted(set(CANDIDATE_CONFIGS))


# ---- 数据结构 ----

@dataclass
class FECAllocation:
    """FEC 分配方案"""
    lod_configs: dict[int, tuple[int, int]]  # lod -> (k, n)
    total_overhead_packets: int
    total_overhead_ratio: float
    weighted_recovery_rate: float         # 加权期望恢复率
    per_lod_recovery_rate: dict[int, float]  # 每 LOD 期望恢复率
    per_lod_bandwidth_cost: dict[int, float]
    baseline_recovery_rate: float = 0.0   # 无 FEC 时的加权恢复率 (1-p * Σ w_i s_i / Σ w_i)

    def summary(self) -> str:
        lines = [
            f"FECAllocation(overhead={self.total_overhead_ratio:.2%}, "
            f"wrr={self.weighted_recovery_rate:.4f}, "
            f"baseline={(1 - self.baseline_recovery_rate):.1%} loss → "
            f"Δ={self.weighted_recovery_rate - (1 - self.baseline_recovery_rate):+.4f})",
        ]
        for lod in sorted(self.lod_configs):
            k, n = self.lod_configs[lod]
            r = self.per_lod_recovery_rate.get(lod, 0.0)
            bw = self.per_lod_bandwidth_cost.get(lod, 0.0)
            lines.append(
                f"  LOD {lod}: RS({n},{k}) recovery={r:.4f} bw={bw:.4f}"
            )
        return '\n'.join(lines)


# ---- 解析优化器 ----

class AnalyticalOptimizer:
    """基于 RS(n,k) 块恢复概率模型的解析优化器

    使用动态规划（0/1 背包变种）求最优 FEC 分配。

    关键 insight: 无 FEC 时每个包独立以 (1-p) 概率存活。
    RS 块编码将多个包绑定（all-or-nothing），只有当绑定后的
    恢复概率 > (1-p) 时才有净增益。

    benefit 公式使用高斯数量（而非数据包数）作为权重，
    避免因 LOD 间包数差异巨大（~10,000x）导致预算分配失衡。
    边际收益递减项 (1/sqrt(parity+1)) 鼓励预算分散到多个 LOD。
    """

    @staticmethod
    def no_fec_recovery_rate(p: float) -> float:
        """无 FEC 时每个数据包的独立存活概率"""
        return 1.0 - p

    @staticmethod
    def block_recovery_prob(k: int, n: int, p: float) -> float:
        """RS(n,k) 编码块的 all-or-nothing 恢复概率

        P_recover = Σ_{j=0}^{n-k} C(n,j) * p^j * (1-p)^{n-j}

        当前模型使用 all-or-nothing 近似：若丢包数 ≤ n-k，块全部恢复；
        若丢包数 > n-k，块全部丢失。
        这是 erasure decoding 的下界估计（保守），因为 reedsolo 在
        丢包位置已知时可纠正最多 n-k 个 erasure，且列级独立意味着
        部分列即使总丢包 > n-k 也可能恢复。

        Args:
            k: 数据包数 (data symbols)
            n: 总包数 (data + parity)
            p: 独立丢包率 [0, 1]
        """
        if p <= 0.0:
            return 1.0
        if p >= 1.0:
            return 0.0

        max_losses = n - k
        if max_losses <= 0:
            return 0.0  # k >= n with FEC path: shouldn't reach here

        prob = 0.0
        log_p = math.log(p)
        log_1mp = math.log(1 - p)
        for j in range(max_losses + 1):
            log_comb = (
                math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1)
            )
            log_term = log_comb + j * log_p + (n - j) * log_1mp
            prob += math.exp(log_term)
        return min(prob, 1.0)

    @staticmethod
    def expected_recovery(k: int, n: int, p: float) -> float:
        """RS(n,k) 编码下的期望数据恢复率

        - k == n (无 FEC): 每包独立 (1-p) 存活
        - k < n (有 FEC): 块 all-or-nothing，成功率 = P_recover_block
        """
        if k == n:
            return AnalyticalOptimizer.no_fec_recovery_rate(p)
        else:
            return AnalyticalOptimizer.block_recovery_prob(k, n, p)

    @staticmethod
    def optimize(
        lod_sizes: list[int],
        loss_rate: float,
        bandwidth_budget: float,
        lod_weights: list[float] | None = None,
        lod_gaussian_counts: list[int] | None = None,
    ) -> FECAllocation:
        """DP 求解最优 FEC 分配

        Args:
            lod_sizes: 每 LOD 的数据包数列表
            loss_rate: 网络丢包率 [0, 1]
            bandwidth_budget: 总带宽预算比例 (如 0.20 = 20%)
            lod_weights: 每 LOD 的重要性权重，默认 w_i = 1/(i+1)
            lod_gaussian_counts: 每 LOD 的高斯数量，用于 benefit 归一化。
                                 为 None 时回退到使用 lod_sizes（向后兼容）。

        Returns:
            FECAllocation: 最优分配方案
        """
        K = len(lod_sizes)
        if lod_weights is None:
            lod_weights = [1.0 / (i + 1) for i in range(K)]
        if lod_gaussian_counts is None:
            lod_gaussian_counts = lod_sizes  # 向后兼容

        total_data_packets = sum(lod_sizes)

        # Baseline: 无 FEC 时的 per-packet recovery rate
        baseline_rec = AnalyticalOptimizer.no_fec_recovery_rate(loss_rate)

        log.info(
            "AnalyticalOptimizer.optimize: K=%d LODs, loss_rate=%.3f, "
            "budget=%.2f%%, total_packets=%d, baseline_rec=%.4f",
            K, loss_rate, bandwidth_budget * 100, total_data_packets, baseline_rec,
        )
        for i in range(K):
            log.info(
                "  LOD %d: %d packets, %d gaussians, weight=%.4f",
                i, lod_sizes[i], lod_gaussian_counts[i], lod_weights[i],
            )

        # 0 预算或无丢包：全部无 FEC
        if bandwidth_budget <= 0.0 or loss_rate <= 0.0:
            log.info("  → trivial: budget=0 or no loss, all (8,8) no-FEC")
            configs = {i: (8, 8) for i in range(K)}
            rates = {i: 1.0 for i in range(K)}
            return FECAllocation(
                lod_configs=configs,
                total_overhead_packets=0,
                total_overhead_ratio=0.0,
                weighted_recovery_rate=1.0 if loss_rate <= 0.0 else baseline_rec,
                per_lod_recovery_rate=rates,
                per_lod_bandwidth_cost={i: 0.0 for i in range(K)},
                baseline_recovery_rate=loss_rate,
            )

        max_parity_budget = int(bandwidth_budget * total_data_packets)
        log.info("  max_parity_budget=%d packets", max_parity_budget)

        # 预计算每个 LOD 的候选 (k,n) 组合
        lod_candidates: list[list[dict]] = []
        for i in range(K):
            g_count = lod_gaussian_counts[i]
            candidates = []
            # 无 FEC 候选 (k=n): 所有 k 值等效，统一用 baseline_rec
            candidates.append({
                'k': 8, 'n': 8,
                'parity': 0,
                'recovery': baseline_rec,
                'benefit': 0.0,  # 增量 = 0
            })

            for k, n in CANDIDATE_CONFIGS:
                if k >= n:  # 跳过无 FEC (已在上面添加)
                    continue
                if k <= 0:
                    continue

                rec = AnalyticalOptimizer.block_recovery_prob(k, n, loss_rate)
                delta = rec - baseline_rec
                if delta <= 0:
                    # FEC 反而降低恢复率，跳过
                    continue

                num_blocks = (lod_sizes[i] + k - 1) // k
                parity = num_blocks * (n - k)

                if parity > max_parity_budget:
                    continue

                # benefit = w_i * gaussian_count * delta / sqrt(parity + 1)
                # 使用高斯数量消除 LOD 间包数差异（~10,000x）
                # 除以 sqrt(parity+1) 引入边际收益递减
                benefit = (lod_weights[i] * g_count * delta) / math.sqrt(parity + 1)
                candidates.append({
                    'k': k, 'n': n,
                    'parity': parity,
                    'recovery': rec,
                    'benefit': benefit,
                })

            log.info(
                "  LOD %d: %d candidates (gaussians=%d)",
                i, len(candidates), g_count,
            )
            lod_candidates.append(candidates)

        # DP: dp[i][p] = 前 i 个 LOD 用 p 个校验包的最大加权增量 benefit
        # 使用 list-of-dict 实现：index = parity_used, value = (benefit, choices)
        dp: list[dict[int, tuple[float, dict]]] = [{0: (0.0, {})}]

        for i in range(K):
            new_dp: dict[int, tuple[float, dict]] = {}
            for parity_used, (benefit, choices) in dp[i].items():
                for cand in lod_candidates[i]:
                    new_parity = parity_used + cand['parity']
                    if new_parity > max_parity_budget:
                        continue
                    new_benefit = benefit + cand['benefit']
                    key = new_parity
                    if key not in new_dp or new_benefit > new_dp[key][0]:
                        new_choices = dict(choices)
                        new_choices[i] = (
                            cand['k'], cand['n'],
                            cand['recovery'], cand['parity'],
                        )
                        new_dp[key] = (new_benefit, new_choices)
            dp.append(new_dp)

        # 找最优解
        best_benefit = 0.0
        best_parity = 0
        best_choices: dict = {}
        for parity, (benefit, choices) in dp[K].items():
            if benefit > best_benefit:
                best_benefit = benefit
                best_parity = parity
                best_choices = choices

        log.info(
            "  DP result: best_benefit=%.6f, best_parity=%d/%d (%.2f%%)",
            best_benefit, best_parity, max_parity_budget,
            best_parity / max(total_data_packets, 1) * 100,
        )

        # 构建返回值
        lod_configs: dict[int, tuple[int, int]] = {}
        per_lod_rec: dict[int, float] = {}
        per_lod_bw: dict[int, float] = {}
        total_weighted_size = sum(
            lod_weights[i] * lod_sizes[i] for i in range(K)
        )

        for i in range(K):
            if i in best_choices:
                k, n, rec, parity = best_choices[i]
                lod_configs[i] = (k, n)
                per_lod_rec[i] = rec
                per_lod_bw[i] = parity / max(total_data_packets, 1)
                log.info(
                    "  LOD %d: RS(%d,%d) recovery=%.4f parity=%d (vs baseline %.4f)",
                    i, n, k, rec, parity, baseline_rec,
                )
            else:
                lod_configs[i] = (8, 8)
                per_lod_rec[i] = baseline_rec
                per_lod_bw[i] = 0.0
                log.info(
                    "  LOD %d: (8,8) no-FEC recovery=%.4f (baseline)", i, baseline_rec,
                )

        # weighted_recovery_rate = weighted average over LODs
        wrr = sum(
            lod_weights[i] * lod_sizes[i] * per_lod_rec[i]
            for i in range(K)
        ) / max(total_weighted_size, 1)

        log.info(
            "  Final: overhead=%.2f%%, wrr=%.4f (baseline=%.4f, Δ=%+.4f)",
            best_parity / max(total_data_packets, 1) * 100,
            wrr, baseline_rec, wrr - baseline_rec,
        )

        return FECAllocation(
            lod_configs=lod_configs,
            total_overhead_packets=best_parity,
            total_overhead_ratio=best_parity / max(total_data_packets, 1),
            weighted_recovery_rate=wrr,
            per_lod_recovery_rate=per_lod_rec,
            per_lod_bandwidth_cost=per_lod_bw,
            baseline_recovery_rate=loss_rate,
        )


# ---- 策略对象 ----

class AdaptiveFECPolicy:
    """自适应 FEC 策略对象

    可注入 RTPEncoder / RTPDecoder 的 fec_policy 参数，
    替代硬编码的 UEP_POLICY。

    Usage:
        allocation = AnalyticalOptimizer.optimize(lod_sizes, loss_rate, budget)
        policy = AdaptiveFECPolicy(allocation)
        encoder = RTPEncoder(fec_policy=policy)
        decoder = RTPDecoder(fec_policy=policy)
    """

    def __init__(self, allocation: FECAllocation):
        self.allocation = allocation

    def get_config(self, lod: int) -> tuple[int, int]:
        """返回指定 LOD 的 (k, n) 配置"""
        if lod not in self.allocation.lod_configs:
            log.warning(
                "AdaptiveFECPolicy: LOD %d not in allocation, "
                "falling back to (8,8) no-FEC", lod,
            )
            return (8, 8)
        return self.allocation.lod_configs[lod]

    @property
    def configs(self) -> dict[int, tuple[int, int]]:
        return self.allocation.lod_configs

    def summary(self) -> str:
        return self.allocation.summary()
