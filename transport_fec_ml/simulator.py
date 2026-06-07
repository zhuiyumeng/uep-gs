"""
离线 FEC 模拟器 (FECSimulator)

在不启动真实 UDP 传输的情况下，模拟丢包 → FEC 恢复的完整流程。
用于快速评估和对比不同 FEC 策略在多种丢包率下的表现。

模拟流程：
1. 输入：LOD 分布（包数/高斯数）、丢包率、FEC 策略
2. 模拟丢包：对每个 RS 块，独立丢包（p 概率），记录丢失的数据包
3. 模拟恢复：RS(n,k) 在丢包数 ≤ n-k 时全部恢复，否则全丢（保守估计）
4. 输出：per-LOD 恢复率、有效包数/高斯数

不依赖 reedsolo — 使用统计模型近似（与 AnalyticalOptimizer 一致）。
"""

import math
import random
import logging
from dataclasses import dataclass, field

from .adaptive_fec import (
    AnalyticalOptimizer,
    AdaptiveFECPolicy,
    FECAllocation,
    CANDIDATE_CONFIGS,
)

log = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """单次模拟的结果"""
    loss_rate: float
    # Per-LOD 统计
    per_lod_total_packets: list[int] = field(default_factory=list)
    per_lod_lost_packets: list[int] = field(default_factory=list)
    per_lod_recovered_packets: list[int] = field(default_factory=list)
    per_lod_unrecovered_packets: list[int] = field(default_factory=list)
    per_lod_recovery_rate: list[float] = field(default_factory=list)
    # 汇总
    total_packets: int = 0
    total_lost: int = 0
    total_recovered: int = 0
    total_unrecovered: int = 0
    overall_recovery_rate: float = 0.0
    # 高斯级指标
    effective_gaussians: int = 0
    total_gaussians: int = 0
    gaussian_recovery_rate: float = 0.0

    def summary(self) -> str:
        lines = [
            f"SimulationResult(loss={self.loss_rate:.3f}, "
            f"packets: {self.total_packets} total, {self.total_lost} lost, "
            f"{self.total_recovered} recovered, {self.total_unrecovered} unrecovered)",
            f"  overall_recovery_rate={self.overall_recovery_rate:.4f}",
            f"  gaussians: {self.effective_gaussians}/{self.total_gaussians} "
            f"({self.gaussian_recovery_rate:.4f})",
        ]
        for i in range(len(self.per_lod_total_packets)):
            lines.append(
                f"  LOD {i}: {self.per_lod_recovery_rate[i]:.4f} recovery "
                f"({self.per_lod_recovered_packets[i]}/{self.per_lod_lost_packets[i]} "
                f"recovered, {self.per_lod_unrecovered_packets[i]} unrecovered)"
            )
        return '\n'.join(lines)


@dataclass
class StrategyComparison:
    """多策略对比结果"""
    loss_rates: list[float] = field(default_factory=list)
    strategy_names: list[str] = field(default_factory=list)
    # strategy_name → loss_rate → SimulationResult
    results: dict[str, dict[float, SimulationResult]] = field(default_factory=dict)

    def summary(self) -> str:
        if not self.results:
            return "StrategyComparison: no results"

        lines = ["Strategy Comparison:"]
        header = f"{'Strategy':>25s}"
        for lr in self.loss_rates:
            header += f"  {'loss=' + str(lr):>20s}"
        lines.append(header)
        lines.append("-" * (25 + 22 * len(self.loss_rates)))

        for name in self.strategy_names:
            row = f"{name:>25s}"
            for lr in self.loss_rates:
                r = self.results.get(name, {}).get(lr)
                if r:
                    row += f"  rec={r.overall_recovery_rate:.4f}"
                else:
                    row += f"  {'N/A':>20s}"
            lines.append(row)

        return '\n'.join(lines)


class FECSimulator:
    """离线 FEC 模拟器

    使用统计丢包模型 + RS 块恢复近似，模拟完整传输过程。
    可通过 seed 控制随机性，实现可复现的对比实验。
    """

    def __init__(self, seed: int | None = None):
        """
        Args:
            seed: 随机种子，用于可复现的模拟
        """
        self.seed = seed
        self._rng = random.Random(seed)

    def simulate(
        self,
        lod_sizes: list[int],
        lod_gaussian_counts: list[int] | None,
        loss_rate: float,
        allocation: FECAllocation,
    ) -> SimulationResult:
        """使用给定 FEC 分配模拟一次传输

        Args:
            lod_sizes: 每 LOD 的数据包数
            lod_gaussian_counts: 每 LOD 的高斯数量（用于计算高斯级恢复率）
            loss_rate: 模拟丢包率 [0, 1]
            allocation: FEC 分配方案

        Returns:
            SimulationResult: 模拟结果
        """
        K = len(lod_sizes)
        if lod_gaussian_counts is None:
            lod_gaussian_counts = lod_sizes

        per_lod_total = list(lod_sizes)
        per_lod_lost = [0] * K
        per_lod_recovered = [0] * K
        per_lod_unrecovered = [0] * K

        for lod in range(K):
            k, n = allocation.lod_configs.get(lod, (8, 8))
            n_packets = lod_sizes[lod]

            if k == n or n_packets == 0:
                # 无 FEC：每包独立丢包
                for _ in range(n_packets):
                    if self._rng.random() < loss_rate:
                        per_lod_lost[lod] += 1
                        per_lod_unrecovered[lod] += 1
                continue

            # 有 FEC：按 RS(n,k) 块模拟
            num_blocks = (n_packets + k - 1) // k
            pkt_idx = 0
            for bi in range(num_blocks):
                block_size = min(k, n_packets - pkt_idx)
                # 模拟该块中每个包的丢包
                block_lost = 0
                for _ in range(block_size):
                    if self._rng.random() < loss_rate:
                        block_lost += 1

                per_lod_lost[lod] += block_lost

                if block_lost <= (n - k):
                    # 全部可恢复
                    per_lod_recovered[lod] += block_lost
                else:
                    # 全部不可恢复（保守估计）
                    per_lod_unrecovered[lod] += block_lost

                pkt_idx += block_size

        # 汇总
        total_packets = sum(per_lod_total)
        total_lost = sum(per_lod_lost)
        total_recovered = sum(per_lod_recovered)
        total_unrecovered = sum(per_lod_unrecovered)
        overall_rec = (
            (total_packets - total_unrecovered) / max(total_packets, 1)
        )

        per_lod_rec = [
            (per_lod_total[i] - per_lod_unrecovered[i]) / max(per_lod_total[i], 1)
            if per_lod_total[i] > 0 else 1.0
            for i in range(K)
        ]

        # 高斯级指标：假设每个未恢复包中丢失的高斯均匀分布
        total_gaussians = sum(lod_gaussian_counts)
        effective_gaussians = 0
        for i in range(K):
            if per_lod_total[i] > 0:
                g_per_pkt = lod_gaussian_counts[i] / per_lod_total[i]
                lost_gaussians = per_lod_unrecovered[i] * g_per_pkt
                effective_gaussians += lod_gaussian_counts[i] - lost_gaussians
            else:
                effective_gaussians += lod_gaussian_counts[i]

        return SimulationResult(
            loss_rate=loss_rate,
            per_lod_total_packets=per_lod_total,
            per_lod_lost_packets=per_lod_lost,
            per_lod_recovered_packets=per_lod_recovered,
            per_lod_unrecovered_packets=per_lod_unrecovered,
            per_lod_recovery_rate=per_lod_rec,
            total_packets=total_packets,
            total_lost=total_lost,
            total_recovered=total_recovered,
            total_unrecovered=total_unrecovered,
            overall_recovery_rate=overall_rec,
            effective_gaussians=round(effective_gaussians),
            total_gaussians=total_gaussians,
            gaussian_recovery_rate=(
                effective_gaussians / max(total_gaussians, 1)
            ),
        )

    def simulate_multi(
        self,
        lod_sizes: list[int],
        lod_gaussian_counts: list[int] | None,
        loss_rate: float,
        allocation: FECAllocation,
        num_trials: int = 100,
    ) -> SimulationResult:
        """多次模拟取平均

        Args:
            num_trials: 模拟次数（默认 100）

        Returns:
            SimulationResult: 平均结果
        """
        results = []
        for _ in range(num_trials):
            results.append(
                self.simulate(lod_sizes, lod_gaussian_counts, loss_rate, allocation)
            )

        # 平均
        n = len(results)
        avg = SimulationResult(loss_rate=loss_rate)
        K = len(lod_sizes)

        avg.per_lod_total_packets = list(results[0].per_lod_total_packets)
        avg.per_lod_lost_packets = [
            round(sum(r.per_lod_lost_packets[i] for r in results) / n)
            for i in range(K)
        ]
        avg.per_lod_recovered_packets = [
            round(sum(r.per_lod_recovered_packets[i] for r in results) / n)
            for i in range(K)
        ]
        avg.per_lod_unrecovered_packets = [
            round(sum(r.per_lod_unrecovered_packets[i] for r in results) / n)
            for i in range(K)
        ]
        avg.per_lod_recovery_rate = [
            sum(r.per_lod_recovery_rate[i] for r in results) / n
            for i in range(K)
        ]
        avg.total_packets = results[0].total_packets
        avg.total_lost = round(sum(r.total_lost for r in results) / n)
        avg.total_recovered = round(sum(r.total_recovered for r in results) / n)
        avg.total_unrecovered = round(sum(r.total_unrecovered for r in results) / n)
        avg.overall_recovery_rate = (
            sum(r.overall_recovery_rate for r in results) / n
        )
        avg.effective_gaussians = round(
            sum(r.effective_gaussians for r in results) / n
        )
        avg.total_gaussians = results[0].total_gaussians
        avg.gaussian_recovery_rate = (
            sum(r.gaussian_recovery_rate for r in results) / n
        )

        return avg

    def compare_strategies(
        self,
        lod_sizes: list[int],
        lod_gaussian_counts: list[int] | None,
        loss_rates: list[float],
        strategies: dict[str, FECAllocation],
        num_trials: int = 100,
    ) -> StrategyComparison:
        """对比多种策略在多个丢包率下的表现

        Args:
            lod_sizes: 每 LOD 的数据包数
            lod_gaussian_counts: 每 LOD 的高斯数量
            loss_rates: 要测试的丢包率列表
            strategies: {策略名称: FECAllocation} 字典
            num_trials: 每个 (策略, 丢包率) 组合的模拟次数

        Returns:
            StrategyComparison: 对比结果
        """
        comparison = StrategyComparison(
            loss_rates=list(loss_rates),
            strategy_names=list(strategies.keys()),
        )

        for name, allocation in strategies.items():
            comparison.results[name] = {}
            for lr in loss_rates:
                log.info(
                    "Simulating: strategy=%s, loss=%.3f, trials=%d",
                    name, lr, num_trials,
                )
                result = self.simulate_multi(
                    lod_sizes, lod_gaussian_counts, lr, allocation, num_trials,
                )
                comparison.results[name][lr] = result
                log.info(
                    "  → recovery_rate=%.4f, gaussian_rec=%.4f",
                    result.overall_recovery_rate, result.gaussian_recovery_rate,
                )

        return comparison
