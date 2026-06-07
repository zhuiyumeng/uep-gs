"""
自适应 FEC 编排器 (AdaptiveFECPipeline)

将 NetworkEstimator + AnalyticalOptimizer 串联为完整的自适应反馈回路。
用于替代硬编码 UEP_POLICY，根据实时网络状态动态调整 FEC 策略。

典型用法:
    pipeline = AdaptiveFECPipeline(alpha=0.3, bandwidth_budget=0.20)

    # 每收到一个 RTCP RR:
    state = pipeline.feed_rtcp(rr, fec_stats)
    if state.loss_rate > 0.01:
        allocation = pipeline.optimize(lod_sizes, lod_gaussian_counts)
        encoder.fec_policy = pipeline.get_policy()

    # 查看策略历史
    print(pipeline.summary())
"""

import logging
from dataclasses import dataclass, field

from .adaptive_fec import AnalyticalOptimizer, AdaptiveFECPolicy, FECAllocation
from .network_estimator import NetworkEstimator, NetworkState
from .rtcp import RTCPReceiverReport, FECStatsReport

log = logging.getLogger(__name__)


@dataclass
class PipelineStep:
    """单次策略更新的快照"""
    step: int
    loss_rate: float
    jitter: float
    rtt: float
    allocation: FECAllocation
    per_lod_loss: list[float] = field(default_factory=list)
    per_lod_fec_rec: list[float] = field(default_factory=list)


class AdaptiveFECPipeline:
    """自适应 FEC 编排器

    职责：
    1. 接收 RTCP RR → NetworkEstimator → 更新网络状态
    2. 根据最新网络状态 → AnalyticalOptimizer → 生成 FEC 分配
    3. 生成/更新 AdaptiveFECPolicy → 供 Encoder/Decoder 注入

    策略切换条件（可选）：
    - loss_change_threshold: 丢包率变化超过此阈值时才重新优化
    - min_interval_steps: 两次优化之间的最小步数间隔
    """

    def __init__(
        self,
        alpha: float = 0.3,
        alpha_jitter: float = 0.3,
        bandwidth_budget: float = 0.20,
        loss_change_threshold: float = 0.005,
        min_interval_steps: int = 1,
    ):
        """
        Args:
            alpha: 丢包率 EMA 平滑系数
            alpha_jitter: 抖动 EMA 平滑系数
            bandwidth_budget: 带宽预算比例 (如 0.20 = 20%)
            loss_change_threshold: 丢包率变化阈值，超过才重新优化
            min_interval_steps: 优化最小间隔步数
        """
        self.estimator = NetworkEstimator(alpha=alpha, alpha_jitter=alpha_jitter)
        self.bandwidth_budget = bandwidth_budget
        self.loss_change_threshold = loss_change_threshold
        self.min_interval_steps = min_interval_steps

        self.current_policy: AdaptiveFECPolicy | None = None
        self.current_allocation: FECAllocation | None = None
        self.history: list[PipelineStep] = []
        self._step_count: int = 0
        self._last_optimize_loss: float = -1.0
        self._steps_since_optimize: int = 999

    @property
    def state(self) -> NetworkState:
        return self.estimator.state

    @property
    def loss_rate(self) -> float:
        return self.estimator.loss_rate

    def feed_rtcp(
        self,
        rr: RTCPReceiverReport,
        fec_stats: FECStatsReport | None = None,
    ) -> NetworkState:
        """处理一次 RTCP 反馈

        Args:
            rr: RTCP Receiver Report
            fec_stats: 可选的 FEC 恢复统计扩展

        Returns:
            更新后的 NetworkState
        """
        self.estimator.feed_rtcp_report(rr, fec_stats)
        self._step_count += 1
        self._steps_since_optimize += 1
        return self.estimator.state

    def feed_simulated(
        self,
        loss_rate: float,
        jitter: float = 0.0,
        fec_stats: dict | None = None,
    ) -> NetworkState:
        """直接注入模拟网络状态（离线实验用）

        Args:
            loss_rate: 模拟丢包率
            jitter: 模拟抖动
            fec_stats: 可选的 per-LOD 统计

        Returns:
            更新后的 NetworkState
        """
        self.estimator.feed_simulated(loss_rate, jitter, fec_stats)
        self._step_count += 1
        self._steps_since_optimize += 1
        return self.estimator.state

    def should_optimize(self) -> bool:
        """判断是否应该触发策略重优化"""
        if self._step_count == 0:
            return True

        # 最小间隔检查
        if self._steps_since_optimize < self.min_interval_steps:
            return False

        # 丢包率变化检查
        current_loss = self.estimator.loss_rate
        if abs(current_loss - self._last_optimize_loss) < self.loss_change_threshold:
            return False

        return True

    def optimize(
        self,
        lod_sizes: list[int],
        lod_gaussian_counts: list[int] | None = None,
        lod_weights: list[float] | None = None,
        force: bool = False,
    ) -> FECAllocation | None:
        """基于当前网络状态计算最优 FEC 分配

        只有在 should_optimize() 返回 True 或 force=True 时才重新计算。
        否则返回 None（保持当前策略不变）。

        Args:
            lod_sizes: 每 LOD 的数据包数
            lod_gaussian_counts: 每 LOD 的高斯数量（用于 benefit 归一化）
            lod_weights: 每 LOD 的重要性权重
            force: 强制重新优化（忽略阈值检查）

        Returns:
            新的 FECAllocation，或 None（无需更新）
        """
        if not force and not self.should_optimize():
            log.debug(
                "AdaptiveFECPipeline: skipping optimize (step=%d, "
                "loss=%.4f, last_optimize_loss=%.4f)",
                self._step_count, self.estimator.loss_rate, self._last_optimize_loss,
            )
            return None

        state = self.estimator.state
        log.info(
            "AdaptiveFECPipeline.optimize: step=%d, loss_rate=%.4f, "
            "jitter=%.1f, rtt=%.4fs, budget=%.2f%%",
            self._step_count, state.loss_rate, state.jitter,
            state.rtt_estimate, self.bandwidth_budget * 100,
        )

        allocation = AnalyticalOptimizer.optimize(
            lod_sizes=lod_sizes,
            loss_rate=state.loss_rate,
            bandwidth_budget=self.bandwidth_budget,
            lod_weights=lod_weights,
            lod_gaussian_counts=lod_gaussian_counts,
        )

        self.current_allocation = allocation
        self.current_policy = AdaptiveFECPolicy(allocation)
        self._last_optimize_loss = state.loss_rate
        self._steps_since_optimize = 0

        step_record = PipelineStep(
            step=self._step_count,
            loss_rate=state.loss_rate,
            jitter=state.jitter,
            rtt=state.rtt_estimate,
            allocation=allocation,
            per_lod_loss=list(state.per_lod_loss_rate),
            per_lod_fec_rec=list(state.per_lod_fec_recovery_rate),
        )
        self.history.append(step_record)

        log.info(
            "AdaptiveFECPipeline: new policy applied — %s", allocation.summary()
        )

        return allocation

    def get_policy(self) -> AdaptiveFECPolicy | None:
        """返回当前策略对象（供 Encoder/Decoder 注入）"""
        return self.current_policy

    def get_allocation(self) -> FECAllocation | None:
        """返回当前分配方案"""
        return self.current_allocation

    def summary(self) -> str:
        """返回策略切换历史摘要"""
        if not self.history:
            return "AdaptiveFECPipeline: no history (no optimizations yet)"

        lines = [
            f"AdaptiveFECPipeline: {len(self.history)} optimizations "
            f"over {self._step_count} steps",
            f"{'Step':>5s}  {'Loss':>7s}  {'Jitter':>7s}  {'RTT':>8s}  "
            f"{'Overhead':>9s}  {'WRR':>8s}  {'Configs'}",
            "-" * 80,
        ]
        for s in self.history:
            config_str = ", ".join(
                f"L{i}:RS({n},{k})"
                for i, (k, n) in sorted(s.allocation.lod_configs.items())
                if k != n
            )
            if not config_str:
                config_str = "all no-FEC"
            lines.append(
                f"{s.step:5d}  {s.loss_rate:7.4f}  {s.jitter:7.1f}  "
                f"{s.rtt:8.4f}  {s.allocation.total_overhead_ratio:9.2%}  "
                f"{s.allocation.weighted_recovery_rate:8.4f}  {config_str}"
            )

        return '\n'.join(lines)

    def reset(self):
        """重置编排器状态"""
        self.estimator.reset()
        self.current_policy = None
        self.current_allocation = None
        self.history.clear()
        self._step_count = 0
        self._last_optimize_loss = -1.0
        self._steps_since_optimize = 999
