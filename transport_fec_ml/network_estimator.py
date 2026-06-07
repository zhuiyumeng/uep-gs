"""
网络状态估计器（NetworkEstimator）

从 RTCP Receiver Report 中估计当前网络状态，使用可配置的 EMA 指数平滑。
"""

import time
import logging
from dataclasses import dataclass, field
from .rtcp import RTCPReceiverReport, RTCPReportBlock, FECStatsReport

log = logging.getLogger(__name__)


@dataclass
class NetworkState:
    """当前网络状态快照"""
    loss_rate: float = 0.0              # EMA 平滑后的丢包率 [0, 1]
    jitter: float = 0.0                 # EMA 平滑后的到达间隔抖动
    rtt_estimate: float = 0.0           # 从 SR/DLSR 估计的 RTT (秒)
    per_lod_loss_rate: list[float] = field(default_factory=list)
    per_lod_fec_recovery_rate: list[float] = field(default_factory=list)
    raw_loss_rate: float = 0.0          # 最近一次上报的原始丢包率（未平滑）


class NetworkEstimator:
    """从 RTCP RR 中估计当前网络状态

    使用可配置的 EMA（指数移动平均）对丢包率和抖动进行平滑。
    alpha 控制平滑度 vs 响应灵敏度的权衡：

    - alpha=0.0: 完全平滑（对变化无响应，退化到恒定初始值）
    - alpha=1.0: 无平滑（即时响应，但对噪声极度敏感）
    - 默认 0.3:  在 1s 上报间隔下提供适度平滑

    冷启动行为：
    首个 RTCP RR 使用高 alpha (0.6) + 保守先验 (5% 丢包率) 进行初始估计，
    避免首个报告恰好 p=0 时 EMA 被钉在 0 附近。

    选择指南：
    | 场景                          | alpha | rtcp_interval |
    |-------------------------------|-------|---------------|
    | 稳态网络（实验室）              | 0.2   | 1.0s          |
    | 波动网络（真实互联网）          | 0.5   | 0.5s          |
    | 突发丢包（闪断模拟）            | 0.7   | 0.1s          |
    | 带宽极度受限                    | 0.5   | 2.0s          |
    """

    # 冷启动参数
    COLD_START_ALPHA = 0.6          # 首次报告使用的高 alpha
    COLD_START_PRIOR_LOSS = 0.05    # 保守先验丢包率

    def __init__(self, alpha: float = 0.3, alpha_jitter: float = 0.3):
        """
        Args:
            alpha: 丢包率 EMA 平滑系数 (0 ~ 1)
            alpha_jitter: 抖动 EMA 平滑系数 (0 ~ 1)，通常需要比 alpha 更小的值
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'alpha must be in [0, 1], got {alpha}')
        if not 0.0 <= alpha_jitter <= 1.0:
            raise ValueError(f'alpha_jitter must be in [0, 1], got {alpha_jitter}')
        self.alpha = alpha
        self.alpha_jitter = alpha_jitter
        self._state = NetworkState()
        self._report_count: int = 0
        # Per-LOD EMA 状态
        self._per_lod_loss_ema: list[float] = []
        self._per_lod_fec_rec_ema: list[float] = []

    @property
    def state(self) -> NetworkState:
        return self._state

    @property
    def loss_rate(self) -> float:
        return self._state.loss_rate

    @property
    def jitter(self) -> float:
        return self._state.jitter

    def feed_rtcp_report(self, rr: RTCPReceiverReport,
                         fec_stats: FECStatsReport | None = None):
        """处理一次 RTCP Receiver Report

        Args:
            rr: 解析后的 RTCP Receiver Report
            fec_stats: 可选的 FEC 恢复统计扩展
        """
        if not rr.report_blocks:
            return

        # 取第一个 (通常也是唯一一个) Report Block
        rb = rr.report_blocks[0]
        p_raw = rb.fraction_lost / 256.0

        # EMA 平滑更新（含冷启动处理）
        if self._report_count == 0:
            # 冷启动：高 alpha + 保守先验，避免首报 p=0 导致 EMA 锁定在 0
            self._state.loss_rate = (
                self.COLD_START_ALPHA * p_raw
                + (1 - self.COLD_START_ALPHA) * self.COLD_START_PRIOR_LOSS
            )
            self._state.jitter = float(rb.interarrival_jitter)
            log.info(
                "NetworkEstimator cold start: raw=%.4f, alpha_first=%.2f, "
                "prior=%.2f → loss_rate=%.4f",
                p_raw, self.COLD_START_ALPHA,
                self.COLD_START_PRIOR_LOSS, self._state.loss_rate,
            )
        else:
            self._state.loss_rate = (
                self.alpha * p_raw + (1 - self.alpha) * self._state.loss_rate
            )
            self._state.jitter = (
                self.alpha_jitter * rb.interarrival_jitter
                + (1 - self.alpha_jitter) * self._state.jitter
            )

        self._state.raw_loss_rate = p_raw

        # RTT 估计：从 last_sr + delay_since_last_sr 计算
        # A = 当前到达时间, LSR = 被引用 SR 的 NTP 中段 (1/65536 s),
        # DLSR = 客户端处理延迟 (1/65536 s),
        # RTT = A - LSR - DLSR
        if rb.last_sr != 0 and rb.delay_since_last_sr != 0:
            now_ntp = int(time.time() * 65536) & 0xFFFFFFFF
            rtt_raw = (now_ntp - rb.last_sr - rb.delay_since_last_sr) / 65536.0
            if rtt_raw > 0:
                if self._report_count == 0:
                    self._state.rtt_estimate = rtt_raw
                else:
                    self._state.rtt_estimate = (
                        self.alpha * rtt_raw
                        + (1 - self.alpha) * self._state.rtt_estimate
                    )

        self._report_count += 1

        # 处理 FECStats → per-LOD 统计 (EMA 平滑)
        if fec_stats is not None and fec_stats.per_lod_total:
            n = len(fec_stats.per_lod_total)
            per_lod_raw = [
                lost / max(total, 1)
                for lost, total in zip(
                    fec_stats.per_lod_lost, fec_stats.per_lod_total,
                )
            ]
            per_lod_fec_raw = [
                recovered / max(lost, 1) if lost > 0 else 1.0
                for recovered, lost in zip(
                    fec_stats.per_lod_recovered, fec_stats.per_lod_lost,
                )
            ]

            if len(self._per_lod_loss_ema) != n:
                # 首次或 LOD 数量变化 → 直接赋值
                self._per_lod_loss_ema = list(per_lod_raw)
                self._per_lod_fec_rec_ema = list(per_lod_fec_raw)
            else:
                # EMA 平滑
                self._per_lod_loss_ema = [
                    self.alpha * new + (1 - self.alpha) * old
                    for new, old in zip(per_lod_raw, self._per_lod_loss_ema)
                ]
                self._per_lod_fec_rec_ema = [
                    self.alpha * new + (1 - self.alpha) * old
                    for new, old in zip(per_lod_fec_raw, self._per_lod_fec_rec_ema)
                ]

            self._state.per_lod_loss_rate = list(self._per_lod_loss_ema)
            self._state.per_lod_fec_recovery_rate = list(self._per_lod_fec_rec_ema)

    def feed_simulated(self, loss_rate: float, jitter: float = 0.0,
                       fec_stats: dict | None = None):
        """直接注入模拟的网络状态（用于离线实验，不需要真实 RTCP 包）

        Args:
            loss_rate: 模拟的丢包率 [0, 1]
            jitter: 模拟的抖动值
            fec_stats: 可选的 per-LOD 统计字典
        """
        if self._report_count == 0:
            self._state.loss_rate = (
                self.COLD_START_ALPHA * loss_rate
                + (1 - self.COLD_START_ALPHA) * self.COLD_START_PRIOR_LOSS
            )
            self._state.jitter = jitter
        else:
            self._state.loss_rate = (
                self.alpha * loss_rate + (1 - self.alpha) * self._state.loss_rate
            )
            self._state.jitter = (
                self.alpha_jitter * jitter
                + (1 - self.alpha_jitter) * self._state.jitter
            )
        self._state.raw_loss_rate = loss_rate
        self._report_count += 1

        if fec_stats is not None:
            per_lod_loss = fec_stats.get('per_lod_loss_rate', [])
            per_lod_fec = fec_stats.get('per_lod_fec_recovery_rate', [])
            if per_lod_loss:
                if len(self._per_lod_loss_ema) != len(per_lod_loss):
                    self._per_lod_loss_ema = list(per_lod_loss)
                else:
                    self._per_lod_loss_ema = [
                        self.alpha * new + (1 - self.alpha) * old
                        for new, old in zip(per_lod_loss, self._per_lod_loss_ema)
                    ]
                self._state.per_lod_loss_rate = list(self._per_lod_loss_ema)
            if per_lod_fec:
                if len(self._per_lod_fec_rec_ema) != len(per_lod_fec):
                    self._per_lod_fec_rec_ema = list(per_lod_fec)
                else:
                    self._per_lod_fec_rec_ema = [
                        self.alpha * new + (1 - self.alpha) * old
                        for new, old in zip(per_lod_fec, self._per_lod_fec_rec_ema)
                    ]
                self._state.per_lod_fec_recovery_rate = list(self._per_lod_fec_rec_ema)

    def reset(self):
        """重置估计器状态"""
        self._state = NetworkState()
        self._report_count = 0
        self._per_lod_loss_ema = []
        self._per_lod_fec_rec_ema = []
