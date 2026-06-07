"""
transport_fec_ml — 自适应 FEC 编码模块

基于 ML/RL 驱动，根据 LOD 分布、网络状态（通过 RTCP 反馈获取）、带宽预算，
动态决策各 LOD 的 RS(k, n) 参数，实现不等差错保护（UEP）。

核心组件：
- AnalyticalOptimizer: 基于 RS 块恢复概率模型的 DP 优化器
- AdaptiveFECPolicy: 策略对象，可注入 RTPEncoder/RTPDecoder
- AdaptiveFECPipeline: 编排器，串联 NetworkEstimator + Optimizer 形成闭环
- FECSimulator: 离线模拟器，在无真实传输时评估策略效果
- NetworkEstimator: 从 RTCP Receiver Report 估计网络状态
- RTCP 包格式: 最小 RFC 3550 Receiver Report + FECStats 应用扩展
"""

from .adaptive_fec import AnalyticalOptimizer, AdaptiveFECPolicy, FECAllocation
from .network_estimator import NetworkEstimator, NetworkState
from .pipeline import AdaptiveFECPipeline, PipelineStep
from .simulator import FECSimulator, SimulationResult, StrategyComparison
from .rtcp import (
    RTCPReceiverReport,
    RTCPSenderReport,
    FECStatsReport,
    pack_rtcp_rr,
    parse_rtcp_rr,
    pack_rtcp_sr,
    parse_rtcp_sr,
    pack_fec_stats_report,
    parse_fec_stats_report,
    RTCP_PT_SR,
    RTCP_PT_RR,
)

__all__ = [
    # adaptive_fec
    "AnalyticalOptimizer",
    "AdaptiveFECPolicy",
    "FECAllocation",
    # network_estimator
    "NetworkEstimator",
    "NetworkState",
    # pipeline
    "AdaptiveFECPipeline",
    "PipelineStep",
    # simulator
    "FECSimulator",
    "SimulationResult",
    "StrategyComparison",
    # rtcp
    "RTCPReceiverReport",
    "RTCPSenderReport",
    "FECStatsReport",
    "pack_rtcp_rr",
    "parse_rtcp_rr",
    "pack_rtcp_sr",
    "parse_rtcp_sr",
    "pack_fec_stats_report",
    "parse_fec_stats_report",
    "RTCP_PT_SR",
    "RTCP_PT_RR",
]
