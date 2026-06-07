"""
统一 FEC 恢复统计数据结构

被 transport_fec（内部跟踪）和 transport_fec_ml（RTCP 上报）共享。
"""
from dataclasses import dataclass, field


@dataclass
class FECStatsReport:
    """Per-LOD FEC 恢复统计，供 RTCP 上报使用"""
    per_lod_total: list[int] = field(default_factory=list)
    per_lod_lost: list[int] = field(default_factory=list)
    per_lod_recovered: list[int] = field(default_factory=list)
    per_lod_unrecovered: list[int] = field(default_factory=list)
