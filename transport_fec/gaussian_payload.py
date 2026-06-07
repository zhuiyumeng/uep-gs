import struct

# ---- from common (shared with transport/) ----
from common.payload import (  # noqa: F401 — re-export
    PayloadHeader,
    SceneMeta,
    make_dummy_gaussian,
    PAYLOAD_HEADER_SIZE,
    UNIT_SCENE_META,
    UNIT_GAUSSIAN_DATA,
    UNIT_END_OF_FRAME,
)
from common.fec_stats import FECStatsReport

# backward-compatible alias
FECStats = FECStatsReport

# ---- FEC-specific constants ----

UNIT_FEC_PARITY = 4

UEP_POLICY = {
    0: {"k": 1, "n": 2},   # RS(2,1),    +100%  repetition
    1: {"k": 8, "n": 10},  # RS(10,8),   +25%   bandwidth
    2: {"k": 8, "n": 9},   # RS(9,8),    +12.5% bandwidth
    3: {"k": 8, "n": 9},   # RS(9,8),    +12.5% bandwidth
}


def get_fec_config(lod: int, fec_policy=None) -> tuple[int, int]:
    """返回 (k, n) 配置。

    Args:
        lod: LOD 层级索引。
        fec_policy: 可选的自定义策略对象，需实现 get_config(lod) -> (k, n)。
                    为 None 时使用硬编码 UEP_POLICY 作为默认。
    """
    if fec_policy is not None:
        return fec_policy.get_config(lod)
    policy = UEP_POLICY.get(lod, {"k": 8, "n": 8})
    return policy["k"], policy["n"]


FEC_HEADER_SIZE = 3


def pack_fec_header(k: int, n: int, parity_index: int) -> bytes:
    return struct.pack('BBB', k, n, parity_index)


def parse_fec_header(data: bytes) -> tuple[int, int, int]:
    k, n, parity_index = struct.unpack('BBB', data[:3])
    return k, n, parity_index
