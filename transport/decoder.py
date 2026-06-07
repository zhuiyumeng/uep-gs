from common.ply_utils import build_ply_header, write_ply  # noqa: F401 — re-export
from common.decoder import BaseRTPDecoder


class RTPDecoder(BaseRTPDecoder):
    """RTP 解码器（无 FEC）：线性收包 → 重组 → 写 PLY"""
    pass
