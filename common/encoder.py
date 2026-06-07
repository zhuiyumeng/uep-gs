"""
BaseRTPEncoder — 共享的 PLY 加载 + SceneMeta 构建逻辑

transport/encoder.py 和 transport_fec/encoder.py 的公共基类。
子类只需覆写 encode_frame() 实现各自的分片/封包策略。
"""
import os
import struct
import random

from common.rtp import RTPPacket
from common.payload import (
    PayloadHeader,
    SceneMeta,
    make_dummy_gaussian,
    PAYLOAD_HEADER_SIZE,
)

MAX_PAYLOAD = 1400


class BaseRTPEncoder:
    """共享编码器基类：加载 LOD PLY → 构建 SceneMeta → 存储 frame_data。

    子类覆写 encode_frame() 实现分片/封包（可包含 FEC 或纯数据）。
    """

    def __init__(self, ssrc: int | None = None, fec_policy=None):
        self.ssrc = ssrc if ssrc is not None else random.randint(1, 0xFFFFFFFF)
        self.frame_data: bytes = b''
        self.meta: SceneMeta | None = None
        self.fec_policy = fec_policy

    def load_lod_plys(self, lod_files: list[str], sh_degree: int):
        """加载 LOD PLY 文件，拼接为完整帧数据"""
        lod_sizes: list[int] = []
        all_data = bytearray()

        for fpath in lod_files:
            if not os.path.exists(fpath):
                raise FileNotFoundError(f'LOD PLY not found: {fpath}')
            with open(fpath, 'rb') as fh:
                while True:
                    line = fh.readline()
                    if not line:
                        raise ValueError(f'Unexpected EOF in PLY header: {fpath}')
                    if line.strip() == b'end_header':
                        break
                chunk = fh.read()
            all_data.extend(chunk)

            if sh_degree == 3:
                stride = 17 + 45
            elif sh_degree == 2:
                stride = 17 + 24
            else:
                stride = 17 + 3 * ((sh_degree + 1) ** 2 - 1)
            n_gaussians = len(chunk) // (stride * 4)
            lod_sizes.append(n_gaussians)

        total_gaussians = sum(lod_sizes)
        gaussian_stride = 17 + 3 * ((sh_degree + 1) ** 2 - 1)
        total_bytes = len(all_data)

        dummy = make_dummy_gaussian(sh_degree)
        dummy_bytes = struct.pack(f'<{len(dummy)}f', *dummy)

        self.meta = SceneMeta(
            sh_degree=sh_degree,
            num_lods=len(lod_files),
            lod_sizes=lod_sizes,
            total_gaussians=total_gaussians,
            total_bytes=total_bytes,
            gaussian_stride=gaussian_stride,
            dummy_gaussian_hex=dummy_bytes.hex(),
        )
        self.frame_data = bytes(all_data)

    def encode_frame(
        self,
        seq_start: int,
        timestamp: int,
        max_payload: int = MAX_PAYLOAD,
    ) -> list[RTPPacket]:
        raise NotImplementedError('Subclass must implement encode_frame()')
