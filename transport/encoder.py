from common.encoder import BaseRTPEncoder, MAX_PAYLOAD
from common.rtp import RTPPacket
from common.payload import (
    PayloadHeader,
    PAYLOAD_HEADER_SIZE,
    UNIT_SCENE_META,
    UNIT_GAUSSIAN_DATA,
    UNIT_END_OF_FRAME,
)


class RTPEncoder(BaseRTPEncoder):
    """RTP 编码器（无 FEC）：线性分片，顺序封包"""

    def __init__(self, ssrc: int | None = None):
        super().__init__(ssrc=ssrc)

    def encode_frame(
        self,
        seq_start: int,
        timestamp: int,
        max_payload: int = MAX_PAYLOAD,
    ) -> list[RTPPacket]:
        """将帧数据编码为 RTP 包列表（无 FEC）"""
        if self.meta is None or not self.frame_data:
            raise RuntimeError('No frame loaded. Call load_lod_plys() first.')

        packets: list[RTPPacket] = []
        seq = seq_start

        # 1) SceneMeta 包
        meta_json = self.meta.to_json()
        meta_ph = PayloadHeader(
            start=1, end=1, unit_type=UNIT_SCENE_META,
        )
        packets.append(
            RTPPacket(
                sequence_number=seq,
                timestamp=timestamp,
                ssrc=self.ssrc,
                payload=meta_ph.serialize() + meta_json,
            )
        )
        seq += 1

        # 2) GaussianData 分片
        data = self.frame_data
        data_per_packet = max_payload - PAYLOAD_HEADER_SIZE
        offset = 0

        # 预计算 LOD 边界（字节偏移）
        lod_boundaries: list[int] = []
        cum = 0
        for sz in self.meta.lod_sizes:
            lod_boundaries.append(cum)
            cum += sz * self.meta.gaussian_stride * 4

        total = len(data)
        while offset < total:
            chunk = data[offset : offset + data_per_packet]
            is_start = 1 if offset == 0 else 0
            is_end = 1 if (offset + len(chunk)) >= total else 0

            # 根据 offset 推断 LOD 层级
            lod = 0
            for i in range(len(lod_boundaries) - 1, -1, -1):
                if offset >= lod_boundaries[i]:
                    lod = i
                    break

            ph = PayloadHeader(
                start=is_start,
                end=is_end,
                lod=lod,
                unit_type=UNIT_GAUSSIAN_DATA,
                fragment_offset=offset,
            )
            packets.append(
                RTPPacket(
                    sequence_number=seq,
                    timestamp=timestamp,
                    ssrc=self.ssrc,
                    payload=ph.serialize() + chunk,
                )
            )
            seq += 1
            offset += len(chunk)

        # 3) EndOfFrame 包
        eof_ph = PayloadHeader(
            start=1, end=1, unit_type=UNIT_END_OF_FRAME,
        )
        packets.append(
            RTPPacket(
                marker=1,
                sequence_number=seq,
                timestamp=timestamp,
                ssrc=self.ssrc,
                payload=eof_ph.serialize(),
            )
        )

        return packets
