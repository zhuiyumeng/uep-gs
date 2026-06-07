import random

from common.encoder import BaseRTPEncoder, MAX_PAYLOAD
from common.rtp import RTPPacket
from common.payload import (
    PayloadHeader,
    PAYLOAD_HEADER_SIZE,
    UNIT_SCENE_META,
    UNIT_GAUSSIAN_DATA,
    UNIT_END_OF_FRAME,
)
from .gaussian_payload import (
    UNIT_FEC_PARITY,
    FEC_HEADER_SIZE,
    pack_fec_header,
    get_fec_config,
)
from .fec import column_rs_encode

DATA_PER_PACKET = MAX_PAYLOAD - PAYLOAD_HEADER_SIZE


class RTPEncoder(BaseRTPEncoder):
    """RTP 编码器（含 FEC）：按 LOD 分组 → RS 列编码 → 数据+校验交叠发送"""

    def __init__(self, ssrc: int | None = None, fec_policy=None):
        super().__init__(ssrc=ssrc, fec_policy=fec_policy)
        self._fragments: list[dict] = []

    def get_fragments(self) -> list[dict]:
        return self._fragments

    def encode_frame(
        self,
        seq_start: int,
        timestamp: int,
        max_payload: int = MAX_PAYLOAD,
        fec: bool = True,
    ) -> list[RTPPacket]:
        if self.meta is None or not self.frame_data:
            raise RuntimeError('No frame loaded. Call load_lod_plys() first.')

        data_per_packet = max_payload - PAYLOAD_HEADER_SIZE
        total = len(self.frame_data)
        data = self.frame_data

        # --- fragment info list ---
        lod_boundaries: list[int] = []
        cum = 0
        for sz in self.meta.lod_sizes:
            lod_boundaries.append(cum)
            cum += sz * self.meta.gaussian_stride * 4

        fragments: list[dict] = []
        offset = 0
        while offset < total:
            chunk = data[offset: offset + data_per_packet]
            is_start = 1 if offset == 0 else 0
            is_end = 1 if (offset + len(chunk)) >= total else 0

            lod = 0
            for i in range(len(lod_boundaries) - 1, -1, -1):
                if offset >= lod_boundaries[i]:
                    lod = i
                    break

            fragments.append({
                'lod': lod,
                'offset': offset,
                'data': chunk,
                'is_start': is_start,
                'is_end': is_end,
            })
            offset += len(chunk)

        self._fragments = fragments

        # --- build ordered fragment list with optional FEC ---
        ordered: list[dict] = []

        # 1) SceneMeta
        meta_json = self.meta.to_json()
        meta_ph = PayloadHeader(
            start=1, end=1, unit_type=UNIT_SCENE_META,
        )
        ordered.append({
            'type': 'meta',
            'payload': meta_ph.serialize() + meta_json,
        })

        # 2) GaussianData + optional FECParity
        if fec:
            lod_groups: dict[int, list[dict]] = {}
            for fr in fragments:
                lod_groups.setdefault(fr['lod'], []).append(fr)

            global_block_id = 0
            for lod in sorted(lod_groups):
                group = lod_groups[lod]
                k_val, n_val = get_fec_config(lod, self.fec_policy)

                for block_start in range(0, len(group), k_val):
                    block = group[block_start:block_start + k_val]
                    if len(block) == k_val:
                        use_k, use_n = k_val, n_val
                        has_fec = k_val < n_val
                    else:
                        use_k = len(block)
                        use_n = use_k + 1
                        has_fec = (k_val < n_val) and use_k >= 1

                    # data packets
                    for fr in block:
                        ph = PayloadHeader(
                            start=fr['is_start'],
                            end=fr['is_end'],
                            lod=fr['lod'],
                            unit_type=UNIT_GAUSSIAN_DATA,
                            fragment_offset=fr['offset'],
                        )
                        ordered.append({
                            'type': 'data',
                            'payload': ph.serialize() + fr['data'],
                        })

                    # parity packets
                    if has_fec:
                        payloads = [fr['data'] for fr in block]
                        parity_list = column_rs_encode(payloads, use_k, use_n)
                        for pi, pp in enumerate(parity_list):
                            ph = PayloadHeader(
                                start=0, end=0,
                                lod=lod,
                                unit_type=UNIT_FEC_PARITY,
                                fragment_offset=global_block_id,
                            )
                            fec_hdr = pack_fec_header(use_k, use_n, pi)
                            ordered.append({
                                'type': 'fec',
                                'payload': ph.serialize() + fec_hdr + pp,
                            })
                        global_block_id += 1

        else:
            # no FEC — original behaviour
            for fr in fragments:
                ph = PayloadHeader(
                    start=fr['is_start'],
                    end=fr['is_end'],
                    lod=fr['lod'],
                    unit_type=UNIT_GAUSSIAN_DATA,
                    fragment_offset=fr['offset'],
                )
                ordered.append({
                    'type': 'data',
                    'payload': ph.serialize() + fr['data'],
                })

        # 3) EndOfFrame
        eof_ph = PayloadHeader(
            start=1, end=1, unit_type=UNIT_END_OF_FRAME,
        )
        ordered.append({
            'type': 'eof',
            'payload': eof_ph.serialize(),
        })

        # --- assign sequence numbers & build RTPPackets ---
        packets: list[RTPPacket] = []
        seq = seq_start
        for item in ordered:
            marker = 1 if item['type'] == 'eof' else 0
            packets.append(
                RTPPacket(
                    marker=marker,
                    sequence_number=seq,
                    timestamp=timestamp,
                    ssrc=self.ssrc,
                    payload=item['payload'],
                )
            )
            seq += 1

        return packets
