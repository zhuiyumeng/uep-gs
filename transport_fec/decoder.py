from dataclasses import dataclass, field

from common.ply_utils import build_ply_header, write_ply  # noqa: F401 — re-export
from common.decoder import BaseRTPDecoder
from common.payload import PayloadHeader

from .gaussian_payload import (
    FEC_HEADER_SIZE,
    parse_fec_header,
    UNIT_FEC_PARITY,
    get_fec_config,
)
from .fec import column_rs_decode

DATA_PER_PACKET = 1392


@dataclass
class _FECBlock:
    block_id: int
    lod: int
    k: int
    n: int
    data: dict[int, bytes] = field(default_factory=dict)
    parity: dict[int, bytes] = field(default_factory=dict)


class RTPDecoder(BaseRTPDecoder):
    """RTP 解码器（含 FEC）：收包 → 重组 → FEC 恢复 → 写 PLY"""

    def __init__(self, use_fec: bool = True, fec_policy=None):
        self._use_fec = use_fec
        super().__init__(fec_policy=fec_policy)

    # ---- hooks ----

    def _init_extra_state(self):
        if not hasattr(self, '_use_fec'):
            self._use_fec = True  # set by __init__ before super()
        self._fec_blocks: dict[int, _FECBlock] | None = None
        self._lod_boundaries: list[int] = []
        self._block_id_base: dict[int, int] = {}
        self._fec_policy = getattr(self, 'fec_policy', None)

    def _reset_extra_state(self):
        self._fec_blocks = None
        self._lod_boundaries = []
        self._block_id_base = {}

    def _handle_extra_unit_types(
        self, ph: PayloadHeader, body: bytes, pkt,
    ) -> bool:
        if ph.unit_type == UNIT_FEC_PARITY and self._use_fec:
            self._handle_fec_parity(ph, body, pkt)
            return True
        return False

    def _on_scene_meta_received(self):
        if self.meta is None:
            return

        # Precompute LOD boundaries
        self._lod_boundaries = []
        cum = 0
        for sz in self.meta.lod_sizes:
            self._lod_boundaries.append(cum)
            cum += sz * self.meta.gaussian_stride * 4
        self._lod_boundaries.append(cum)

        # Precompute block_id base per LOD
        if self._use_fec:
            self._fec_blocks = {}
            self._block_id_base = {}
            base = 0
            for lod in range(self.meta.num_lods):
                k, n = get_fec_config(lod, self._fec_policy)
                if k != n:
                    lod_bytes = (
                        self.meta.lod_sizes[lod]
                        * self.meta.gaussian_stride
                        * 4
                    )
                    n_packets = (
                        lod_bytes + DATA_PER_PACKET - 1
                    ) // DATA_PER_PACKET
                    if n_packets >= k:
                        self._block_id_base[lod] = base
                        base += (n_packets + k - 1) // k
                    else:
                        self._block_id_base[lod] = -1
                else:
                    self._block_id_base[lod] = -1

    def _on_gaussian_data_written(self, ph: PayloadHeader, body: bytes):
        if self._use_fec and self._fec_blocks is not None:
            self._record_fec_data(ph, body)

    def _before_frame_complete(self):
        self._try_fec_recovery()

    # ---- FEC-specific methods ----

    def _record_fec_data(self, ph: PayloadHeader, body: bytes):
        lod = ph.lod
        k, n = get_fec_config(lod, self._fec_policy)
        if k == n:
            return

        base_id = self._block_id_base.get(lod, -1)
        if base_id < 0:
            return

        frag_off = ph.fragment_offset
        lod_start = self._lod_boundaries[lod]
        packet_idx = (frag_off - lod_start) // DATA_PER_PACKET
        block_idx = packet_idx // k
        index_in_block = packet_idx % k

        block_id = base_id + block_idx

        if block_id not in self._fec_blocks:
            self._fec_blocks[block_id] = _FECBlock(
                block_id=block_id, lod=lod, k=k, n=n,
            )
        self._fec_blocks[block_id].data[index_in_block] = body

    def _handle_fec_parity(
        self, ph: PayloadHeader, body: bytes, pkt,
    ):
        if len(body) < FEC_HEADER_SIZE:
            return
        k, n, parity_index = parse_fec_header(body)
        parity_data = body[FEC_HEADER_SIZE:]
        block_id = ph.fragment_offset & 0xFFFF

        if self._fec_blocks is None:
            return

        if block_id not in self._fec_blocks:
            self._fec_blocks[block_id] = _FECBlock(
                block_id=block_id, lod=ph.lod, k=k, n=n,
            )
        else:
            self._fec_blocks[block_id].k = k
            self._fec_blocks[block_id].n = n
        self._fec_blocks[block_id].parity[parity_index] = parity_data

    def _try_fec_recovery(self):
        if not self._use_fec or self._fec_blocks is None:
            return
        if self.buffer is None or self.meta is None:
            return

        for block_id, block in list(self._fec_blocks.items()):
            missing = block.k - len(block.data)
            if missing == 0:
                continue

            data_list = [block.data.get(i) for i in range(block.k)]
            parity_list = [block.parity.get(i) for i in range(block.n - block.k)]

            recovered = column_rs_decode(
                data_list, parity_list, block.k, block.n,
            )
            if recovered is None:
                continue

            # Write recovered data into buffer
            lod = block.lod
            lod_start = self._lod_boundaries[lod]
            k_orig, _ = get_fec_config(lod, self._fec_policy)
            base_packet_idx = (block_id - self._block_id_base.get(lod, 0)) * k_orig
            first_frag_off = (
                (lod_start + DATA_PER_PACKET - 1) // DATA_PER_PACKET
            ) * DATA_PER_PACKET

            for i_in_block, payload in enumerate(recovered):
                if block.data.get(i_in_block) is not None:
                    continue
                packet_idx = base_packet_idx + i_in_block
                frag_off = first_frag_off + packet_idx * DATA_PER_PACKET
                end = frag_off + len(payload)
                if end > len(self.buffer):
                    end = len(self.buffer)
                self.buffer[frag_off: end] = payload[:end - frag_off]
