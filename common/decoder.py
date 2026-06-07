"""
BaseRTPDecoder — 共享 RTP 解码状态机 + 钩子扩展点

transport/decoder.py 和 transport_fec/decoder.py 的公共基类。
FEC 子类通过覆写 6 个钩子注入块记录/校验恢复逻辑，无需复制整个状态机。
"""
from common.rtp import RTPPacket
from common.payload import (
    PayloadHeader,
    SceneMeta,
    PAYLOAD_HEADER_SIZE,
    UNIT_SCENE_META,
    UNIT_GAUSSIAN_DATA,
    UNIT_END_OF_FRAME,
)


class BaseRTPDecoder:
    """共享解码器基类：收包 → 排序 → 缓冲重组 → 帧完成回调

    钩子方法（子类覆写）：
    - _init_extra_state()            — __init__ 末尾调用
    - _reset_extra_state()           — reset() 末尾调用
    - _handle_extra_unit_types()     — _process_packet 开头调用，返回 True 则跳过默认分发
    - _on_scene_meta_received()      — _handle_scene_meta 末尾调用
    - _on_gaussian_data_written()    — _handle_gaussian_data 末尾调用
    - _before_frame_complete()       — flush()/_handle_end_of_frame 标记完成前调用
    """

    def __init__(self, fec_policy=None):
        self.buffer: bytearray | None = None
        self.meta: SceneMeta | None = None
        self.dummy_gaussian: bytes | None = None
        self.current_ts: int = -1
        self._frame_complete: bool = False
        self._pending: dict[int, RTPPacket] = {}
        self.fec_policy = fec_policy
        self._init_extra_state()

    # ---- public API ----

    def feed_packet(self, packet: RTPPacket):
        """注入一个收到的 RTP 包（可乱序）"""
        if self._frame_complete:
            return
        self._pending[packet.sequence_number] = packet
        self._try_assemble()

    def flush(self):
        """强制结束当前帧（即使未收到 EndOfFrame），填充丢失片段为 dummy gaussian。"""
        if self._frame_complete:
            return
        if self.buffer is None or self.meta is None:
            return
        self._before_frame_complete()
        self._frame_complete = True
        self._on_frame_complete()

    def get_complete_frame(self) -> bytes | None:
        """获取已完成的帧数据"""
        if self.buffer is None:
            return None
        return bytes(self.buffer)

    def _on_frame_complete(self):
        """帧完成回调 — 子类或外部重写"""
        pass

    def reset(self):
        """重置解码状态"""
        self.buffer = None
        self.meta = None
        self.dummy_gaussian = None
        self.current_ts = -1
        self._frame_complete = False
        self._pending.clear()
        self._reset_extra_state()

    # ---- internal: assembly ----

    def _try_assemble(self):
        if not self._pending:
            return

        for seq in sorted(self._pending.keys()):
            if self._frame_complete:
                self._pending.clear()
                break
            pkt = self._pending.pop(seq)
            self._process_packet(pkt)

    def _process_packet(self, pkt: RTPPacket):
        payload = pkt.payload
        if len(payload) < PAYLOAD_HEADER_SIZE:
            return

        ph = PayloadHeader.parse(payload[:PAYLOAD_HEADER_SIZE])
        body = payload[PAYLOAD_HEADER_SIZE:]

        # hook: allow subclass to intercept extra unit types (e.g. FEC_PARITY)
        if self._handle_extra_unit_types(ph, body, pkt):
            return

        if ph.unit_type == UNIT_SCENE_META:
            self._handle_scene_meta(body, pkt)

        elif ph.unit_type == UNIT_GAUSSIAN_DATA:
            self._handle_gaussian_data(ph, body, pkt)

        elif ph.unit_type == UNIT_END_OF_FRAME:
            self._handle_end_of_frame()

    # ---- internal: handlers ----

    def _handle_scene_meta(self, body: bytes, pkt: RTPPacket):
        if self.buffer is not None and not self._frame_complete:
            return
        self.meta = SceneMeta.from_json(body)
        dummy_bytes = bytes.fromhex(self.meta.dummy_gaussian_hex)
        self.dummy_gaussian = dummy_bytes
        self.current_ts = pkt.timestamp

        # 初始化缓冲区，全部用 dummy 填充
        total = self.meta.total_bytes
        self.buffer = bytearray(total)
        stride = len(dummy_bytes)
        for off in range(0, total, stride):
            end = min(off + stride, total)
            self.buffer[off:end] = dummy_bytes[:end - off]

        # hook: post-process (e.g. LOD boundaries, block_id precomputation)
        self._on_scene_meta_received()

    def _handle_gaussian_data(
        self, ph: PayloadHeader, body: bytes, pkt: RTPPacket,
    ):
        if self.buffer is None or self.current_ts != pkt.timestamp:
            return
        end = ph.fragment_offset + len(body)
        if end > len(self.buffer):
            end = len(self.buffer)
        self.buffer[ph.fragment_offset: end] = body[:end - ph.fragment_offset]

        # hook: post-process (e.g. record for FEC)
        self._on_gaussian_data_written(ph, body)

    def _handle_end_of_frame(self):
        if self.meta and self.buffer is not None and not self._frame_complete:
            self._before_frame_complete()
            self._frame_complete = True
            self._on_frame_complete()

    # ---- hooks (no-op defaults, overridden by FEC subclass) ----

    def _init_extra_state(self):
        """初始化子类特有状态（__init__ 末尾调用）"""
        pass

    def _reset_extra_state(self):
        """清理子类特有状态（reset() 末尾调用）"""
        pass

    def _handle_extra_unit_types(
        self, ph: PayloadHeader, body: bytes, pkt: RTPPacket,
    ) -> bool:
        """处理额外 unit_type。返回 True 表示已处理，跳过默认分发。"""
        return False

    def _on_scene_meta_received(self):
        """SceneMeta 处理后回调"""
        pass

    def _on_gaussian_data_written(self, ph: PayloadHeader, body: bytes):
        """Gaussian data 写入缓冲区后回调"""
        pass

    def _before_frame_complete(self):
        """帧完成前回调（flush 或 EndOfFrame 中调用）"""
        pass
