"""
RTCP 包格式定义（RFC 3550 最小子集 + FECStats 应用扩展）

支持两种 RTCP 包类型：
- Sender Report (SR, PT=200): 服务端周期发送，用于 RTT 计算
- Receiver Report (RR, PT=201): 客户端周期发送，携带丢包率/抖动 + FEC 恢复统计

FECStats 扩展使用 RFC 3550 §6.5.2 的应用特定扩展机制：
- 类型标记: FECS (0x4645 4353 = b'FECS')
- 后跟 per-LOD 恢复统计的二进制编码
"""

import struct
from dataclasses import dataclass

from common.fec_stats import FECStatsReport  # noqa: F401 — re-export from shared module

# ---- 常量 ----

RTCP_PT_SR = 200
RTCP_PT_RR = 201
RTP_VERSION = 2

RTCP_HEADER_SIZE = 8          # 固定 RTCP 首部 (不包含 SSRC 列表)
RTCP_RR_BLOCK_SIZE = 24       # 单个 Report Block 大小
RTCP_MIN_PACKET = RTCP_HEADER_SIZE + 4 + RTCP_RR_BLOCK_SIZE  # 32 bytes

# FECStats 扩展类型标记
FEC_STATS_MAGIC = 0x4645_4353  # b'FECS'


# ---- SR / RR 数据结构 ----

@dataclass
class RTCPSenderReport:
    """RTCP Sender Report (SR, PT=200), RFC 3550 §6.4.1"""
    ssrc: int              # 32-bit
    ntp_timestamp_msw: int # 32-bit, NTP 时间戳高 32 位
    ntp_timestamp_lsw: int # 32-bit, NTP 时间戳低 32 位
    rtp_timestamp: int     # 32-bit, 对应的 RTP 时间戳
    packet_count: int      # 32-bit, 累计发送包数
    octet_count: int       # 32-bit, 累计发送字节数
    report_blocks: list['RTCPReportBlock'] | None = None


@dataclass
class RTCPReportBlock:
    """RTCP Report Block, RFC 3550 §6.4.2"""
    ssrc: int                # 32-bit, 被报告的 SSRC
    fraction_lost: int       # 8-bit, 丢包率 × 256
    cumulative_lost: int     # 24-bit, 累计丢包数
    ext_highest_seq: int     # 32-bit, 扩展最高序列号
    interarrival_jitter: int # 32-bit, 到达间隔抖动
    last_sr: int             # 32-bit, 最近收到的 SR 的 NTP 时间戳中段
    delay_since_last_sr: int # 32-bit, 自收到 last SR 以来的延迟 (1/65536 s)


@dataclass
class RTCPReceiverReport:
    """RTCP Receiver Report (RR, PT=201), RFC 3550 §6.4.2"""
    ssrc: int
    report_blocks: list[RTCPReportBlock]


# ---- 序列化 ----

def pack_rtcp_sr(sr: RTCPSenderReport) -> bytes:
    """序列化 RTCP Sender Report"""
    # 20 bytes sender info + N × 24 bytes report blocks
    num_blocks = len(sr.report_blocks) if sr.report_blocks else 0
    length_words = (RTCP_HEADER_SIZE + 20 + num_blocks * RTCP_RR_BLOCK_SIZE) // 4
    data = bytearray()
    # Common header: RC = num_blocks
    first = (RTP_VERSION << 6) | (0 << 5) | num_blocks
    data.extend(struct.pack('!BBH', first, RTCP_PT_SR, length_words))
    # SSRC
    data.extend(struct.pack('!I', sr.ssrc))
    # Sender Info (20 bytes)
    data.extend(struct.pack(
        '!IIIII',
        sr.ntp_timestamp_msw, sr.ntp_timestamp_lsw,
        sr.rtp_timestamp, sr.packet_count, sr.octet_count,
    ))
    # Report Blocks
    if sr.report_blocks:
        for rb in sr.report_blocks:
            data.extend(_pack_report_block(rb))
    return bytes(data)


def parse_rtcp_sr(data: bytes) -> RTCPSenderReport:
    """反序列化 RTCP Sender Report"""
    if len(data) < RTCP_HEADER_SIZE + 4 + 20:
        raise ValueError(f'RTCP SR too short: {len(data)} bytes')
    first, pt, _ = struct.unpack('!BBH', data[:4])
    num_blocks = first & 0x1F
    ssrc = struct.unpack('!I', data[4:8])[0]
    ntp_msw, ntp_lsw, rtp_ts, pkt_cnt, oct_cnt = struct.unpack(
        '!IIIII', data[8:28],
    )
    blocks = []
    offset = 28
    for _ in range(num_blocks):
        blocks.append(_parse_report_block(data[offset:offset + RTCP_RR_BLOCK_SIZE]))
        offset += RTCP_RR_BLOCK_SIZE
    return RTCPSenderReport(
        ssrc=ssrc, ntp_timestamp_msw=ntp_msw, ntp_timestamp_lsw=ntp_lsw,
        rtp_timestamp=rtp_ts, packet_count=pkt_cnt, octet_count=oct_cnt,
        report_blocks=blocks,
    )


def pack_rtcp_rr(rr: RTCPReceiverReport) -> bytes:
    """序列化 RTCP Receiver Report"""
    num_blocks = len(rr.report_blocks)
    length_words = (RTCP_HEADER_SIZE + num_blocks * RTCP_RR_BLOCK_SIZE) // 4
    first = (RTP_VERSION << 6) | (0 << 5) | num_blocks
    data = bytearray()
    data.extend(struct.pack('!BBH', first, RTCP_PT_RR, length_words))
    data.extend(struct.pack('!I', rr.ssrc))
    for rb in rr.report_blocks:
        data.extend(_pack_report_block(rb))
    return bytes(data)


def parse_rtcp_rr(data: bytes) -> RTCPReceiverReport:
    """反序列化 RTCP Receiver Report"""
    if len(data) < RTCP_HEADER_SIZE + 4:
        raise ValueError(f'RTCP RR too short: {len(data)} bytes')
    first, pt, _ = struct.unpack('!BBH', data[:4])
    num_blocks = first & 0x1F
    ssrc = struct.unpack('!I', data[4:8])[0]
    blocks = []
    offset = 8
    for _ in range(num_blocks):
        if offset + RTCP_RR_BLOCK_SIZE > len(data):
            break
        blocks.append(_parse_report_block(data[offset:offset + RTCP_RR_BLOCK_SIZE]))
        offset += RTCP_RR_BLOCK_SIZE
    return RTCPReceiverReport(ssrc=ssrc, report_blocks=blocks)


def _pack_report_block(rb: RTCPReportBlock) -> bytes:
    """序列化单个 Report Block"""
    cumulative = rb.cumulative_lost & 0xFFFFFF
    fraction_cum = (rb.fraction_lost << 24) | cumulative
    return struct.pack(
        '!IIIII',
        rb.ssrc, fraction_cum, rb.ext_highest_seq,
        rb.interarrival_jitter, rb.last_sr, rb.delay_since_last_sr,
    )


def _parse_report_block(data: bytes) -> RTCPReportBlock:
    """反序列化单个 Report Block"""
    ssrc, fc, ext_seq, jitter, last_sr, dlsr = struct.unpack('!IIIII', data)
    fraction_lost = (fc >> 24) & 0xFF
    cumulative_lost = fc & 0xFFFFFF
    return RTCPReportBlock(
        ssrc=ssrc, fraction_lost=fraction_lost, cumulative_lost=cumulative_lost,
        ext_highest_seq=ext_seq, interarrival_jitter=jitter,
        last_sr=last_sr, delay_since_last_sr=dlsr,
    )


# ---- FECStats 扩展编码 ----

def pack_fec_stats_report(stats: FECStatsReport) -> bytes:
    """将 FECStats 编码为 RTCP 应用特定扩展的二进制格式

    格式:
        [magic:4B] [num_lods:1B] [reserved:3B]
        [total_0:4B] ... [total_N:4B]
        [lost_0:4B] ... [lost_N:4B]
        [recovered_0:4B] ... [recovered_N:4B]
        [unrecovered_0:4B] ... [unrecovered_N:4B]
    """
    n = len(stats.per_lod_total)
    buf = bytearray()
    buf.extend(struct.pack('!I', FEC_STATS_MAGIC))
    buf.append(n & 0xFF)
    buf.extend(b'\x00' * 3)  # reserved
    for arr in [stats.per_lod_total, stats.per_lod_lost,
                stats.per_lod_recovered, stats.per_lod_unrecovered]:
        fmt = f'!{n}I'
        buf.extend(struct.pack(fmt, *arr))
    return bytes(buf)


def parse_fec_stats_report(data: bytes) -> FECStatsReport | None:
    """从二进制数据解析 FECStats 扩展"""
    if len(data) < 8:
        return None
    magic = struct.unpack('!I', data[:4])[0]
    if magic != FEC_STATS_MAGIC:
        return None
    n = data[4]
    if n == 0:
        return FECStatsReport([], [], [], [])
    expected_len = 8 + n * 16  # 4 arrays × 4 bytes × n
    if len(data) < expected_len:
        return None
    offset = 8
    stride = n * 4
    per_lod_total = list(struct.unpack(f'!{n}I', data[offset:offset + stride])); offset += stride
    per_lod_lost = list(struct.unpack(f'!{n}I', data[offset:offset + stride])); offset += stride
    per_lod_recovered = list(struct.unpack(f'!{n}I', data[offset:offset + stride])); offset += stride
    per_lod_unrecovered = list(struct.unpack(f'!{n}I', data[offset:offset + stride]))
    return FECStatsReport(
        per_lod_total=per_lod_total,
        per_lod_lost=per_lod_lost,
        per_lod_recovered=per_lod_recovered,
        per_lod_unrecovered=per_lod_unrecovered,
    )
