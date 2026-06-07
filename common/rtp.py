"""
RTP 固定首部 (RFC 3550, 12 bytes)

序列化/反序列化 RTP 包格式。
"""
import struct

RTP_HEADER_SIZE = 12


class RTPPacket:
    """RTP 固定首部 (RFC 3550, 12 bytes)"""

    def __init__(
        self,
        marker: int = 0,
        payload_type: int = 96,
        sequence_number: int = 0,
        timestamp: int = 0,
        ssrc: int = 0,
        payload: bytes = b'',
    ):
        self.version = 2
        self.padding = 0
        self.extension = 0
        self.csrc_count = 0
        self.marker = marker & 0x01
        self.payload_type = payload_type & 0x7F
        self.sequence_number = sequence_number & 0xFFFF
        self.timestamp = timestamp & 0xFFFFFFFF
        self.ssrc = ssrc & 0xFFFFFFFF
        self.payload = payload

    def serialize(self) -> bytes:
        first = (
            (self.version << 6)
            | (self.padding << 5)
            | (self.extension << 4)
            | self.csrc_count
        )
        second = (self.marker << 7) | self.payload_type
        return (
            struct.pack(
                '!BBHLL',
                first,
                second,
                self.sequence_number,
                self.timestamp,
                self.ssrc,
            )
            + self.payload
        )

    @classmethod
    def parse(cls, data: bytes) -> 'RTPPacket':
        if len(data) < 12:
            raise ValueError(f'RTP packet too short: {len(data)} bytes')
        first = data[0]
        second = data[1]
        version = (first >> 6) & 0x03
        if version != 2:
            raise ValueError(f'Invalid RTP version: {version}')
        seq = struct.unpack('!H', data[2:4])[0]
        ts = struct.unpack('!L', data[4:8])[0]
        ssrc = struct.unpack('!L', data[8:12])[0]
        marker = (second >> 7) & 0x01
        payload_type = second & 0x7F
        return cls(
            marker=marker,
            payload_type=payload_type,
            sequence_number=seq,
            timestamp=ts,
            ssrc=ssrc,
            payload=data[12:],
        )

    def __repr__(self) -> str:
        return (
            f'RTPPacket(seq={self.sequence_number}, ts={self.timestamp}, '
            f'marker={self.marker}, pt={self.payload_type}, '
            f'ssrc={self.ssrc:#x}, payload={len(self.payload)}B)'
        )
