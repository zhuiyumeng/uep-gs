"""
共享应用层载荷格式 — PayloadHeader + SceneMeta + Dummy Gaussian + 常量

复用方：transport, transport_fec
"""
import struct
import json

PAYLOAD_HEADER_SIZE = 8

UNIT_SCENE_META = 1
UNIT_GAUSSIAN_DATA = 2
UNIT_END_OF_FRAME = 3


class PayloadHeader:
    """载荷分片首部 (8 bytes)"""

    __slots__ = ('start', 'end', 'lod', 'unit_type', 'fragment_offset')

    def __init__(
        self,
        start: int = 0,
        end: int = 0,
        lod: int = 0,
        unit_type: int = 0,
        fragment_offset: int = 0,
    ):
        self.start = start & 0x01
        self.end = end & 0x01
        self.lod = lod & 0x0F
        self.unit_type = unit_type & 0xFFFF
        self.fragment_offset = fragment_offset & 0xFFFFFFFF

    def serialize(self) -> bytes:
        first_byte = (self.start << 7) | (self.end << 6) | self.lod
        return struct.pack(
            '!BBHL',
            first_byte,
            0,  # reserved byte
            self.unit_type,
            self.fragment_offset,
        )

    @classmethod
    def parse(cls, data: bytes) -> 'PayloadHeader':
        if len(data) < 8:
            raise ValueError(f'Payload header too short: {len(data)} bytes')
        first_byte = data[0]
        start = (first_byte >> 7) & 0x01
        end = (first_byte >> 6) & 0x01
        lod = first_byte & 0x0F
        unit_type = struct.unpack('!H', data[2:4])[0]
        frag_off = struct.unpack('!L', data[4:8])[0]
        return cls(
            start=start, end=end, lod=lod,
            unit_type=unit_type, fragment_offset=frag_off,
        )

    def __repr__(self) -> str:
        return (
            f'PayloadHeader(type={self.unit_type}, S={self.start}, E={self.end}, '
            f'LOD={self.lod}, offset={self.fragment_offset})'
        )


class SceneMeta:
    """帧元数据，由首包携带"""

    __slots__ = (
        'sh_degree', 'num_lods', 'lod_sizes',
        'total_gaussians', 'total_bytes', 'gaussian_stride',
        'dummy_gaussian_hex',
    )

    def __init__(
        self,
        sh_degree: int,
        num_lods: int,
        lod_sizes: list[int],
        total_gaussians: int,
        total_bytes: int,
        gaussian_stride: int,
        dummy_gaussian_hex: str,
    ):
        self.sh_degree = sh_degree
        self.num_lods = num_lods
        self.lod_sizes = lod_sizes
        self.total_gaussians = total_gaussians
        self.total_bytes = total_bytes
        self.gaussian_stride = gaussian_stride
        self.dummy_gaussian_hex = dummy_gaussian_hex

    def to_json(self) -> bytes:
        d = {s: getattr(self, s) for s in self.__slots__}
        return json.dumps(d, ensure_ascii=False).encode('utf-8')

    @classmethod
    def from_json(cls, data: bytes) -> 'SceneMeta':
        d = json.loads(data.decode('utf-8'))
        return cls(**d)

    def __repr__(self) -> str:
        return (
            f'SceneMeta(gaussians={self.total_gaussians}, '
            f'lods={self.num_lods}, bytes={self.total_bytes}, '
            f'sh={self.sh_degree})'
        )


def make_dummy_gaussian(sh_degree: int) -> list[float]:
    """构造安全不可见 Gaussian 模板

    - 位置/颜色/不透明度全零 → 渲染时不可见
    - 用于填充丢包缺失数据，保证帧结构完整
    """
    K = 3 * ((sh_degree + 1) ** 2 - 1)
    return (
        [0.0, 0.0, 0.0]          # xyz
        + [0.0, 0.0, 0.0]        # nx, ny, nz (padding)
        + [0.0, 0.0, 0.0]        # f_dc
        + [0.0] * K               # f_rest
        + [-100.0]                # opacity → sigmoid → 0
        + [-100.0, -100.0, -100.0]  # scale → exp → 0
        + [1.0, 0.0, 0.0, 0.0]   # rot → unit quaternion
    )
