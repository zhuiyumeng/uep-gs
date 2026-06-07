"""
PLY 文件工具函数 — 被 transport 和 transport_fec 共享
"""
from common.payload import SceneMeta


def build_ply_header(num_vertices: int, sh_degree: int) -> bytes:
    """构建 PLY 文件头"""
    prop_names = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    prop_names += [f'f_dc_{i}' for i in range(3)]
    K = 3 * ((sh_degree + 1) ** 2 - 1)
    prop_names += [f'f_rest_{i}' for i in range(K)]
    prop_names += ['opacity']
    prop_names += [f'scale_{i}' for i in range(3)]
    prop_names += [f'rot_{i}' for i in range(4)]

    lines = [
        'ply',
        'format binary_little_endian 1.0',
        f'element vertex {num_vertices}',
    ]
    for p in prop_names:
        lines.append(f'property float {p}')
    lines.append('end_header')
    return '\n'.join(lines).encode('ascii') + b'\n'


def write_ply(output_path: str, frame_data: bytes, meta: SceneMeta):
    """将重组后的帧数据写为 PLY 文件"""
    num_vertices = len(frame_data) // (meta.gaussian_stride * 4)
    header = build_ply_header(num_vertices, meta.sh_degree)
    with open(output_path, 'wb') as f:
        f.write(header)
        f.write(frame_data)
