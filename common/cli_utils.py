"""
CLI 工具函数 — 被 transport 和 transport_fec 的 server.py / client.py 共享
"""
import os


def find_lod_plys(lod_dir: str, prefix: str) -> list[str]:
    """按前缀排序查找 LOD PLY 文件"""
    files = sorted(
        os.path.join(lod_dir, f)
        for f in os.listdir(lod_dir)
        if f.startswith(prefix) and f.endswith('.ply')
    )
    return files


def _fmt_size(n: int) -> str:
    """人类可读的文件大小格式化"""
    for unit in ('B', 'KB', 'MB', 'GB'):
        if n < 1024:
            return f'{n:.1f} {unit}'
        n /= 1024
    return f'{n:.1f} TB'


def add_shared_server_args(parser):
    """向 ArgumentParser 添加 transport 和 transport_fec 共享的服务端参数"""
    parser.add_argument(
        '--lod_dir', required=True,
        help='LOD PLY 文件所在目录',
    )
    parser.add_argument(
        '--prefix', default='model',
        help='LOD 文件名前缀',
    )
    parser.add_argument('--sh_degree', type=int, default=3, help='球谐阶数 (2 或 3)')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=5005)
    parser.add_argument('--fps', type=int, default=30, help='发送帧率')
    parser.add_argument('--mtu', type=int, default=1400, help='MTU 安全阈值')
    parser.add_argument('--loop', action='store_true', default=True, help='循环发送')
    parser.add_argument('--oneshot', action='store_true', default=False, help='每客户端仅一帧')
    parser.add_argument('--pace_ms', type=float, default=0.1, help='组间延迟 (ms)')
    parser.add_argument('--pace_every', type=int, default=10, help='每 N 包延迟一次')


def add_shared_client_args(parser):
    """向 ArgumentParser 添加 transport 和 transport_fec 共享的客户端参数"""
    parser.add_argument('--host', default='127.0.0.1', help='服务端地址')
    parser.add_argument('--port', type=int, default=5005, help='服务端端口')
    parser.add_argument('--output', default='received.ply', help='输出 PLY 路径')
    parser.add_argument('--timeout', type=float, default=10.0, help='接收超时秒数')
    parser.add_argument('--loss', type=float, default=0.0, help='模拟丢包率 (0.0~1.0)')
    parser.add_argument('--seed', type=int, default=None, help='随机种子')
    parser.add_argument('--flush_on_timeout', action='store_true', default=True, help='超时时强制完成帧')
