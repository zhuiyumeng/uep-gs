#!/usr/bin/env python3
"""
一站式 3DGS 流式传输实验脚本

完整链路：K-means LOD 分包 → RTP/UDP 传输 (FEC vs 无FEC) → 接收重组 → 合并基线 → 渲染评估

用法:
    python run_experiment.py \
      --ply /root/gaussian-splatting/output/room/point_cloud/iteration_30000/point_cloud.ply \
      --imp_score /root/gaussian-splatting/output/room/imp_score.npz \
      --orig_model /root/gaussian-splatting/output/room \
      --source_path /root/dataset/room \
      --python /root/miniconda3/envs/gs/bin/python \
      -k 4 --loss 0.05 --seed 42
"""

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime

# ---- 项目根目录 ----
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
#  工具函数
# ============================================================

def run_cmd(cmd: list[str], log_file: str, env: dict | None = None,
            cwd: str | None = None, timeout: float | None = None
            ) -> subprocess.CompletedProcess:
    """运行命令，stdout/stderr 同时输出到日志文件和控制台。"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, 'w', encoding='utf-8') as fh:
        fh.write(f'[CMD] {" ".join(cmd)}\n\n')
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env or os.environ.copy(),
            cwd=cwd or PROJECT_DIR,
            timeout=timeout,
        )
        fh.write(proc.stdout.decode('utf-8', errors='replace'))
        if proc.returncode != 0:
            fh.write(f'\n[EXIT] returncode={proc.returncode}\n')
    return proc


def popen_bg(cmd: list[str], log_file: str, env: dict | None = None,
             cwd: str | None = None) -> subprocess.Popen:
    """后台启动进程，输出重定向到日志文件。"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    fh = open(log_file, 'w', encoding='utf-8')
    fh.write(f'[CMD] {" ".join(cmd)}\n\n')
    fh.flush()
    proc = subprocess.Popen(
        cmd,
        stdout=fh,
        stderr=subprocess.STDOUT,
        env=env or os.environ.copy(),
        cwd=cwd or PROJECT_DIR,
        preexec_fn=os.setsid,
    )
    return proc


def kill_bg(proc: subprocess.Popen, timeout: float = 5.0):
    """优雅终止后台进程。"""
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=timeout)
    except (subprocess.TimeoutExpired, ProcessLookupError):
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass


def wait_udp_port(host: str, port: int, deadline: float) -> bool:
    """轮询直到 UDP 端口被占用（说明服务端已就绪）。"""
    while time.time() < deadline:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.bind((host, port))
            s.close()
            time.sleep(0.2)
        except OSError:
            # 端口已被占用 → 服务端已启动
            return True
    return False


def find_free_port(start: int = 5005) -> int:
    """找一个空闲 UDP 端口。"""
    port = start
    while port < start + 100:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.bind(('127.0.0.1', port))
            s.close()
            return port
        except OSError:
            port += 1
    raise RuntimeError(f'No free UDP port found starting from {start}')


# ============================================================
#  各阶段实现
# ============================================================

def stage1_kmeans(args, exp_dir: str, logs_dir: str, env: dict) -> int:
    """阶段1：K-means LOD 分包。"""
    lod_dir = os.path.join(exp_dir, 'lod')
    os.makedirs(lod_dir, exist_ok=True)
    out_prefix = os.path.join(lod_dir, args.prefix)

    k_arg = [str(args.k)] if args.k is not None else []
    cmd = [
        args.python,
        os.path.join(PROJECT_DIR, 'package_network_kmeans.py'),
        '--ply', args.ply,
        '--imp_score', args.imp_score,
        '--out_prefix', out_prefix,
    ] + (['-k', str(args.k)] if args.k is not None else [])

    print(f'[阶段1] K-means LOD 分包 (K={args.k or "auto"})')
    print(f'  输入: {args.ply}')
    print(f'  输出: {lod_dir}/')

    proc = run_cmd(cmd, os.path.join(logs_dir, 'stage1_kmeans.log'), env=env)
    if proc.returncode != 0:
        print(f'  失败! 返回码={proc.returncode}，详见日志。')
        return proc.returncode

    # 统计输出文件
    found = sorted(
        f for f in os.listdir(lod_dir)
        if f.startswith(args.prefix) and f.endswith('.ply')
    )
    if not found:
        print('  失败! 未生成任何 LOD PLY 文件。')
        return 1

    print(f'  生成 {len(found)} 个 LOD 文件:')
    for f in found:
        sz = os.path.getsize(os.path.join(lod_dir, f))
        print(f'    {f}  ({sz / 1024 / 1024:.1f} MB)')
    return 0


def stage23_trial(args, exp_dir: str, logs_dir: str, env: dict,
                  label: str, port: int, fec: bool) -> str | None:
    """阶段2+3：一次传输试验（服务端后台 + 客户端前台）。

    Args:
        label: 试验名称 (如 'fec' 或 'nofec')
        port: UDP 端口号
        fec: 是否启用 FEC

    Returns:
        接收到的 PLY 文件路径，失败返回 None
    """
    lod_dir = os.path.join(exp_dir, 'lod')
    received_dir = os.path.join(exp_dir, 'received')
    os.makedirs(received_dir, exist_ok=True)
    output_ply = os.path.join(received_dir, f'{label}_received.ply')

    fec_flag = [] if fec else ['--no-fec']
    fec_label = 'FEC ON' if fec else 'FEC OFF'

    # 服务端命令
    svr_cmd = [
        args.python, '-m', 'transport_fec.server',
        '--lod_dir', lod_dir,
        '--prefix', args.prefix,
        '--sh_degree', str(args.sh_degree),
        '--host', args.host,
        '--port', str(port),
        '--oneshot',
    ] + fec_flag

    # 客户端命令
    cli_cmd = [
        args.python, '-m', 'transport_fec.client',
        '--host', args.host,
        '--port', str(port),
        '--output', output_ply,
        '--loss', str(args.loss),
        '--seed', str(args.seed),
        '--timeout', str(args.timeout),
    ] + fec_flag

    print(f'\n[阶段2+3] {label.upper()} 传输试验 ({fec_label})')
    print(f'  端口: {port}, 丢包率: {args.loss}, 种子: {args.seed}')
    print(f'  输出: {output_ply}')

    # 启动服务端
    svr_log = os.path.join(logs_dir, f'stage2_server_{label}.log')
    svr_proc = popen_bg(svr_cmd, svr_log, env=env)

    # 等待服务端端口就绪
    deadline = time.time() + 10.0
    if not wait_udp_port(args.host, port, deadline):
        print('  失败! 服务端未在 10 秒内就绪。')
        kill_bg(svr_proc)
        return None
    time.sleep(0.3)

    # 运行客户端
    cli_log = os.path.join(logs_dir, f'stage3_client_{label}.log')
    cli_proc = run_cmd(cli_cmd, cli_log, env=env, timeout=args.timeout + 60)

    # 停止服务端
    kill_bg(svr_proc)

    if cli_proc.returncode != 0:
        print(f'  客户端异常 (返回码={cli_proc.returncode})，详见日志。')
        # 继续——超时也会 flush 写出部分帧

    if os.path.exists(output_ply):
        sz = os.path.getsize(output_ply)
        print(f'  接收完成: {output_ply} ({sz / 1024 / 1024:.1f} MB)')
        return output_ply
    else:
        print(f'  失败! 未生成接收文件。')
        return None


def stage4_merge(args, exp_dir: str, logs_dir: str, env: dict) -> str | None:
    """阶段4：无损合并 LOD 文件（参考基线）。"""
    lod_dir = os.path.join(exp_dir, 'lod')
    merged_dir = os.path.join(exp_dir, 'merged')
    os.makedirs(merged_dir, exist_ok=True)
    out_ply = os.path.join(merged_dir, 'merged_full.ply')

    cmd = [
        args.python,
        os.path.join(PROJECT_DIR, 'merge_ply.py'),
        '--in_dir', lod_dir,
        '--out_file', out_ply,
    ]

    print(f'\n[阶段4] 无损合并基线')
    print(f'  输入: {lod_dir}/')
    print(f'  输出: {out_ply}')

    proc = run_cmd(cmd, os.path.join(logs_dir, 'stage4_merge.log'), env=env)
    if proc.returncode != 0:
        print(f'  失败! 返回码={proc.returncode}')
        return None

    sz = os.path.getsize(out_ply)
    print(f'  合并完成: ({sz / 1024 / 1024:.1f} MB)')
    return out_ply


def stage5_eval(args, exp_dir: str, logs_dir: str, env: dict,
                ply_path: str, label: str) -> dict | None:
    """阶段5：渲染评估单个 PLY 文件。

    Returns:
        {'psnr': ..., 'ssim': ..., 'lpips': ...} 或 None
    """
    eval_dir = os.path.join(exp_dir, f'eval_{label}')
    cmd = [
        args.python,
        os.path.join(PROJECT_DIR, 'auto_eval.py'),
        '--ply', ply_path,
        '--orig_model', args.orig_model,
        '--new_model', eval_dir,
        '--source_path', args.source_path,
        '--iteration', str(args.iteration),
    ]

    print(f'\n[阶段5] 渲染评估: {label} ({ply_path})')
    print(f'  工作区: {eval_dir}')

    log_file = os.path.join(logs_dir, f'stage5_eval_{label}.log')
    proc = run_cmd(cmd, log_file, env=env, timeout=args.eval_timeout)

    if proc.returncode != 0:
        print(f'  评估异常 (返回码={proc.returncode})，详见日志。')

    # 读取 results.json
    results_json = os.path.join(eval_dir, 'results.json')
    if not os.path.exists(results_json):
        print(f'  失败! 未找到 results.json')
        return None

    with open(results_json, 'r', encoding='utf-8') as fh:
        data = json.load(fh)

    # gaussian-splatting results.json 格式: {"ours_30000": {"SSIM": ..., "PSNR": ..., "LPIPS": ...}}
    first_key = next(iter(data), None)
    if first_key is None:
        print('  失败! results.json 为空')
        return None

    metrics = data[first_key]
    psnr = metrics.get('PSNR', float('nan'))
    ssim = metrics.get('SSIM', float('nan'))
    lpips = metrics.get('LPIPS', float('nan'))
    print(f'  PSNR={psnr:.2f}  SSIM={ssim:.4f}  LPIPS={lpips:.4f}')
    return {'psnr': psnr, 'ssim': ssim, 'lpips': lpips}


# ============================================================
#  主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='3DGS 流式传输完整实验 (FEC vs 无FEC)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # 数据路径
    parser.add_argument('--ply', required=True, help='源 PLY 文件路径')
    parser.add_argument('--imp_score', required=True, help='重要性得分 npz 文件')
    parser.add_argument('--orig_model', required=True, help='原始 3DGS 训练输出目录')
    parser.add_argument('--source_path', required=True, help='数据集路径')
    # 参数
    parser.add_argument('--sh_degree', type=int, default=3, help='球谐阶数')
    parser.add_argument('-k', type=int, default=None, help='LOD 层级数 (默认自动检测)')
    parser.add_argument('--loss', type=float, default=0.05, help='模拟丢包率')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--host', default='127.0.0.1', help='服务端/客户端地址')
    parser.add_argument('--port', type=int, default=0, help='UDP 起始端口 (0=自动)')
    parser.add_argument('--timeout', type=float, default=30.0, help='客户端接收超时秒数')
    parser.add_argument('--eval_timeout', type=float, default=3600.0, help='评估超时秒数')
    parser.add_argument('--iteration', type=int, default=30000, help='评估迭代次数')
    # 运行选项
    parser.add_argument('--python', default=sys.executable, help='Python 解释器路径')
    parser.add_argument('--exp_dir', default='experiments', help='实验根目录')
    parser.add_argument('--exp_name', default=None, help='实验子目录名 (默认时间戳)')
    parser.add_argument('--prefix', default='model', help='LOD 文件前缀')
    parser.add_argument('--skip_kmeans', action='store_true', help='跳过阶段1')
    parser.add_argument('--skip_merge', action='store_true', help='跳过阶段4')
    parser.add_argument('--skip_eval', action='store_true', help='跳过阶段5')
    parser.add_argument('--fec_only', action='store_true', help='仅测 FEC')
    parser.add_argument('--nofec_only', action='store_true', help='仅测无FEC')
    parser.add_argument('--continue_on_error', action='store_true', help='出错后继续')

    args = parser.parse_args()

    # ---- 环境与工作目录 ----
    python_ok = shutil.which(args.python) or os.path.exists(args.python)
    if not python_ok:
        print(f'错误: Python 路径无效: {args.python}')
        sys.exit(1)

    env = os.environ.copy()
    env['PYTHONPATH'] = PROJECT_DIR
    # 确保 conda python 的 bin 在 PATH 最前面
    py_bin_dir = os.path.dirname(args.python)
    env['PATH'] = py_bin_dir + os.pathsep + env.get('PATH', '')

    # ---- 实验目录 ----
    if args.exp_name:
        exp_dir = os.path.join(args.exp_dir, args.exp_name)
    else:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_dir = os.path.join(args.exp_dir, f'exp_{ts}')
    os.makedirs(exp_dir, exist_ok=True)

    logs_dir = os.path.join(exp_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)

    # 保存配置快照
    config_path = os.path.join(exp_dir, 'config.json')
    config = {k: str(v) if isinstance(v, (str, bool, int, float, type(None))) else repr(v)
              for k, v in vars(args).items()}
    with open(config_path, 'w', encoding='utf-8') as fh:
        json.dump(config, fh, indent=2, ensure_ascii=False)

    # ---- 分配端口 ----
    if args.port == 0:
        args.port = find_free_port(5005)
        print(f'自动分配 UDP 端口: FEC={args.port}, NoFEC={args.port + 1}')
    fec_port = args.port
    nofec_port = args.port + 1

    print('=' * 60)
    print(f'3DGS 流式传输实验')
    print(f'实验目录: {exp_dir}')
    print(f'Python:    {args.python}')
    print(f'PLY 源:   {args.ply}')
    print(f'参数:     K={args.k or "auto"}, loss={args.loss}, seed={args.seed}')
    print(f'端口:     FEC={fec_port}, NoFEC={nofec_port}')
    print('=' * 60)

    failed = False

    # ---- 阶段1: K-means LOD 分包 ----
    if not args.skip_kmeans:
        rc = stage1_kmeans(args, exp_dir, logs_dir, env)
        if rc != 0 and not args.continue_on_error:
            print('\n阶段1 失败，实验终止。')
            sys.exit(rc)
        elif rc != 0:
            failed = True
    else:
        print('[阶段1] 跳过 (--skip_kmeans)')

    # ---- 阶段2+3: 传输试验 ----
    received = {}  # label -> ply_path

    if not args.nofec_only:
        path = stage23_trial(args, exp_dir, logs_dir, env, 'fec', fec_port, fec=True)
        if path:
            received['fec'] = path
        elif not args.continue_on_error:
            print('\nFEC 传输失败，实验终止。')
            sys.exit(1)
        else:
            failed = True

    if not args.fec_only:
        # 更新 seed 确保丢包模式一致但独立
        path = stage23_trial(args, exp_dir, logs_dir, env, 'nofec', nofec_port, fec=False)
        if path:
            received['nofec'] = path
        elif not args.continue_on_error:
            print('\nNoFEC 传输失败，实验终止。')
            sys.exit(1)
        else:
            failed = True

    # ---- 阶段4: 无损合并基线 ----
    merged_ply = None
    if not args.skip_merge:
        merged_ply = stage4_merge(args, exp_dir, logs_dir, env)
        if merged_ply:
            received['merged'] = merged_ply
        elif not args.continue_on_error:
            print('\n阶段4 失败，实验终止。')
            sys.exit(1)
        else:
            failed = True
    else:
        print('\n[阶段4] 跳过 (--skip_merge)')

    # ---- 阶段5: 渲染评估 ----
    all_metrics = {}
    if not args.skip_eval:
        for label, ply_path in received.items():
            m = stage5_eval(args, exp_dir, logs_dir, env, ply_path, label)
            if m:
                all_metrics[label] = m
            elif not args.continue_on_error:
                print(f'\n阶段5 ({label}) 评估失败，实验终止。')
                sys.exit(1)
            else:
                failed = True
    else:
        print('\n[阶段5] 跳过 (--skip_eval)')

    # ---- 结果汇总 ----
    print('\n' + '=' * 60)
    print('实验结果汇总')
    print('=' * 60)
    print(f'K={args.k or "auto"}, loss={args.loss:.1%}, seed={args.seed}')
    print()

    # 打印指标表
    header = f"{'':>20s}  {'PSNR':>8s}  {'SSIM':>8s}  {'LPIPS':>8s}"
    print(header)
    print('-' * len(header))

    labels_display = [('merged', 'Merged (无损)'), ('fec', 'FEC'), ('nofec', 'No-FEC')]
    for key, name in labels_display:
        if key in all_metrics:
            m = all_metrics[key]
            print(f'{name:>20s}  {m["psnr"]:8.2f}  {m["ssim"]:8.4f}  {m["lpips"]:8.4f}')

    # 计算增益
    if 'fec' in all_metrics and 'nofec' in all_metrics:
        fm = all_metrics['fec']
        nm = all_metrics['nofec']
        d_psnr = fm['psnr'] - nm['psnr']
        d_ssim = fm['ssim'] - nm['ssim']
        d_lpips = fm['lpips'] - nm['lpips']
        print('-' * len(header))
        print(f'{"FEC 增益":>20s}  {d_psnr:+8.2f}  {d_ssim:+8.4f}  {d_lpips:+8.4f}')

    if 'fec' in all_metrics and 'merged' in all_metrics:
        fm = all_metrics['fec']
        mm = all_metrics['merged']
        d_psnr = fm['psnr'] - mm['psnr']
        d_ssim = fm['ssim'] - mm['ssim']
        d_lpips = fm['lpips'] - mm['lpips']
        print(f'{"FEC vs Merged":>20s}  {d_psnr:+8.2f}  {d_ssim:+8.4f}  {d_lpips:+8.4f}')

    # 写入 summary.json
    summary = {
        'experiment': exp_dir,
        'config': config,
        'metrics': all_metrics,
    }
    if 'fec' in all_metrics and 'nofec' in all_metrics:
        summary['deltas'] = {
            'fec_vs_nofec': {
                'psnr_delta': all_metrics['fec']['psnr'] - all_metrics['nofec']['psnr'],
                'ssim_delta': all_metrics['fec']['ssim'] - all_metrics['nofec']['ssim'],
                'lpips_delta': all_metrics['fec']['lpips'] - all_metrics['nofec']['lpips'],
            }
        }

    summary_path = os.path.join(exp_dir, 'summary.json')
    with open(summary_path, 'w', encoding='utf-8') as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False, default=str)
    print(f'\n结果已保存: {summary_path}')

    if failed:
        print('警告: 部分阶段失败，详见日志。')
        sys.exit(2)
    else:
        print('实验完成。')


if __name__ == '__main__':
    main()
