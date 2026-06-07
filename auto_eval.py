import os
import subprocess
import argparse
import sys
import shutil

"""
这个脚本的目的是为了在不干扰原始训练输出的前提下，评估微调后的 PLY 文件。它通过以下步骤实现：
1. 从原始训练输出目录中提取必要的配置文件（如 cfg_args 和 cameras.json）。
2. 创建一个新的独立工作区，并将这些配置文件复制到新工作区中。
3. 将微调后的 PLY 文件放入新工作区的指定位置（point_cloud/iteration_30000/point_cloud.ply）。
4. 在新的工作区中执行渲染和评估，确保所有结果都保存在新工作区中，避免对原始训练输出造成任何影响。
使用示例：
python auto_eval.py --ply /data3/gaussian-splatting/pruned.ply --orig_model /data3/gaussian-splatting/output --new_model /data3/gaussian-splatting/output_uep
请确保在运行此脚本前，已经正确安装了所需的 Python 环境，并且 train.py、render.py 和 metrics.py 脚本位于当前工作目录下。
"""

# 获取 gaussian-splatting 的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
gs_dir = os.path.abspath(os.path.join(current_dir, '..', 'gaussian-splatting'))

def setup_workspace(orig_workspace, new_workspace):
    """
    从原工作区复制配置文件到新工作区
    :param orig_workspace: 原版 3DGS 训练输出目录
    :param new_workspace: 新的评估工作区目录
    """
    orig_workspace = os.path.abspath(orig_workspace)
    new_workspace = os.path.abspath(new_workspace)
    
    # 检查原工作区是否存在 cfg_args
    orig_cfg = os.path.join(orig_workspace, "cfg_args")
    if not os.path.exists(orig_cfg):
        print(f"❌ 错误: 原工作区缺少 cfg_args 文件 ({orig_cfg})。")
        print("请确保你提供的模型路径是 3DGS 训练生成的完整输出目录。")
        sys.exit(1)

    # 创建新工作区
    print(f"📁 正在初始化独立的评估目录: {new_workspace}")
    os.makedirs(new_workspace, exist_ok=True)
    
    # 复制配置文件
    shutil.copy2(orig_cfg, os.path.join(new_workspace, "cfg_args"))
    orig_cams = os.path.join(orig_workspace, "cameras.json")
    if os.path.exists(orig_cams):
        shutil.copy2(orig_cams, os.path.join(new_workspace, "cameras.json"))
    
    return new_workspace

def link_ply(tuned_ply_path, new_workspace, iteration=30000):
    """
    将微调后的 PLY 文件链接或复制到新工作区
    :param tuned_ply_path: 微调后 PLY 文件路径
    :param new_workspace: 新工作区目录
    :param iteration: 迭代次数 (默认: 30000)
    """
    tuned_ply_path = os.path.abspath(tuned_ply_path)
    new_workspace = os.path.abspath(new_workspace)
    
    # 构建点云目录
    target_dir = os.path.join(new_workspace, "point_cloud", f"iteration_{iteration}")
    os.makedirs(target_dir, exist_ok=True)
    target_ply = os.path.join(target_dir, "point_cloud.ply")
    
    # 如果目标文件已存在，先删除
    if os.path.exists(target_ply):
        os.remove(target_ply)
        
    # 优先尝试软链接，失败则降级为复制
    try:
        os.symlink(tuned_ply_path, target_ply)
        print(f"✅ 成功创建软链接: {target_ply} -> {tuned_ply_path}")
    except OSError:
        print("⚠️ 创建软链接失败 (可能是 Windows 权限问题)，正在直接复制文件...")
        shutil.copy2(tuned_ply_path, target_ply)
        print(f"✅ 文件已复制至: {target_ply}")

def run_rendering(model_path, source_path=None):
    """
    执行渲染，采用 test.py 的方式
    :param model_path: 模型目录路径
    :param source_path: 源数据集路径 (可选)
    """
    print("开始渲染 (Rendering)...")
    cmd = ["python", "render.py", "-m", model_path]
    if source_path:
        cmd.extend(["-s", source_path])
    
    subprocess.run(cmd, cwd=gs_dir, check=True)
    print("渲染完成！\n")

def run_evaluation(model_path):
    """
    执行评估，采用 test.py 的方式
    :param model_path: 模型目录路径
    """
    print("开始评估 (Evaluation)...")
    cmd = ["python", "metrics.py", "-m", model_path]
    
    subprocess.run(cmd, cwd=gs_dir, check=True)
    print("评估完成！\n")

def setup_and_evaluate(tuned_ply_path, orig_workspace, new_workspace, source_path=None, iteration=30000):
    """
    完整的评估流程
    :param tuned_ply_path: 微调后 PLY 文件路径
    :param orig_workspace: 原版 3DGS 训练输出目录
    :param new_workspace: 新工作区目录
    :param source_path: 源数据集路径 (可选)
    :param iteration: 迭代次数 (默认: 30000)
    """
    # 1. 设置工作区
    new_workspace = setup_workspace(orig_workspace, new_workspace)
    
    # 2. 链接/复制 PLY 文件
    link_ply(tuned_ply_path, new_workspace, iteration)
    
    # 3. 执行渲染
    try:
        run_rendering(new_workspace, source_path)
    except subprocess.CalledProcessError:
        print("❌ 渲染过程中发生错误。")
        return
    
    # 4. 执行评估
    try:
        run_evaluation(new_workspace)
    except subprocess.CalledProcessError:
        print("❌ 评估过程中发生错误。")
        return
    
    print(f"\n🎉 评估完成！所有结果已安全隔离保存在: {new_workspace}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="克隆配置并评估微调后的 3DGS PLY 文件")
    parser.add_argument("--ply", "-p", type=str, required=True, help="微调后 ply 文件的绝对或相对路径")
    parser.add_argument("--orig_model", "-m", type=str, required=True, help="原版 3DGS 训练输出的目录路径 (提供配置)")
    parser.add_argument("--new_model", "-n", type=str, required=True, help="你想设置的【新文件夹名称或路径】")
    parser.add_argument("--source_path", "-s", type=str, default=None, help="源数据集路径 (可选)")
    parser.add_argument("--iteration", "-i", type=int, default=30000, help="伪造的迭代次数 (默认: 30000)")
    
    args = parser.parse_args()
    setup_and_evaluate(args.ply, args.orig_model, args.new_model, args.source_path, args.iteration)
