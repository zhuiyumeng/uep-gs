import subprocess
import os
import argparse

# 1. 获取各个目录的绝对路径，确保路径正确
current_dir = os.path.dirname(os.path.abspath(__file__))
gs_dir = os.path.abspath(os.path.join(current_dir, '..', 'gaussian-splatting'))

def run_rendering(model_path, source_path):
    print("开始渲染 (Rendering)...")
    # 相当于在终端输入: python render.py -m <model_path> -s <source_path>
    cmd = ["python", "render.py", "-m", model_path, "-s", source_path]
    
    # cwd=gs_dir 极其重要！它确保脚本在 gaussian-splatting 目录下执行
    subprocess.run(cmd, cwd=gs_dir, check=True)
    print("渲染完成！\n")

def run_evaluation(model_path):
    print("开始评估 (Evaluation)...")
    # 相当于在终端输入: python metrics.py -m <model_path>
    cmd = ["python", "metrics.py", "-m", model_path]
    
    subprocess.run(cmd, cwd=gs_dir, check=True)
    print("评估完成！\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='运行 Gaussian Splatting 渲染和评估')
    parser.add_argument('-m', '--model_path', required=True, help='训练好的模型文件夹路径')
    parser.add_argument('-s', '--source_path', required=True, help='源数据集路径')
    
    args = parser.parse_args()
    
    run_rendering(args.model_path, args.source_path)
    run_evaluation(args.model_path)