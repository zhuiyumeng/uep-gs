"""
3DGS 子包无损合并工具

功能说明：
    将通过分包算法产生的多个 .ply 文件重新拼装为单个标准的 3DGS（3D Gaussian Splatting）模型。
    移除了渲染与评估自动化脚本，专注于数据的安全合并。

使用场景：
    当使用 K-means 或其他分包算法将大型点云分割为多个 LOD 层级后，
    需要将其重新合并为完整的点云模型时使用。
"""

import os
import glob
import argparse
import numpy as np
from plyfile import PlyData, PlyElement


def merge_ply_files(input_dir, output_file):
    """
    将指定目录下的所有 PLY 文件合并为一个完整的 PLY 文件

    功能说明：
        1. 扫描输入目录下的所有 .ply 文件
        2. 按文件名排序读取每个 PLY 文件的顶点数据
        3. 使用 numpy 的 np.concatenate() 将所有顶点数据拼接为一个结构化数组
        4. 将合并后的数据写入新的 PLY 文件

    参数说明：
        input_dir (str): 输入目录路径，包含要合并的多个 .ply 文件
        output_file (str): 输出文件路径，合并后的 PLY 文件将保存到此位置

    返回值：
        None（函数直接输出文件，不返回任何值）

    异常处理：
        - 若目录中没有任何 .ply 文件，打印错误信息并正常返回
        - 若读取某个 PLY 文件时发生异常，打印错误信息并正常返回
        注意：本函数不会抛出异常给调用者，而是自行处理并终止执行

    技术要点：
        - 使用 glob.glob() 模式匹配查找所有 .ply 文件
        - 使用 sorted() 确保文件按名称顺序处理（保证合并顺序一致）
        - 使用 np.concatenate() 拼接结构化数组，能完整保留所有顶点属性字段
        - 使用 plyfile 库的 PlyElement.describe() 将 numpy 数组转换为 PLY 格式
    """
    # 打印扫描信息，便于用户了解程序执行进度
    print(f"🔍 正在扫描目录: {input_dir}")

    # 使用 glob.glob() 进行文件模式匹配
    # os.path.join() 确保路径分隔符在不同操作系统上正确（Windows 用 \，Linux/Mac 用 /）
    # "*.ply" 匹配所有以 .ply 结尾的文件
    search_pattern = os.path.join(input_dir, "*.ply")
    ply_files = sorted(glob.glob(search_pattern))

    # 检查是否找到了任何 PLY 文件
    if not ply_files:
        # 如果没有找到文件，打印错误信息并直接返回
        # 不抛出异常，让程序可以继续运行（便于脚本集成）
        print(f"❌ 错误: 在 {input_dir} 中未找到任何 .ply 文件。")
        return

    # 打印发现的子包数量，准备开始合并流程
    print(f"📦 发现 {len(ply_files)} 个子包，准备开始合并...")

    # vertex_arrays: 存储每个 PLY 文件的顶点数据（numpy 结构化数组）
    # total_points: 累计统计所有文件的总点数
    vertex_arrays = []
    total_points = 0

    # 遍历每个 PLY 文件，逐一读取其顶点数据
    for file_path in ply_files:
        try:
            # 打印正在读取的文件名（只用 basename 避免路径过长）
            print(f"  -> 读取: {os.path.basename(file_path)}")

            # PlyData.read() 是 plyfile 库的核心函数，用于读取 PLY 文件
            # 返回的 ply_data 是一个 PlyData 对象，包含所有数据
            ply_data = PlyData.read(file_path)

            # ply_data['vertex'] 访问名为 'vertex' 的元素（PLY 标准格式的顶点块）
            # .data 属性返回 numpy 结构化数组（structured array），包含所有顶点属性
            # 结构化数组的特点：每一列可以是不同数据类型，列名对应属性名（如 x, y, z, nx, ny, nz, red, green, blue 等）
            vertex_data = ply_data['vertex'].data

            # 将读取的顶点数据追加到列表中
            vertex_arrays.append(vertex_data)

            # 累加总点数
            total_points += len(vertex_data)

        except Exception as e:
            # 捕获所有可能的异常（如文件损坏、格式错误、权限问题等）
            # 打印错误信息并返回，避免生成不完整的合并文件
            print(f"❌ 读取 {file_path} 时发生错误: {e}")
            return

    # 打印正在拼接的信息，显示总点数便于用户预估内存使用
    print(f"\n⚙️ 正在内存中拼接属性... (总点数: {total_points})")

    # 核心拼接步骤：使用 np.concatenate() 合并所有结构化数组
    # concatenate() 要求所有数组具有相同的数据类型（结构化数组的 dtype 必须一致）
    # 由于所有 PLY 文件都是由同一个导出流程生成，dtype 完全兼容
    # np.concatenate() 比循环拼接效率高得多，因为它只需一次内存分配
    merged_vertex_data = np.concatenate(vertex_arrays)

    # PlyElement.describe() 是 plyfile 库的核心函数之一
    # 功能：将 numpy 结构化数组转换为 PlyElement 对象，并指定元素名称为 'vertex'
    # 这是 PLY 文件格式的要求：数据必须包装在 PlyElement 中才能写入
    el_merged = PlyElement.describe(merged_vertex_data, 'vertex')

    # 确保输出目录存在，避免写入失败
    # os.path.dirname() 提取文件路径的目录部分（如 "a/b/c.ply" -> "a/b"）
    out_dir = os.path.dirname(output_file)
    if out_dir and not os.path.exists(out_dir):
        # os.makedirs() 可以递归创建多级目录（如 "a/b/c" 其中 a/b 都不存在）
        os.makedirs(out_dir)

    # 打印正在写入的信息
    print(f"💾 正在写入合并后的模型到: {output_file}")

    # PlyData.write() 是 plyfile 库用于写入 PLY 文件的核心函数
    # 参数说明：
    #   - [el_merged]: 一个包含 PlyElement 对象的列表（PLY 文件可以包含多个元素，但这里只有一个 vertex 元素）
    #   - text=False: 使用二进制格式（False 表示非文本/ASCII 格式），显著减小文件体积
    #   - byte_order='<': 强制使用小端序（Little Endian）字节序
    #     小端序是大多数 PC 和 GPU 使用的方式，确保文件可以被 C++/CUDA 渲染器正确读取
    PlyData([el_merged], text=False, byte_order='<').write(output_file)

    # 合并完成，打印成功信息
    print("✅ 合并成功！该模型现在可以直接投入官方 SIBR_viewer 或网络渲染器查看。")


# ====================== 命令行入口 ======================

if __name__ == '__main__':
    """
    程序入口点

    __name__ == '__main__' 是 Python 的标准用法，确保只有在直接运行此脚本时
    才会执行下面的代码。当此脚本被其他脚本 import 时，下面的代码不会执行。
    这是一个"魔法方法"（magic method / dunder method），由 Python 解释器自动提供。
    """
    # argparse 是 Python 标准库，用于解析命令行参数
    # 它提供了自动生成帮助信息、处理默认值、验证参数等功能
    parser = argparse.ArgumentParser(description="无损合并 3DGS PLY 子包")

    # --in_dir: 输入目录（必需），包含要合并的多个 ply 文件
    # required=True 表示此参数必须提供
    parser.add_argument('--in_dir', type=str, required=True,
                        help='包含要合并的多个 ply 文件的目录')

    # --out_file: 输出文件路径（可选），默认为 'merged_full.ply'
    parser.add_argument('--out_file', type=str, default='merged_full.ply',
                        help='合并后的输出文件路径')

    # parse_args() 解析命令行参数，返回一个命名空间对象
    # 命令行输入会被转换为：args.in_dir, args.out_file 等属性
    args = parser.parse_args()

    # os.path.abspath() 将相对路径转换为绝对路径
    # 这是个好习惯，确保路径不会因为工作目录改变而出现问题
    in_dir = os.path.abspath(args.in_dir)
    out_file = os.path.abspath(args.out_file)

    # 调用主函数执行合并操作
    merge_ply_files(in_dir, out_file)
