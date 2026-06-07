"""
LightGaussian Imp Score + K-means 渐进式分类打包 (单层单文件版)

核心思路：
  1. 读取 LightGaussian 的 imp_score.npz 作为重要性得分
  2. 使用 K-means 对得分自动聚类，得到 K 个重要性层级
  3. 按照聚类的中心得分对点云进行排序与分类 (LOD 0, LOD 1 ... LOD K-1)
  4. 每层按得分降序排列后，直接保存为一个完整的 PLY 文件（不再进行切片），方便后续挂载空间树或自定义细分。
"""

import numpy as np
from plyfile import PlyData, PlyElement
import argparse
import os


# ====================== 1D K-means 聚类 ======================

def kmeans_1d(data, k, max_iters=100):
    """
    一维 K-means 聚类算法（纯 numpy 实现，适用于重要性得分这类一维数据）

    原理说明：
        K-means 是一种经典的聚类算法，通过迭代将数据点划分到 K 个簇中，
        使得每个数据点与其所属簇中心的距离平方和最小。一维版本由于只需
        比较数值大小，算法实现更为简洁。

    参数说明：
        data (numpy.ndarray): 输入的一维数据数组（如重要性得分），形状为 (n,) 或 (n,1)
        k (int): 要聚类的簇数量，必须为正整数
        max_iters (int): 最大迭代次数，防止算法无限循环，默认值为 100

    返回值：
        tuple: 包含两个元素的元组
            - labels (numpy.ndarray): 每个数据点的簇标签，形状为 (n,)，值为 0 到 k-1
            - centroids (numpy.ndarray): 各簇的中心值，形状为 (k,)，按升序排列

    异常：
        ValueError: 当 k 小于 1 或数据为空时抛出

    算法流程：
        1. 初始化：在数据的取值范围内均匀选择 k 个中心点
        2. 迭代：
           a. 计算每个数据点到各中心的距离（这里用绝对值，因为是一维）
           b. 将每个数据点分配给距离最近的中心
           c. 重新计算每个簇的中心（平均值）
           d. 检查中心移动距离，若小于阈值则收敛
        3. 对簇中心进行升序排列，并重新映射标签
    """
    # ravel() 将多维数组展平为一维，astype(float64) 确保计算精度
    # 这步是必要的，因为输入可能是二维数组 (n,1) 或一维数组 (n,)
    data = data.ravel().astype(np.float64)
    n = len(data)

    # 边界情况处理：k=1 时，所有点都属于同一个簇
    if k == 1:
        # 返回全零标签和唯一的中心值（数据的均值）
        return np.zeros(n, dtype=np.int32), np.array([data.mean()])

    # 确定数据的取值范围，用于初始化中心点
    data_min, data_max = data.min(), data.max()

    # 初始化策略：在 [min, max] 范围内均匀分布选择 k 个中心点
    # [1:-1] 去掉首尾，保留 k 个内部点作为初始中心
    # 例如：data_min=0, data_max=10, k=3 时，得到 [2.5, 5.0, 7.5]
    centroids = np.linspace(data_min, data_max, k + 2)[1:-1]
    centroids = centroids.astype(np.float64)

    # K-means 迭代主循环
    for _ in range(max_iters):
        # 计算每个数据点到所有中心的距离
        # data[:, np.newaxis] 将 data 转换为 (n,1) 形状，与 centroids (k,) 进行广播
        # 结果 dists 形状为 (n, k)，dists[i,j] 表示第 i 个点到第 j 个中心的距离
        dists = np.abs(data[:, np.newaxis] - centroids)

        # 为每个数据点分配最近的簇标签
        # np.argmin(dists, axis=1) 返回每行最小值的索引，即该点所属的簇编号
        labels = np.argmin(dists, axis=1)

        # 重新计算每个簇的中心（均值）
        # 使用列表推导式遍历每个簇，计算该簇内所有数据点的平均值
        new_centroids = np.array([
            # (labels == j) 生成布尔数组，筛选出属于簇 j 的数据点
            # .sum() > 0 判断该簇是否有数据点，避免空簇导致均值计算错误
            data[labels == j].mean() if (labels == j).sum() > 0 else centroids[j]
            for j in range(k)
        ])

        # 检查收敛：计算新旧中心的最大移动距离
        # 若移动距离小于阈值，认为算法已收敛，可以提前退出
        shift = np.abs(new_centroids - centroids).max()
        centroids = new_centroids
        if shift < 1e-6:
            break

    # 对簇中心进行升序排列，并建立标签映射
    # 排序后，centroids[0] 是最小值（最低重要性），centroids[k-1] 是最大值（最高重要性）
    order = np.argsort(centroids)
    centroids = centroids[order]

    # 创建标签重映射表，将原始标签顺序转换为"升序排列后的顺序"
    # 例如：若原始中心顺序是 [5, 2, 8]，排序后 order=[1,0,2]
    # 则 label_map[1]=0, label_map[0]=1, label_map[2]=2
    # 最终 labels 中的值就是基于升序排列的新标签
    label_map = np.empty(k, dtype=np.int32)
    label_map[order] = np.arange(k)
    labels = label_map[labels]

    return labels, centroids


# ====================== 肘部法则自动选择 K ======================

def detect_optimal_k(data, k_max=8):
    """
    使用肘部法则（Elbow Method）自动确定最优的聚类数量 K

    原理说明：
        肘部法则通过绘制不同 K 值对应的惯性（Inertia，即各点到所属簇中心的距离平方和）
        来寻找最优 K。随着 K 增大，惯性会减小，但减小的速度会逐渐变慢。
        在拐点处（即"肘部"）继续增加 K 不再能显著降低惯性，因此该点是最优选择。

    参数说明：
        data (numpy.ndarray): 输入的一维数据数组
        k_max (int): 最多尝试的 K 值数量，默认值为 8

    返回值：
        int: 自动检测到的最优聚类数量 K，最小为 2，最大不超过 k_max

    特殊情况处理：
        - 若数据点少于 4 个，返回默认值 3
        - 若不同 K 下的惯性差异极小（< 1e-10），返回默认值 3
    """
    # 数据点太少时，直接返回默认 K 值 3
    if len(data) < 4:
        return 3

    # 确保 k_max 不超过数据的唯一值数量，同时至少为 2
    k_max = min(k_max, len(np.unique(data)))
    k_max = max(k_max, 2)

    # 存储每个 K 值对应的惯性（距离平方和）
    inertias = np.empty(k_max)

    # 遍历 1 到 k_max，计算每个 K 对应的惯性
    for k in range(1, k_max + 1):
        # 调用 kmeans_1d 进行聚类
        labels, centroids = kmeans_1d(data, k)

        # 计算惯性：所有数据点到其所属簇中心的距离平方和
        # 使用生成器表达式逐簇累加
        inertias[k - 1] = sum(
            ((data[labels == j] - centroids[j]) ** 2).sum()
            for j in range(k)
        )

    # 对惯性进行归一化处理，将值缩放到 [0, 1] 范围
    i_min, i_max = inertias.min(), inertias.max()

    # 若所有惯性值几乎相同（差异极小），返回默认 K 值 3
    if i_max - i_min < 1e-10:
        return 3

    # 归一化公式：(x - min) / (max - min)
    inertias_n = (inertias - i_min) / (i_max - i_min)

    # 准备绘制参考直线（从第一个点到最后一个点）
    ks = np.arange(1, k_max + 1)  # K 值的取值范围 [1, 2, ..., k_max]
    p1 = np.array([1, inertias_n[0]])  # 直线起点 (K=1 时的惯性)
    p2 = np.array([k_max, inertias_n[-1]])  # 直线终点 (K=k_max 时的惯性)

    # 计算参考直线的方向向量
    line_vec = p2 - p1

    # 计算直线长度，用于后续距离计算
    line_len = np.linalg.norm(line_vec)

    # 避免除零错误
    if line_len < 1e-10:
        return 3

    # 计算每个点 (k, inertias_n[k-1]) 到参考直线的垂直距离
    # 原理：两条向量叉积的模 / 底边长度 = 点到直线的距离
    # np.cross(line_vec, point - p1) 得到平行四边形的面积
    dists = np.array([
        abs(np.cross(line_vec, np.array([k, inertias_n[k - 1]]) - p1)) / line_len
        for k in ks
    ])

    # 找到距离最大的点，该点就是"肘部"，对应的 K 值即为最优
    best_k = int(np.argmax(dists) + 1)  # +1 因为 K 从 1 开始计数

    return best_k


# ====================== 主流程 ======================

def package_for_network(input_ply, imp_score_path, output_prefix, k=None):
    """
    将点云按照 LightGaussian 重要性得分进行 K-means 聚类，并打包为多个 LOD 级别的 PLY 文件

    功能说明：
        1. 读取 PLY 格式的点云文件和对应的 importance score 文件
        2. 对得分进行 K-means 聚类，自动或手动确定聚类数量
        3. 按聚类中心得分降序排列，生成多个 LOD（Level of Detail）级别的 PLY 文件
        4. 每个 LOD 内部按得分降序排列，便于后续空间细分

    参数说明：
        input_ply (str): 输入的 PLY 点云文件路径（支持 LightGaussian 导出的格式）
        imp_score_path (str): 重要性得分 npz 文件路径，通常包含 'arr_0' 或第一个数组
        output_prefix (str): 输出文件的前缀路径，如 "./output/model"
            将生成 "model_lod0.ply", "model_lod1.ply" 等文件
        k (int, optional): 聚类层级数，若为 None 则使用肘部法则自动检测

    返回值：
        None（函数直接输出文件，不返回任何值）

    异常：
        ValueError: 当 PLY 文件中的点数与 npz 文件中的得分数量不匹配时抛出
        FileNotFoundError: 当输入文件不存在时由 PlyData.read() 抛出

    输出文件说明：
        - 命名格式：{output_prefix}_lod{rank}.ply
        - rank=0 表示最高分（最重要）的层级，rank=K-1 表示最低分（最不重要）的层级
        - 每个文件包含完整的点云数据，按得分降序排列
        - 使用小端序（byte_order='<'）二进制格式保存，确保跨平台兼容性
    """
    # 读取 PLY 文件并解析顶点数据
    print(f"Reading PLY: {input_ply}")
    ply = PlyData.read(input_ply)

    # ply['vertex'].data 是一个 numpy 结构化数组，包含所有顶点属性
    v = ply['vertex'].data
    N = len(v)  # 总点数
    print(f"Total Gaussians: {N}")

    # === Step 1: 读取得分 ===
    print(f"\n=== Step 1: Load Importance Score ===")

    # 加载 npz 文件（numpy 的压缩归档格式）
    score_data = np.load(imp_score_path)

    # npz 文件可能包含多个数组，尝试读取 'arr_0' 或第一个可用的数组
    # 这种写法兼容两种常见格式：有明确名称的数组或默认的 'arr_0'
    if 'arr_0' in score_data:
        scores = score_data['arr_0']
    else:
        scores = score_data[score_data.files[0]]

    # 校验数据一致性：点数必须与得分数量匹配
    if len(scores) != N:
        raise ValueError(f"Error: 数量不匹配！PLY有 {N} 个点，npz有 {len(scores)} 个得分。")

    # 打印得分分布信息，便于分析
    print(f"Score range: [{scores.min():.6f}, {scores.max():.6f}], mean={scores.mean():.6f}")

    # === Step 2: 确定 K ===
    if k is None:
        # 使用肘部法则自动检测最优 K 值
        print("\n=== Step 2: Auto-detect optimal K ===")
        k = detect_optimal_k(scores)
        print(f"Auto-detected K = {k}")
    else:
        print(f"\n=== Step 2: Using K = {k} (user-specified) ===")

    # === Step 3: K-means 聚类 ===
    print("\n=== Step 3: K-means clustering ===")
    labels, centroids = kmeans_1d(scores, k)

    # 按聚类中心降序排列：索引 0 是最高分（最重要），索引 K-1 是最低分（最不重要）
    # argsort 默认升序，[::-1] 反转得到降序
    sorted_cluster_indices = np.argsort(centroids)[::-1]

    # 打印每个 LOD 层级的统计信息
    for rank, cluster_idx in enumerate(sorted_cluster_indices):
        # 统计该簇包含的点数
        count = (labels == cluster_idx).sum()
        print(f"  [LOD {rank}] Centroid: {centroids[cluster_idx]:.6f} | Points: {count} ({count/N:.2%})")

    # === Step 4: 按 LOD 分类打包 ===
    print(f"\n=== Step 4: LOD Packaging ===")

    # 创建输出目录（若不存在）
    out_dir = os.path.dirname(output_prefix)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # 存储所有生成的打包文件路径
    package_list = []

    # 遍历每个 LOD 层级，生成对应的 PLY 文件
    for rank, cluster_idx in enumerate(sorted_cluster_indices):
        # 布尔索引：找出属于该簇的所有点的索引
        idx_in_cluster = np.where(labels == cluster_idx)[0]

        # 获取该簇内所有点的得分
        cluster_score = scores[idx_in_cluster]

        # 类内按得分严格降序排列（最重要的排在前面）
        sorted_local = np.argsort(cluster_score)[::-1]

        # 最终排列后的全局索引
        final_idx = idx_in_cluster[sorted_local]

        # 生成输出文件名：{prefix}_lod{rank}.ply
        filename = f"{output_prefix}_lod{rank}.ply"
        cluster_size = len(final_idx)

        print(f"  -> Packaging: {os.path.basename(filename)} | Size: {cluster_size} ({cluster_size/N:.2%})")

        # 截取对应顶点数据
        chunk_vertices = v[final_idx]

        # PlyElement.describe() 将 numpy 结构化数组转换为 PLY 格式的顶点描述
        el_chunk = PlyElement.describe(chunk_vertices, 'vertex')

        # PlyData.write() 保存为 PLY 文件
        # text=False: 使用二进制格式（非 ASCII 文本），减小文件体积
        # byte_order='<': 强制小端序（Little Endian），提高跨平台兼容性
        PlyData([el_chunk], text=False, byte_order='<').write(filename)

        # 记录已生成的文件
        package_list.append(filename)

    # 打印完成信息
    print(f"\n[Success] Generated {len(package_list)} LOD packages ready for further subdivision.")


# ====================== 命令行入口 ======================

if __name__ == "__main__":
    """
    程序入口点：使用 argparse 解析命令行参数

    argparse 是 Python 标准库，用于处理命令行输入
    --xxx 表示可选参数，xxx 表示位置参数（必需）
    """
    parser = argparse.ArgumentParser(
        description="LightGaussian Imp Score 渐进式分类打包工具 (单层单文件版)")

    # --ply: 输入的 PLY 文件路径（必需）
    parser.add_argument("--ply", required=True,
                        help="输入 PLY 文件路径")

    # --imp_score: 重要性得分文件路径（必需）
    parser.add_argument("--imp_score", required=True,
                        help="输入 imp_score.npz 文件路径")

    # --out_prefix: 输出文件前缀（必需）
    # 例如：--out_prefix ./output/model 将生成 model_lod0.ply, model_lod1.ply 等
    parser.add_argument("--out_prefix", required=True,
                        help="输出文件前缀（如 ./web_assets/model，将生成 model_lod0.ply, model_lod1.ply）")

    # -k: 聚类层级数（可选，默认自动检测）
    parser.add_argument("-k", type=int, default=None,
                        help="聚类层级数（默认：肘部法则自动检测）")

    # 解析命令行参数并转换为命名空间对象
    args = parser.parse_args()

    # 调用主函数，传入解析后的参数
    package_for_network(
        input_ply=args.ply,
        imp_score_path=args.imp_score,
        output_prefix=args.out_prefix,
        k=args.k,
    )
