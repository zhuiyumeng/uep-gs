# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 环境

- 系统：Ubuntu-24.04 
- Conda 环境：`gs`（**所有 Python 命令必须先执行 `conda activate gs`**）
- Python 路径：`/root/miniconda3/envs/gs/bin/python`
- 数据集路径：`/root/dataset/room`

> **重要：** 执行任何 Python 脚本前，务必先 `conda activate gs` 或使用完整路径 `/home/zyw/anaconda3/envs/ugs/bin/python`。Bash 工具中每条命令是独立 shell，需在每条命令中激活环境或使用完整 Python 路径。


## 项目概述

3D Gaussian Splatting (3DGS) 流式传输研究项目，实现通过 RTP/UDP 协议对多 LOD 高斯点云数据进行实时流式传输，支持前向纠错 (FEC) 和自适应不等差错保护 (UEP)。

## 项目结构

```
uep-gs/
├── common/                     # 共享基础组件（RTP、载荷、编解码基类、CLI 工具）
│   ├── rtp.py                  # RTPPacket: RFC 3550 固定首部 (12 bytes) 序列化/反序列化
│   ├── payload.py              # PayloadHeader (8 bytes) + SceneMeta + make_dummy_gaussian()
│   ├── encoder.py              # BaseRTPEncoder: PLY 加载 + SceneMeta 构建 + frame_data 存储
│   ├── decoder.py              # BaseRTPDecoder: 收包状态机 + 6 个钩子扩展点
│   ├── cli_utils.py            # find_lod_plys(), add_shared_server_args(), add_shared_client_args()
│   ├── ply_utils.py            # build_ply_header(), write_ply()
│   └── fec_stats.py            # FECStatsReport dataclass (per-LOD 恢复统计)
├── transport/                  # 基础 RTP 流传输（无 FEC）
│   ├── server.py               # UDP 服务端：加载 PLY → 编码 → 发送 RTP 流
│   ├── client.py               # UDP 客户端：接收 RTP 流 → 解码 → 写出 PLY
│   ├── encoder.py              # RTPEncoder: 线性分片，顺序封包（继承 BaseRTPEncoder）
│   ├── decoder.py              # RTPDecoder: 线性收包 → 重组（继承 BaseRTPDecoder，无额外逻辑）
│   ├── rtp_packet.py           # 从 common.rtp 重导出
│   └── gaussian_payload.py     # 从 common.payload 重导出
├── transport_fec/              # 带 FEC 的 RTP 流传输（RS 列编码，支持 UEP）
│   ├── server.py               # 支持 --no-fec 选项的 FEC 服务端
│   ├── client.py               # 支持 --no-fec 选项的 FEC 客户端（带 FEC 解码恢复）
│   ├── encoder.py              # RTPEncoder: 按 LOD 分组 → RS 列编码 → 数据+校验交叠发送
│   ├── decoder.py              # RTPDecoder: 收包 → 重组 → FEC 恢复（覆写 6 个 BaseRTPDecoder 钩子）
│   ├── fec.py                  # column_rs_encode / column_rs_decode（基于 reedsolo）
│   ├── gaussian_payload.py     # FEC 常量 (UNIT_FEC_PARITY=4) + UEP_POLICY + FECHeader 编解码
│   └── rtp_packet.py           # 从 common.rtp 重导出
├── transport_fec_ml/           # 自适应 FEC 策略（DP 优化 + 网络状态估计）
│   ├── adaptive_fec.py         # AnalyticalOptimizer (DP 求解 RS(n,k)) + AdaptiveFECPolicy
│   ├── network_estimator.py    # NetworkEstimator: 从 RTCP RR 中用 EMA 平滑估计丢包率/抖动
│   └── rtcp.py                 # RTCP SR/RR 包序列化 (RFC 3550 子集) + FECStats 应用扩展
├── merge_ply.py                # 多个 PLY 子包无损合并工具
├── package_network_kmeans.py   # LightGaussian 重要性得分 + K-means LOD 分包
├── render_mertics.py           # 调用 gaussian-splatting 渲染和评估
└── auto_eval.py                # 隔离环境中评估微调后的 PLY 文件
```

## 关键 Python 依赖

- `numpy` — 数组操作
- `plyfile` — PLY 文件读写
- `reedsolo` — Reed-Solomon 纠错码（`pip install reedsolo`）

## 运行方式

```bash
# 基础传输（无 FEC）
python -m transport.server --lod_dir ./data --prefix model --sh_degree 3 --port 5005
python -m transport.client --host 127.0.0.1 --port 5005 --output received.ply

# 带 FEC 传输
python -m transport_fec.server --lod_dir ./data --prefix model --sh_degree 3 --port 5005
python -m transport_fec.client --host 127.0.0.1 --port 5005 --output received.ply --loss 0.05

# FEC 客户端禁用 FEC 解码
python -m transport_fec.client --host 127.0.0.1 --port 5005 --output received.ply --no-fec

# PLY 合并
python merge_ply.py --in_dir ./lod_output/ --out_file merged.ply

# K-means LOD 分包
python package_network_kmeans.py --ply input.ply --imp_score scores.npz --out_prefix ./output/model -k 4
```

## 架构要点

### 三层包之间的继承关系

```
common/encoder.py:BaseRTPEncoder  ←── transport/encoder.py:RTPEncoder (纯数据分片)
                                 ←── transport_fec/encoder.py:RTPEncoder (FEC 列编码)

common/decoder.py:BaseRTPDecoder  ←── transport/decoder.py:RTPDecoder (无额外逻辑)
                                 ←── transport_fec/decoder.py:RTPDecoder (覆写 6 个钩子)
```

- `BaseRTPEncoder` 负责 PLY 加载、Gaussian stride 计算、SceneMeta 构建、frame_data 拼接。子类只需覆写 `encode_frame()` 实现各自的分片/封包策略。
- `BaseRTPDecoder` 是一个完整的收包状态机（乱序排序 → 缓冲重组 → 帧完成回调），通过 6 个钩子方法 (`_init_extra_state`, `_reset_extra_state`, `_handle_extra_unit_types`, `_on_scene_meta_received`, `_on_gaussian_data_written`, `_before_frame_complete`) 供 FEC 子类注入块记录/校验恢复逻辑，无需复制整个状态机。
- 丢包导致未收到的 fragment 会被 `make_dummy_gaussian()` 填充（不可见高斯，位置/颜色/不透明度全零），保证帧结构完整。

### LOD（细节层次）结构
- 场景由多个 LOD 层级的高斯点云组成，低 LOD 索引 = 更高重要性（LOD 0 为底层粗结构）
- 每帧按 SceneMeta → GaussianData 分片 → EndOfFrame 顺序发送，分片按 LOD 0→N 排列
- `SceneMeta` 记录每 LOD 的高斯数量、SH degree、gaussian_stride、total_bytes 等元信息
- Gaussian stride 和 SH degree 的关系：`stride = 17 + 3 * ((sh_degree + 1)^2 - 1)`，即位置(3) + 法线(3) + f_dc(3) + f_rest(SH 系数) + opacity(1) + scale(3) + rot(4)

### FEC 策略（UEP — 不等差错保护）
- LOD 0（最重要）使用 RS(2,1)，100% 冗余（完全重复）
- LOD 1 使用 RS(10,8)，25% 带宽开销
- LOD 2/3 使用 RS(9,8)，12.5% 带宽开销
- 策略定义在 `transport_fec/gaussian_payload.py` 的 `UEP_POLICY` 字典中，通过 `get_fec_config(lod, fec_policy)` 查询
- RS 编码采用**列编码**方式：将 k 个数据包的每列字节组成 codeword，产生 n-k 个校验包，每列独立编解码
- `transport_fec_ml/adaptive_fec.py` 的 `AnalyticalOptimizer` 使用 DP（0/1 背包变种）根据丢包率和带宽预算动态计算最优 (k,n)，优化目标为加权期望恢复率（权重随 LOD 重要性递减）

### 自适应 FEC 流程（transport_fec_ml）
1. 客户端通过 RTCP Receiver Report (PT=201) 上报丢包率/抖动 + FECStats 应用扩展
2. 服务端 `NetworkEstimator` 用可配置 EMA (默认 α=0.3) 平滑估计网络状态
3. `AnalyticalOptimizer.optimize(lod_sizes, loss_rate, bandwidth_budget, lod_weights)` 以 DP 求解最优 FEC 分配
4. `AdaptiveFECPolicy(allocation)` 作为策略对象注入 Encoder/Decoder 的 `fec_policy` 参数，覆盖硬编码 UEP_POLICY

### RTP 包格式
- RTP 固定头 (12 bytes, RFC 3550) + PayloadHeader (8 bytes) + 载荷数据
- PayloadHeader: `[S(1)|E(1)|LOD(4)|reserved(2)] [reserved(8)] [unit_type(16)] [fragment_offset(32)]`
- 载荷类型：`UNIT_SCENE_META=1`, `UNIT_GAUSSIAN_DATA=2`, `UNIT_END_OF_FRAME=3`, `UNIT_FEC_PARITY=4`
- FEC 校验包在 PayloadHeader 之后额外携带 FECHeader (3 bytes: k, n, parity_index)
