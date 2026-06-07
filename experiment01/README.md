# experiment01: transport_fec_ml 模块优化验证

## 实验目的

验证 `transport_fec_ml` 模块 3 个维度修复 + 2 个新增组件的正确性：

| # | 实验 | 验证内容 |
|---|------|---------|
| 1 | `test_optimizer.py` | Benefit 公式归一化 + CANDIDATE_K 扩展 + DP 正确性 |
| 2 | `test_network.py` | 冷启动修复 + Per-LOD EMA + RTT 估计 |
| 3 | `compare_fec.py` | FECSimulator 离线策略对比 (UEP vs Old DP vs New DP) |
| 4 | `test_pipeline.py` | AdaptiveFECPipeline 闭环编排 |

## 修改范围

仅修改 `transport_fec_ml/` 模块：

### 修改的文件
- `adaptive_fec.py` — Benefit 归一化 (lod_gaussian_counts + sqrt(parity+1))，CANDIDATE_K 扩展 [1-12]，DP 优化，结构化日志
- `network_estimator.py` — 冷启动 (保守先验 5%)，Per-LOD EMA 平滑，RTT 估计
- `rtcp.py` — 删除死代码 `_rtcp_common_header`

### 新增的文件
- `pipeline.py` — AdaptiveFECPipeline 编排器
- `simulator.py` — FECSimulator 离线模拟器

## 运行方式

```bash
cd /root/uep-gs
conda activate gs

mkdir -p experiment01/{logs,results}

# 逐实验运行
python experiment01/test_optimizer.py 2>&1 | tee experiment01/logs/01_optimizer_test.log
python experiment01/test_network.py   2>&1 | tee experiment01/logs/02_network_test.log
python experiment01/compare_fec.py    2>&1 | tee experiment01/logs/03_compare_fec.log
python experiment01/test_pipeline.py  2>&1 | tee experiment01/logs/04_pipeline_test.log
```

## 关键实验结果

### 实验 1 — DP 优化器

| 场景 | 旧公式 (bug) | 新公式 (fixed) |
|------|-------------|---------------|
| 合成 6%, 20% 预算 | LOD0-3 全部 RS(12,10), overhead=20% | LOD0-2: RS(7,5)/RS(12,9)/RS(12,9), LOD3: (8,8) **overhead=0.66%** ✅ |
| 真实 6%, 20% 预算 | LOD0: RS(5,1) 100%冗余, LOD3: RS(12,10) | LOD0: RS(2,1), LOD2/3: RS(12,10) **更均衡** ✅ |
| 真实 10%, 20% 预算 | RS(5,1) 在 LOD0-2, overkill | RS(2,1)+RS(12,8), **更精确分配** ✅ |

**关键发现**：新公式在低丢包率下只用 0.66% 开销（vs 旧公式 20%）就达到几乎相同的 WRR。`sqrt(parity+1)` 边际递减项有效抑制了"all to LOD3"的倾向。

### 实验 2 — NetworkEstimator

- ✅ 冷启动：首个 p=0 报告 → EMA=0.02（不锁定在 0）
- ✅ EMA 跟踪：p 序列 0→0.05→0.10→0.03 → EMA 平滑跟随
- ✅ Per-LOD EMA：正确平滑 per-LOD 丢包率和恢复率
- ✅ RTT：与预期值误差 < 0.001s

### 实验 3 — 策略模拟对比

| 丢包率 | UEP_POLICY | Old DP | New DP | 胜出 |
|--------|-----------|--------|--------|------|
| 3% | 99.43% gauss rec | 99.92% (+0.48%) | 99.92% (+0.48%) | **DP (olds≈new)** |
| 6% | 97.90% | 99.41% (+1.5%) | 99.40% (+1.5%) | **DP (olds≈new)** |
| 10% | **94.80%** | 90.13% (-4.7%) | 90.13% (-4.7%) | **UEP 胜出** ⚠️ |

**10% 丢包率下 DP 策略退化的原因**：优化器模型（all-or-nothing block recovery）在 10% 丢包时，大多数 RS(k,n) 的 delta ≤ 0（FEC "看起来"不划算）。但实际上 reedsolo 的 erasure decoding 在高丢包下仍能部分恢复。当前保守模型导致 DP 选择 minimal FEC (0.69%)，UEP 的固定 12.5% 反而更有效。

**启示**：恢复率模型需要从 all-or-nothing 升级为更精确的列级恢复概率估计。

### 实验 4 — Pipeline 闭环

- ✅ 策略切换历史正确记录（5 次优化，9 次 RTCP 反馈）
- ✅ 丢包率上升 → 策略自适应调整（WRR 从 0.9906 → 0.9281）
- ✅ `should_optimize()` 阈值机制正确（min_interval + loss_change_threshold）
- ✅ `force=True` 绕过阈值限制

## 已知局限

1. **恢复模型偏保守**：all-or-nothing 假设在 >10% 丢包时低估实际恢复能力，导致 DP 比固定 UEP 更差
2. **真实 LOD0 仅 1 个高斯**：K-means 分包产生的极端分布导致 LOD0 benefit 极低，新公式在低丢包率下不给 LOD0 分配 FEC（合理行为——保护 1 个高斯不值得）
3. **无真实 RTCP 集成**：Pipeline 已实现，但未接入 transport_fec server/client（按计划约束）

## 文件树

```
experiment01/
├── README.md                   # 本文件
├── test_optimizer.py           # 实验 1: DP 优化器正确性测试
├── test_network.py             # 实验 2: NetworkEstimator 测试
├── compare_fec.py              # 实验 3: 离线策略对比 + FECSimulator
├── test_pipeline.py            # 实验 4: Pipeline 闭环测试
├── logs/
│   ├── 01_optimizer_test.log
│   ├── 02_network_test.log
│   ├── 03_compare_fec.log
│   └── 04_pipeline_test.log
└── results/
    └── allocation_comparison.json
```
