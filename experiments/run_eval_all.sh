#!/bin/bash
# ============================================================
# 批量渲染评估脚本
# 对 transport_fec 实验的产物运行 auto_eval.py，
# 计算 PSNR/SSIM/LPIPS 指标
# ============================================================
set -euo pipefail

export PYTHONUNBUFFERED=1
export PATH=/root/miniconda3/envs/gs/bin:$PATH

PYTHON=/root/miniconda3/envs/gs/bin/python
RECV_DIR=/root/uep-gs/experiments/multi_loss/received
EVAL_DIR=/root/uep-gs/experiments/multi_loss/eval
ORIG_MODEL=/root/gaussian-splatting/output/room
SOURCE_PATH=/root/dataset/room
ITERATION=30000

# 待评估的 PLY 文件列表 (label:filename)
declare -A FILES=(
  ["merged"]="merged_baseline.ply"
  ["fec_3"]="fec_loss0.03.ply"
  ["fec_6"]="fec_loss0.06.ply"
  ["fec_10"]="fec_loss0.10.ply"
  ["nofec_3"]="nofec_loss0.03.ply"
  ["nofec_6"]="nofec_loss0.06.ply"
  ["nofec_10"]="nofec_loss0.10.ply"
)

echo "============================================================"
echo "  批量渲染评估 (PSNR / SSIM / LPIPS)"
echo "  共 ${#FILES[@]} 个模型"
echo "============================================================"

declare -A METRICS

for label in "${!FILES[@]}"; do
  fname="${FILES[$label]}"
  ply_path="$RECV_DIR/$fname"
  workspace="$EVAL_DIR/$label"

  if [ ! -f "$ply_path" ]; then
    echo "[$label] SKIP: file not found: $ply_path"
    continue
  fi

  echo ""
  echo "--- [$label] $fname ---"
  echo "    PLY: $ply_path"
  echo "    WS:  $workspace"

  # 检查是否已有结果
  if [ -f "$workspace/results.json" ]; then
    echo "    [CACHED] results.json already exists, skipping"
  else
    $PYTHON -u auto_eval.py \
      --ply "$ply_path" \
      --orig_model "$ORIG_MODEL" \
      --new_model "$workspace" \
      --source_path "$SOURCE_PATH" \
      --iteration "$ITERATION" 2>&1 || echo "    [WARN] auto_eval returned non-zero"

    echo "    Done."
  fi

  # 读取指标
  if [ -f "$workspace/results.json" ]; then
    METRICS[$label]=$(cat "$workspace/results.json")
  fi
done

# ============================================================
# 汇总
# ============================================================
echo ""
echo "============================================================"
echo "  评估结果汇总"
echo "============================================================"
echo ""

$PYTHON -u << PYEOF
import json, os

evals = [
    ('merged',  'Merged (无损)'),
    ('fec_3',   'FEC 3%'),
    ('fec_6',   'FEC 6%'),
    ('fec_10',  'FEC 10%'),
    ('nofec_3', 'NoFEC 3%'),
    ('nofec_6', 'NoFEC 6%'),
    ('nofec_10','NoFEC 10%'),
]

results = {}
for label, _ in evals:
    rpath = f'/root/uep-gs/experiments/multi_loss/eval/{label}/results.json'
    if os.path.exists(rpath):
        with open(rpath) as f:
            data = json.load(f)
        first_key = next(iter(data), None)
        if first_key:
            results[label] = data[first_key]

if not results:
    print('No results found.')
    exit(1)

print(f'{"":>20s}  {"PSNR":>8s}  {"SSIM":>8s}  {"LPIPS":>8s}')
print(f'{"":>20s}  {"dB":>8s}  {"":>8s}  {"":>8s}')
print('-' * 56)

model_names = {
    'merged': 'Merged (lossless)',
    'fec_3': 'FEC 3% loss',
    'fec_6': 'FEC 6% loss',
    'fec_10': 'FEC 10% loss',
    'nofec_3': 'NoFEC 3% loss',
    'nofec_6': 'NoFEC 6% loss',
    'nofec_10': 'NoFEC 10% loss',
}

for label, _ in evals:
    if label not in results:
        print(f'{model_names.get(label, label):>20s}  {"N/A":>8s}  {"N/A":>8s}  {"N/A":>8s}')
        continue
    m = results[label]
    print(f'{model_names.get(label, label):>20s}  {m["PSNR"]:8.2f}  {m["SSIM"]:8.4f}  {m["LPIPS"]:8.4f}')

# FEC 增益 (vs NoFEC at same loss rate)
print()
print('FEC 增益 (FEC - NoFEC):')
for loss in ['3', '6', '10']:
    f_key = f'fec_{loss}'
    n_key = f'nofec_{loss}'
    if f_key in results and n_key in results:
        fm = results[f_key]
        nm = results[n_key]
        d_psnr = fm['PSNR'] - nm['PSNR']
        d_ssim = fm['SSIM'] - nm['SSIM']
        d_lpips = fm['LPIPS'] - nm['LPIPS']
        print(f'  loss={loss}%:  PSNR {d_psnr:+.2f} dB  SSIM {d_ssim:+.4f}  LPIPS {d_lpips:+.4f}')
PYEOF

echo ""
echo "Complete. Results in: $EVAL_DIR/"
