#!/bin/bash
# ============================================================
# transport_fec_ml 真实传输实验
#
# 使用 ML DP 优化的 FEC 策略在 3%/6%/10% 丢包率下传输，
# 然后对产物进行 PSNR/SSIM/LPIPS 渲染评估。
#
# 用法: bash experiment01/run_ml_transmission.sh
# ============================================================
set -euo pipefail

export PYTHONUNBUFFERED=1

PYTHON=/root/miniconda3/envs/gs/bin/python
PROJECT_DIR=/root/uep-gs
EXP_DIR=$PROJECT_DIR/experiment01
ML_RECV_DIR=$EXP_DIR/ml_received
ML_LOG_DIR=$EXP_DIR/ml_logs
ML_EVAL_DIR=$EXP_DIR/ml_eval

# 复用已有 LOD 数据
LOD_DIR=/root/uep-gs/experiments/multi_loss/lod
PREFIX=model
SH_DEGREE=3
HOST=127.0.0.1
SEED=42
TIMEOUT=300

# ML 参数
ML_BUDGET=0.20
LOSS_RATES=(0.03 0.06 0.10)
# LOD 高斯数（L0:1, L1:356, L2:5967, L3:450621）
LOD_INFO="1,356,5967,450621"

# 渲染参数
ORIG_MODEL=/root/gaussian-splatting/output/room
SOURCE_PATH=/root/dataset/room
ITERATION=30000

mkdir -p "$ML_RECV_DIR" "$ML_LOG_DIR" "$ML_EVAL_DIR"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# ---- 辅助函数 ----

find_free_port() {
    local port=$1
    $PYTHON -c "
import socket
for p in range($port, $port + 100):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.bind(('127.0.0.1', p))
        s.close()
        print(p)
        break
    except OSError:
        s.close()
        continue
"
}

wait_port_ready() {
    local port=$1
    local deadline
    deadline=$($PYTHON -c "import time; print(time.time() + 15)")
    while true; do
        local now
        now=$($PYTHON -c "import time; print(time.time())")
        if $PYTHON -c "exit(0 if float($now) >= float($deadline) else 1)"; then
            return 1
        fi
        $PYTHON -c "
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    s.bind(('127.0.0.1', $port))
    s.close()
    exit(1)
except OSError:
    exit(0)
" 2>/dev/null && return 0
        sleep 0.5
    done
}

# ============================================================
# Step 1: FEC 传输试验 (3 次，ML 策略)
# ============================================================
echo -e "${CYAN}============================================================${NC}"
echo -e "${CYAN}  Transport FEC-ML 传输实验${NC}"
echo -e "${CYAN}  丢包率: ${LOSS_RATES[*]}${NC}"
echo -e "${CYAN}  ML Budget: ${ML_BUDGET}${NC}"
echo -e "${CYAN}============================================================${NC}"

PORT_BASE=5200

for loss in "${LOSS_RATES[@]}"; do
    echo ""
    echo -e "${YELLOW}--- ML-FEC loss=${loss} ---${NC}"

    port=$(find_free_port $PORT_BASE)
    out_ply="$ML_RECV_DIR/ml_fec_loss${loss}.ply"
    svr_log="$ML_LOG_DIR/ml_server_loss${loss}.log"
    cli_log="$ML_LOG_DIR/ml_client_loss${loss}.log"

    # 启动 ML 服务端
    $PYTHON -u -m transport_fec_ml.server \
        --lod_dir "$LOD_DIR" --prefix "$PREFIX" \
        --sh_degree "$SH_DEGREE" --host "$HOST" --port "$port" \
        --oneshot \
        --ml-policy "$loss" --ml-budget "$ML_BUDGET" \
        > "$svr_log" 2>&1 &

    svr_pid=$!

    if ! wait_port_ready "$port"; then
        echo -e "  ${RED}FAIL${NC} 服务端未就绪"
        kill "$svr_pid" 2>/dev/null || true
        continue
    fi
    sleep 0.3

    # 运行 ML 客户端
    $PYTHON -u -m transport_fec_ml.client \
        --host "$HOST" --port "$port" \
        --output "$out_ply" \
        --loss "$loss" --seed "$SEED" \
        --timeout "$TIMEOUT" \
        --ml-policy "$loss" --ml-budget "$ML_BUDGET" \
        --lod-info "$LOD_INFO" \
        > "$cli_log" 2>&1 || true

    kill "$svr_pid" 2>/dev/null || true
    wait "$svr_pid" 2>/dev/null || true

    if [ -f "$out_ply" ]; then
        sz=$(stat -c%s "$out_ply")
        sz_mb=$($PYTHON -c "print(f'${sz}/1024/1024:.1f')")
        echo -e "  ${GREEN}OK${NC}  $(basename $out_ply)  (${sz_mb} MB, ${sz} bytes)"
    else
        echo -e "  ${RED}FAIL${NC}  $(basename $out_ply)  (文件未生成)"
    fi
done

# ============================================================
# Step 2: 渲染评估 (auto_eval.py)
# ============================================================
echo ""
echo -e "${CYAN}============================================================${NC}"
echo -e "${CYAN}  PSNR/SSIM/LPIPS 渲染评估${NC}"
echo -e "${CYAN}============================================================${NC}"

export PATH=/root/miniconda3/envs/gs/bin:$PATH

for loss in "${LOSS_RATES[@]}"; do
    label="ml_fec_${loss}"
    ply_path="$ML_RECV_DIR/ml_fec_loss${loss}.ply"
    workspace="$ML_EVAL_DIR/$label"

    if [ ! -f "$ply_path" ]; then
        echo "[$label] SKIP: file not found"
        continue
    fi

    echo ""
    echo "--- [$label] ---"

    if [ -f "$workspace/results.json" ]; then
        echo "  [CACHED] results.json exists, skipping"
    else
        $PYTHON -u auto_eval.py \
            --ply "$ply_path" \
            --orig_model "$ORIG_MODEL" \
            --new_model "$workspace" \
            --source_path "$SOURCE_PATH" \
            --iteration "$ITERATION" 2>&1 || echo "  [WARN] auto_eval returned non-zero"
        echo "  Done."
    fi
done

# ============================================================
# Step 3: 结果汇总
# ============================================================
echo ""
echo -e "${CYAN}============================================================${NC}"
echo -e "${CYAN}  ML-FEC 评估结果${NC}"
echo -e "${CYAN}============================================================${NC}"
echo ""

$PYTHON -u << PYEOF
import json, os

evals = [
    ('ml_fec_0.03', 'ML-FEC 3% loss'),
    ('ml_fec_0.06', 'ML-FEC 6% loss'),
    ('ml_fec_0.10', 'ML-FEC 10% loss'),
]

print(f"{'Model':>22s}  {'PSNR':>8s}  {'SSIM':>8s}  {'LPIPS':>8s}")
print(f"{'':>22s}  {'dB':>8s}  {'':>8s}  {'':>8s}")
print('-' * 54)

for label, name in evals:
    rpath = f'/root/uep-gs/experiment01/ml_eval/{label}/results.json'
    if os.path.exists(rpath):
        with open(rpath) as f:
            data = json.load(f)
        first_key = next(iter(data), None)
        if first_key:
            m = data[first_key]
            print(f'{name:>22s}  {m["PSNR"]:8.2f}  {m["SSIM"]:8.4f}  {m["LPIPS"]:8.4f}')
    else:
        print(f'{name:>22s}  {"N/A":>8s}  {"N/A":>8s}  {"N/A":>8s}')

# 与之前 UEP_POLICY 结果对比
print()
print("对比之前 UEP_POLICY 结果:")
print(f"{'Loss':>8s}  {'ML-FEC PSNR':>12s}  {'UEP PSNR':>12s}  {'Δ':>8s}")
print('-' * 44)

uep_results = {
    '3%': 23.03, '6%': 22.21, '10%': 22.03,
}
for loss_pct, loss_val in [('3%', '0.03'), ('6%', '0.06'), ('10%', '0.10')]:
    rpath = f'/root/uep-gs/experiment01/ml_eval/ml_fec_{loss_val}/results.json'
    if os.path.exists(rpath):
        with open(rpath) as f:
            data = json.load(f)
        first_key = next(iter(data), None)
        if first_key:
            ml_psnr = data[first_key]['PSNR']
            uep_psnr = uep_results[loss_pct]
            delta = ml_psnr - uep_psnr
            print(f'{loss_pct:>8s}  {ml_psnr:12.2f}  {uep_psnr:12.2f}  {delta:+8.2f}')
    else:
        print(f'{loss_pct:>8s}  {"N/A":>12s}  {uep_results[loss_pct]:12.2f}  {"N/A":>8s}')

PYEOF

echo ""
echo -e "${GREEN}ML 传输实验完成。${NC}"
echo "  产物: $ML_RECV_DIR/"
echo "  日志: $ML_LOG_DIR/"
echo "  评估: $ML_EVAL_DIR/"
