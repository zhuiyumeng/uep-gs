#!/bin/bash
# ============================================================
# Transport FEC 多丢包率对比实验
#
# 验证 UEP FEC 策略在 3%/6%/10% 丢包率下的纠错效果。
# 对比 FEC ON vs FEC OFF，输出文件大小与无损合并基线比较。
#
# 用法: bash experiments/run_multi_loss.sh
# ============================================================
set -euo pipefail

export PYTHONUNBUFFERED=1

PYTHON=/root/miniconda3/envs/gs/bin/python
PROJECT_DIR=/root/uep-gs
EXP_DIR=$PROJECT_DIR/experiments/multi_loss
LOD_DIR=$EXP_DIR/lod
RECV_DIR=$EXP_DIR/received
LOG_DIR=$EXP_DIR/logs

# 实验参数
K=4
SH_DEGREE=3
LOSS_RATES=(0.03 0.06 0.10)
SEED=42
TIMEOUT=300  # 每 trial 最多 5 分钟 (FEC 编码 ~90s + 传输)
PREFIX=model
HOST=127.0.0.1

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${CYAN}============================================================${NC}"
echo -e "${CYAN}  Transport FEC 多丢包率对比实验${NC}"
echo -e "${CYAN}  丢包率: ${LOSS_RATES[*]}${NC}"
echo -e "${CYAN}  种子: $SEED | K=$K | SH=$SH_DEGREE${NC}"
echo -e "${CYAN}============================================================${NC}"

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
    local deadline=$(python3 -c "import time; print(time.time() + 10)")
    while true; do
        local now=$(python3 -c "import time; print(time.time())")
        if python3 -c "exit(0 if float($now) >= float($deadline) else 1)"; then
            return 1  # timeout
        fi
        $PYTHON -c "
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    s.bind(('127.0.0.1', $port))
    s.close()
    exit(1)  # port is free, not ready yet
except OSError:
    exit(0)  # port is taken, server is up
" 2>/dev/null && return 0
        sleep 0.3
    done
}

run_trial() {
    local label=$1      # fec 或 nofec
    local loss=$2
    local port=$3
    local out_ply=$4
    local fec_flag=$5   # "--no-fec" 或 ""

    local svr_log=$LOG_DIR/server_${label}_loss${loss}.log
    local cli_log=$LOG_DIR/client_${label}_loss${loss}.log

    echo -e "  ${YELLOW}[${label^^}]${NC} loss=${loss}, port=${port}"

    # 启动服务端 (使用 -u 避免缓冲)
    $PYTHON -u -m transport_fec.server \
        --lod_dir "$LOD_DIR" --prefix "$PREFIX" \
        --sh_degree "$SH_DEGREE" --host "$HOST" --port "$port" \
        --oneshot $fec_flag \
        > "$svr_log" 2>&1 &

    local svr_pid=$!

    # 等待服务端端口就绪
    if ! wait_port_ready "$port"; then
        echo -e "    ${RED}FAIL${NC}  服务端未在 10s 内就绪 (port=$port)"
        kill "$svr_pid" 2>/dev/null || true
        return
    fi
    sleep 0.3

    # 运行客户端 (使用 -u 避免缓冲)
    $PYTHON -u -m transport_fec.client \
        --host "$HOST" --port "$port" \
        --output "$out_ply" \
        --loss "$loss" --seed "$SEED" \
        --timeout "$TIMEOUT" \
        $fec_flag \
        > "$cli_log" 2>&1 || true

    # 停止服务端
    kill "$svr_pid" 2>/dev/null || true
    wait "$svr_pid" 2>/dev/null || true

    # 输出结果
    if [ -f "$out_ply" ]; then
        local sz=$(stat -c%s "$out_ply")
        local sz_mb=$(python3 -c "print(f'{${sz}/1024/1024:.1f}')")
        echo -e "    ${GREEN}OK${NC}  $(basename $out_ply)  (${sz_mb} MB, ${sz} bytes)"
    else
        echo -e "    ${RED}FAIL${NC}  $(basename $out_ply)  (文件未生成)"
    fi
}

# ============================================================
# Step 1: K-means LOD 分包
# ============================================================
echo -e "\n${CYAN}[Step 1/3]${NC} K-means LOD 分包 (K=$K)"
STEP1_LOG=$LOG_DIR/step1_kmeans.log

PLY_SRC=/root/gaussian-splatting/output/room/point_cloud/iteration_30000/point_cloud.ply
IMP_SCORE=/root/gaussian-splatting/output/room/imp_score.npz

$PYTHON -u $PROJECT_DIR/package_network_kmeans.py \
    --ply "$PLY_SRC" \
    --imp_score "$IMP_SCORE" \
    --out_prefix "$LOD_DIR/$PREFIX" \
    -k "$K" \
    > "$STEP1_LOG" 2>&1

echo "  生成的 LOD 文件:"
for f in "$LOD_DIR"/${PREFIX}_lod*.ply; do
    sz=$(stat -c%s "$f")
    sz_mb=$(python3 -c "print(f'{${sz}/1024/1024:.1f}')")
    echo -e "    ${GREEN}$(basename $f)${NC}  (${sz_mb} MB)"
done

# ============================================================
# Step 2: 传输试验
# ============================================================
echo -e "\n${CYAN}[Step 2/3]${NC} FEC 传输试验 (3×2=6 次)"

PORT_BASE=5100

for loss in "${LOSS_RATES[@]}"; do
    echo -e "\n  ${YELLOW}--- loss=${loss} ---${NC}"

    # FEC ON
    port_fec=$(find_free_port $PORT_BASE)
    run_trial "fec" "$loss" "$port_fec" \
        "$RECV_DIR/fec_loss${loss}.ply" ""

    # FEC OFF
    port_nofec=$(find_free_port $((port_fec + 2)))
    run_trial "nofec" "$loss" "$port_nofec" \
        "$RECV_DIR/nofec_loss${loss}.ply" "--no-fec"
done

# ============================================================
# Step 3: 无损合并基线
# ============================================================
echo -e "\n${CYAN}[Step 3/3]${NC} 无损合并基线"
STEP3_LOG=$LOG_DIR/step3_merge.log

BASELINE_PLY=$RECV_DIR/merged_baseline.ply

$PYTHON -u $PROJECT_DIR/merge_ply.py \
    --in_dir "$LOD_DIR" \
    --out_file "$BASELINE_PLY" \
    > "$STEP3_LOG" 2>&1

baseline_sz=$(stat -c%s "$BASELINE_PLY")
baseline_mb=$(python3 -c "print(f'{${baseline_sz}/1024/1024:.1f}')")
echo -e "  ${GREEN}Baseline:${NC} ${baseline_mb} MB (${baseline_sz} bytes)"

# ============================================================
# 结果汇总
# ============================================================
echo -e "\n${CYAN}============================================================${NC}"
echo -e "${CYAN}  结果汇总${NC}"
echo -e "${CYAN}============================================================${NC}"
echo ""
printf "  %-22s  %10s  %10s  %10s\n" "文件" "大小(MB)" "vs Baseline" "备注"
echo "  -----------------------------------------------------------------"

for f in "$BASELINE_PLY" "$RECV_DIR"/*.ply; do
    [ ! -f "$f" ] && continue
    if [ "$(basename "$f")" = "merged_baseline.ply" ]; then
        continue
    fi
    sz=$(stat -c%s "$f")
    sz_mb=$(python3 -c "print(f'{${sz}/1024/1024:.1f}')")
    delta=$(python3 -c "print(f'{(${sz} - ${baseline_sz}) / ${baseline_sz} * 100:+.1f}')")

    name=$(basename "$f" .ply)
    # 解析 label
    if [[ "$name" == fec_* ]]; then
        note="FEC ON"
    elif [[ "$name" == nofec_* ]]; then
        note="FEC OFF"
    else
        note=""
    fi
    printf "  %-22s  %10s  %10s%%  %s\n" "$name" "$sz_mb" "$delta" "$note"
done

echo ""
printf "  %-22s  %10s  %10s  %s\n" "merged_baseline" "$baseline_mb" "0.0%" "(无损参考)"
echo ""
echo -e "${GREEN}实验完成。${NC} 日志: $LOG_DIR/"
echo -e "产物: $RECV_DIR/"
