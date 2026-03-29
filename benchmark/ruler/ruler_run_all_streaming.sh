#!/bin/bash
# RULER benchmark for StreamingLLM on Llama-3.1-8B.
# Splits 39 jobs (3 contexts × 13 tasks) across 4 machines.
#
# Usage:
#   bash ruler_run_all_streaming.sh <worker_id> <device>
#
#   worker_id: 0, 1, 2, or 3
#   device:    cuda:0 (default)
#
# Example (on each of 4 machines):
#   Machine 0:  bash ruler_run_all_streaming.sh 0 cuda:0
#   Machine 1:  bash ruler_run_all_streaming.sh 1 cuda:0
#   Machine 2:  bash ruler_run_all_streaming.sh 2 cuda:0
#   Machine 3:  bash ruler_run_all_streaming.sh 3 cuda:0

set -e

WORKER_ID=${1:?Usage: $0 <worker_id 0-3> [device]}
DEVICE=${2:-cuda:0}
MODEL="llama-3.1-8b"
START=4
NUM_WORKERS=2

# ---- Cache model weights in /tmp to save home space ----
export HF_HOME="/tmp/hf_cache"
mkdir -p "$HF_HOME"

# Copy cached weights from home if they exist (faster than re-downloading)
SRC_CACHE="$HOME/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct"
DST_CACHE="$HF_HOME/hub/models--meta-llama--Llama-3.1-8B-Instruct"
if [ -d "$SRC_CACHE" ] && [ ! -d "$DST_CACHE" ]; then
    echo "Copying model weights from home to /tmp ..."
    mkdir -p "$HF_HOME/hub"
    cp -r "$SRC_CACHE" "$DST_CACHE"
    echo "Done. Model cached at $DST_CACHE"
elif [ -d "$DST_CACHE" ]; then
    echo "Model already cached at $DST_CACHE"
else
    echo "No local cache found. Model will be downloaded to $HF_HOME on first run."
fi

# Also copy tokenizer token if needed
if [ -f "$HOME/.cache/huggingface/token" ] && [ ! -f "$HF_HOME/token" ]; then
    cp "$HOME/.cache/huggingface/token" "$HF_HOME/token"
fi

# ---- Build all jobs: 3 contexts × 13 tasks = 39 jobs ----
CONTEXTS=(65536)

TASKS=(
    niah_single_1
    niah_single_2
    niah_single_3
    niah_multikey_1
    niah_multikey_2
    niah_multikey_3
    niah_multivalue
    niah_multiquery
    vt
    cwe
    fwe
    qa_1
    qa_2
)

ALL_JOBS=()
for CTX in "${CONTEXTS[@]}"; do
    for TASK in "${TASKS[@]}"; do
        ALL_JOBS+=("${CTX}:${TASK}")
    done
done

TOTAL_JOBS=${#ALL_JOBS[@]}

# ---- Select this worker's jobs ----
MY_JOBS=()
for i in "${!ALL_JOBS[@]}"; do
    if [ $((i % NUM_WORKERS)) -eq ${WORKER_ID} ]; then
        MY_JOBS+=("${ALL_JOBS[$i]}")
    fi
done

echo "=========================================="
echo " RULER Benchmark: StreamingLLM"
echo " Worker:  ${WORKER_ID} / ${NUM_WORKERS}"
echo " Jobs:    ${#MY_JOBS[@]} / ${TOTAL_JOBS}"
echo " Model:   ${MODEL}"
echo " Device:  ${DEVICE}"
echo " HF_HOME: ${HF_HOME}"
echo "=========================================="
echo ""
echo "My jobs:"
for JOB in "${MY_JOBS[@]}"; do
    IFS=":" read CTX TASK <<< "$JOB"
    RECENT=$((CTX / 4))
    echo "  ctx=${CTX}  task=${TASK}  start=${START}  recent=${RECENT}"
done
echo ""

# ---- Run ----
DONE=0
for JOB in "${MY_JOBS[@]}"; do
    IFS=":" read CTX TASK <<< "$JOB"
    RECENT=$((CTX / 4))
    DONE=$((DONE + 1))

    echo ""
    echo "====== [Worker ${WORKER_ID}] Job ${DONE}/${#MY_JOBS[@]}: ctx=${CTX} ${TASK} ======"
    bash ruler_run_streaming.sh ${MODEL} ${CTX} ${TASK} streaming ${START} ${RECENT} ${DEVICE}
done

echo ""
echo "=========================================="
echo " Worker ${WORKER_ID} finished all ${#MY_JOBS[@]} jobs!"
echo "=========================================="
