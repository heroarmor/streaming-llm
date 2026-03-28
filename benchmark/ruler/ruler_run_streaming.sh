#!/bin/bash
# RULER benchmark runner for StreamingLLM.
#
# Usage:
#   bash ruler_run_streaming.sh <model_name> <context_length> <task> <attn_type> <start_size> <recent_size> <device>
#
# Examples:
#   # Single task, StreamingLLM
#   bash ruler_run_streaming.sh llama-3.1-8b 4096 niah_single_1 streaming 4 256 cuda:0
#
#   # Single task, Full attention (baseline)
#   bash ruler_run_streaming.sh llama-3.1-8b 4096 niah_single_1 full 4 256 cuda:0
#
#   # Run all tasks: see ruler_run_all_streaming.sh

if [ $# -ne 7 ]; then
    echo "Usage: $0 <model_name> <context_length> <task> <attn_type> <start_size> <recent_size> <device>"
    echo ""
    echo "  model_name:     llama-3.1-8b | qwen2.5-7b | llama-3-8b-1048k"
    echo "  context_length: e.g. 4096, 8192, 16384, 32768"
    echo "  task:           niah_single_1, vt, cwe, qa_1, etc."
    echo "  attn_type:      streaming | full"
    echo "  start_size:     attention sink tokens (e.g. 4)"
    echo "  recent_size:    recent tokens to keep (e.g. 256, 512, 1024)"
    echo "  device:         cuda:0, cuda:1, etc."
    exit 1
fi

MODEL_NAME_KEY=${1}
MAX_SEQ_LENGTH=${2}
TASK=${3}
ATTN_TYPE=${4}
START_SIZE=${5}
RECENT_SIZE=${6}
DEVICE=${7}

NUM_SAMPLES=200
BENCHMARK="synthetic"
DTYPE="fp16"

# Resolve model path
source ruler_config_models.sh
MODEL_CONFIG=$(MODEL_SELECT ${MODEL_NAME_KEY})
IFS=":" read MODEL_NAME MODEL_TEMPLATE_TYPE MODEL_FRAMEWORK TOKENIZER_PATH TOKENIZER_TYPE <<< "$MODEL_CONFIG"
if [ -z "${MODEL_NAME}" ]; then
    echo "Error: Model '${MODEL_NAME_KEY}' is not supported."
    echo "Supported: llama-3.1-8b, qwen2.5-7b, llama-3-8b-1048k, qwen2.5-72b"
    exit 1
fi

# Directories
ROOT_DIR="./ruler_eval_result"
if [ "${ATTN_TYPE}" == "streaming" ]; then
    RESULTS_DIR="${ROOT_DIR}/${MODEL_NAME}/${BENCHMARK}/${MAX_SEQ_LENGTH}/streaming_s${START_SIZE}_r${RECENT_SIZE}"
else
    RESULTS_DIR="${ROOT_DIR}/${MODEL_NAME}/${BENCHMARK}/${MAX_SEQ_LENGTH}/full"
fi
DATA_DIR="${RESULTS_DIR}/data"
PRED_DIR="${RESULTS_DIR}/pred"
mkdir -p ${DATA_DIR}
mkdir -p ${PRED_DIR}

echo "============================================"
echo "Model:    ${MODEL_NAME}"
echo "Task:     ${TASK}"
echo "Context:  ${MAX_SEQ_LENGTH}"
echo "Attn:     ${ATTN_TYPE} (start=${START_SIZE}, recent=${RECENT_SIZE})"
echo "Device:   ${DEVICE}"
echo "Results:  ${RESULTS_DIR}"
echo "============================================"

# Step 1: Prepare data
echo ""
echo "[Step 1/3] Preparing data..."
python -u data/prepare.py \
    --save_dir ${DATA_DIR} \
    --benchmark ${BENCHMARK} \
    --task ${TASK} \
    --tokenizer_path ${TOKENIZER_PATH} \
    --tokenizer_type ${TOKENIZER_TYPE} \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --model_template_type ${MODEL_TEMPLATE_TYPE} \
    --num_samples ${NUM_SAMPLES}

# Step 2: Run inference
echo ""
echo "[Step 2/3] Running inference..."
python -u pred/call_api_streaming.py \
    --model_name ${MODEL_NAME} \
    --attn_type ${ATTN_TYPE} \
    --max_len ${MAX_SEQ_LENGTH} \
    --data_dir ${DATA_DIR} \
    --save_dir ${PRED_DIR} \
    --benchmark ${BENCHMARK} \
    --task ${TASK} \
    --dtype ${DTYPE} \
    --device ${DEVICE} \
    --start_size ${START_SIZE} \
    --recent_size ${RECENT_SIZE}

# Step 3: Evaluate
echo ""
echo "[Step 3/3] Evaluating..."
python -u eval/evaluate.py \
    --data_dir ${PRED_DIR} \
    --benchmark ${BENCHMARK}

echo ""
echo "Done! Results in ${PRED_DIR}/summary*.csv"
