# !/bin/bash

if [ $# -ne 6 ]; then
    echo "Usage: $0 <model> $1 <attn_type> $2 <budget_ratio> $3 <estimate_ratio> $4 <dtype> $5 <device>"
    exit 1
fi

MODEL=${1}
ATTN_TYPE=${2}
BUDGET_RATIO=${3}
ESTIMATE_RATIO=${4}
DTYPE=${5}
DEVICE=${6}

RESULT_DIR="./results/pred/${MODEL}/${ATTN_TYPE}"

tasks=(2wikimqa gov_report hotpotqa lcc multi_news multifieldqa_en musique narrativeqa passage_retrieval_en qasper qmsum repobench-p triviaqa)
# tasks=(hotpotqa multi_news multifieldqa_en musique narrativeqa passage_retrieval_en qmsum)
# tasks=(qasper repobench-p lcc gov_report triviaqa)
# tasks=(repobench-p lcc gov_report triviaqa)
# tasks=(qasper)

for task in "${tasks[@]}"; do
    echo "Parameters: ${MODEL} ${task} ${ATTN_TYPE} ${DTYPE} ${BUDGET_RATIO} ${ESTIMATE_RATIO} ${DEVICE}"
    bash pred.sh ${MODEL} ${task} ${ATTN_TYPE} ${DTYPE} ${BUDGET_RATIO} ${ESTIMATE_RATIO} ${DEVICE}
done

echo "Start to evaluate..."
python -u eval.py \
    --attn_type ${ATTN_TYPE} \
    --model ${MODEL} \

echo "Results:"
cat "${RESULT_DIR}/result.json"
