#!/bin/bash
# Usage:
#   bash scripts/precompute.sh exp3 0
#   bash scripts/precompute.sh syn 0,1,2,3
#   bash scripts/precompute.sh all 0,1 fishing

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${DIR}/common.sh"

SPLIT="${1:-}"
GPUS="${2:-}"
SCENE="${3:-}"

if [[ -z "${SPLIT}" || -z "${GPUS}" ]]; then
    echo "Usage: bash scripts/precompute.sh <exp2|exp3|syn|all> <gpu_list> [scene]" >&2
    exit 1
fi

activate_env

launch_one() {
    local gpu="$1" split="$2" scene="$3"
    local cfg="${ROOT}/configs/${split}/${scene}.yaml"
    local log="${LOG_DIR}/precompute_${split}_${scene}_gpu${gpu}.log"
    echo "[precompute] ${split}/${scene} gpu=${gpu} -> ${log}"
    python mosca_precompute.py \
        --cfg "${cfg}" \
        --data_path "${DATA_PATH}" \
        --gpu "${gpu}" \
        > "${log}" 2>&1
}

run_pool "${SPLIT}" "${GPUS}" "${SCENE}"
echo "Precompute finished"
