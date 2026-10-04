#!/bin/bash
# Full config: train camera Gaussians, then initialize world Gaussians from them.
# Usage:
#   bash scripts/train.sh exp3 0
#   bash scripts/train.sh syn 0,1,2,3
#   bash scripts/train.sh all 0 sce4

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${DIR}/common.sh"

SPLIT="${1:-}"
GPUS="${2:-}"
SCENE="${3:-}"

if [[ -z "${SPLIT}" || -z "${GPUS}" ]]; then
    echo "Usage: bash scripts/train.sh <exp2|exp3|syn|all> <gpu_list> [scene]" >&2
    exit 1
fi

activate_env

launch_one() {
    local gpu="$1" split="$2" scene="$3"
    local cfg="${ROOT}/configs/${split}/${scene}.yaml"
    local log="${LOG_DIR}/train_${split}_${scene}_gpu${gpu}.log"
    echo "[train] ${split}/${scene} gpu=${gpu} -> ${log}"
    python train.py \
        --cfg "${cfg}" \
        --data_path "${DATA_PATH}" \
        --gpu "${gpu}" \
        > "${log}" 2>&1
}

run_pool "${SPLIT}" "${GPUS}" "${SCENE}"
echo "Training finished. Results are written to newhermite2/cam_gs and newhermite2/world_gs under each scene."
