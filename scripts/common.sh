#!/bin/bash
# Full newhermite2 config: experiment name and loss weights come from configs/base_config.yaml
# plus any per-scene overrides already written in the scene yaml.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_PATH="${DATA_PATH:-/export1/jliugk/project2/data_hdr}"
ENV_NAME="${ENV_NAME:-mono4dgs}"
LOG_DIR="${ROOT}/logs_full"

EXP2_SCENES=(Ninja sce1 sce2 sce3 sce4 sce5 ThrowingTowel WavingHands)
EXP3_SCENES=(CheckingEmail Cleaning Skateboarder Dog sce1 sce2 sce3 sce4)
SYN_SCENES=(bridge bridge_2 cars fishing hallway students students_2 welding welding_2)

activate_env() {
    if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV}" != "${ENV_NAME}" ]]; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "${ENV_NAME}"
    fi
    cd "${ROOT}"
    mkdir -p "${LOG_DIR}"
}

scenes_for() {
    local split="$1"
    case "${split}" in
        exp2) printf '%s\n' "${EXP2_SCENES[@]}" ;;
        exp3) printf '%s\n' "${EXP3_SCENES[@]}" ;;
        syn)  printf '%s\n' "${SYN_SCENES[@]}" ;;
        all)
            printf '%s\n' "${EXP2_SCENES[@]}" | sed 's|^|exp2 |'
            printf '%s\n' "${EXP3_SCENES[@]}" | sed 's|^|exp3 |'
            printf '%s\n' "${SYN_SCENES[@]}" | sed 's|^|syn |'
            ;;
        *)
            echo "Unknown split: ${split} (choose exp2, exp3, syn, or all)" >&2
            return 1
            ;;
    esac
}

# One scene per GPU at a time; extra scenes wait in a queue.
# Caller must define launch_one <gpu> <split> <scene>.
run_pool() {
    local split="$1"
    local gpu_list="$2"
    shift 2
    local only_scene="${1:-}"

    IFS=',' read -r -a GPUS <<< "${gpu_list}"
    local n_gpu="${#GPUS[@]}"
    if (( n_gpu < 1 )); then
        echo "Pass a GPU list, for example 0 or 0,1,2,3" >&2
        return 1
    fi

    local -a jobs=()
    if [[ "${split}" == "all" ]]; then
        while read -r sp sc; do
            [[ -n "${only_scene}" && "${sc}" != "${only_scene}" ]] && continue
            jobs+=("${sp} ${sc}")
        done < <(scenes_for all)
    else
        while read -r sc; do
            [[ -n "${only_scene}" && "${sc}" != "${only_scene}" ]] && continue
            jobs+=("${split} ${sc}")
        done < <(scenes_for "${split}")
    fi

    if (( ${#jobs[@]} == 0 )); then
        echo "No matching scene" >&2
        return 1
    fi

    local start=0
    while (( start < ${#jobs[@]} )); do
        local pids=()
        local i
        for (( i = 0; i < n_gpu && start + i < ${#jobs[@]}; i++ )); do
            local sp sc
            read -r sp sc <<< "${jobs[$((start + i))]}"
            launch_one "${GPUS[$i]}" "${sp}" "${sc}" &
            pids+=($!)
        done
        local pid
        for pid in "${pids[@]}"; do
            wait "${pid}"
        done
        start=$((start + n_gpu))
    done
}
