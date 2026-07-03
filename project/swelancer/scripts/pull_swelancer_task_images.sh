#!/usr/bin/env bash
set -euo pipefail

platform=${DOCKER_DEFAULT_PLATFORM:-linux/amd64}
prefix=${SWELANCER_DOCKER_IMAGE_PREFIX:-swelancer/swelancer_x86}
tag=${SWELANCER_DOCKER_IMAGE_TAG:-releasev1}
parallel=${PULL_CONCURRENCY:-4}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<EOF
Usage:
  $0 TASK_ID [TASK_ID ...]
  TASKS='49298_510 44509_397' $0
  PULL_ALL_DIAMOND_IC_SWE=true $0

Environment:
  DOCKER_DEFAULT_PLATFORM=${platform}
  SWELANCER_DOCKER_IMAGE_PREFIX=${prefix}
  SWELANCER_DOCKER_IMAGE_TAG=${tag}
  PULL_CONCURRENCY=${parallel}
EOF
    exit 0
fi

if [[ $# -gt 0 ]]; then
    task_ids=$(printf '%s\n' "$@" | sed '/^$/d')
elif [[ -n "${TASKS:-}" ]]; then
    task_ids=$(printf '%s\n' "$TASKS" | tr ', ' '\n' | sed '/^$/d')
elif [[ "${PULL_ALL_DIAMOND_IC_SWE:-false}" == "true" ]]; then
    task_ids=$(
        uv run python - <<'PY'
import pandas as pd

tasks = pd.read_csv("all_swelancer_tasks.csv")
tasks = tasks[(tasks["set"] == "diamond") & (tasks["variant"] == "ic_swe")]
for task_id in tasks["question_id"].astype(str):
    print(task_id)
PY
    )
else
    echo "Pass task ids, set TASKS, or set PULL_ALL_DIAMOND_IC_SWE=true." >&2
    exit 2
fi

pull_one() {
    local task_id=$1
    local image="${prefix}_${task_id}:${tag}"
    echo "Pulling ${image} (${platform})"
    docker pull --platform "$platform" "$image"
    docker image inspect "$image" --format '{{json .RepoTags}} {{.Architecture}} {{.Os}} {{.Size}}'
}

export -f pull_one
export platform prefix tag

printf '%s\n' "$task_ids" \
    | sed '/^$/d' \
    | sort -u \
    | xargs -n 1 -P "$parallel" bash -c 'pull_one "$@"' _
