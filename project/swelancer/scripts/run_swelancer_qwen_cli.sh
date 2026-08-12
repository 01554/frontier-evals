#!/usr/bin/env bash
# Run SWE-Lancer with Qwen Code CLI driving the rollout inside the task container.
#
#   scripts/run_swelancer_qwen_cli.sh 28096_836
#   TASKS='28096_836 18827_741' scripts/run_swelancer_qwen_cli.sh
#
# Expects an OpenAI-compatible server on the host, reachable from the container
# (default http://host.docker.internal:8090/v1). The solver reports no token
# usage; read correctness from `correct`/`earned`.
set -euo pipefail

if [[ $# -eq 0 && -z "${TASKS:-}" ]]; then
    echo "Usage: $0 TASK_ID [TASK_ID ...]   |   TASKS='a b' $0" >&2
    exit 2
fi

if [[ $# -gt 0 ]]; then
    task_ids=$(printf '%s\n' "$@" | sed '/^$/d')
else
    task_ids=$(printf '%s\n' "$TASKS" | tr ', ' '\n' | sed '/^$/d')
fi

taskset=$(
    printf '%s\n' "$task_ids" \
        | sed "s/.*/'&'/" \
        | paste -sd, - \
        | sed 's/^/[/' \
        | sed 's/$/]/'
)

model_id=${QWEN_MODEL_ID:-qwen3.8}
base_url=${QWEN_CONTAINER_BASE_URL:-http://host.docker.internal:8090/v1}

echo "Running SWE-Lancer taskset: ${taskset}"
echo "Solver:      QwenCliSolver (agent loop runs in-container)"
echo "Model id:    ${model_id}"
echo "Endpoint:    ${base_url}"
echo "Rollout cap: ${ROLLOUT_TIMEOUT:-10800}s"

ALCATRAZ_SKIP_JUPYTER_SETUP=true \
ALCATRAZ_JUPYTER_READY_TIMEOUT="${ALCATRAZ_JUPYTER_READY_TIMEOUT:-180}" \
uv run python swelancer/run_swelancer.py \
    swelancer.split=diamond \
    swelancer.task_type=ic_swe \
    swelancer.taskset="$taskset" \
    swelancer.solver=swelancer.solvers.swelancer_agent.qwen_cli_solver:QwenCliSolver \
    swelancer.solver.model_id="${model_id}" \
    swelancer.solver.base_url="${base_url}" \
    swelancer.solver.rollout_timeout="${ROLLOUT_TIMEOUT:-10800}" \
    swelancer.solver.computer_runtime=nanoeval_alcatraz.alcatraz_computer_interface:AlcatrazComputerRuntime \
    swelancer.solver.computer_runtime.env=alcatraz.clusters.local:LocalConfig \
    swelancer.solver.computer_runtime.env.pull_from_registry="${SWELANCER_PULL_FROM_REGISTRY:-False}" \
    swelancer.docker_image_prefix=swelancer/swelancer_x86 \
    swelancer.docker_image_tag=releasev1 \
    swelancer.disable_internet=False \
    swelancer.n_test_runs=1 \
    runner.concurrency=1 \
    runner.enable_slackbot=False \
    runner.should_backup=False \
    runner.recorder=nanoeval.recorder:dummy_recorder \
    runner.max_retries=0
