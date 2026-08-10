#!/usr/bin/env bash
# Run SWE-Lancer with Kimi Code CLI driving the rollout inside the task container.
#
#   scripts/run_swelancer_kimi_cli.sh 28096_836
#   TASKS='28096_836 18827_741' scripts/run_swelancer_kimi_cli.sh
#
# Separate from run_swelancer_taskset_local_model.sh because that script always
# passes swelancer.solver.model_response_timeout and .max_turns, and KimiCliSolver
# has neither -- the CLI owns the agent loop, so per-turn budgets are not the
# solver's to set. chz rejects unknown fields rather than ignoring them, so the
# two argument sets cannot share a runner.
#
# Expects an OpenAI-compatible K3 server on the host, reachable from the
# container at host.docker.internal:8080. Start it with:
#   kimi-k3-mlx/scripts/serve_k3_supervised.sh --model <path> --trust-remote-code --min-p 0.05
#
# Note the solver reports no token usage -- results.csv will show 0 input/output
# tokens. Read correctness from `correct`/`earned`.

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

model_id=${K3_MODEL_ID:-/Users/hello/mac_workspace/models/kimi-k3-reap73-mlx-mxfp4-q8}
base_url=${K3_CONTAINER_BASE_URL:-http://host.docker.internal:8080/v1}

echo "Running SWE-Lancer taskset: ${taskset}"
echo "Solver:      KimiCliSolver (agent loop runs in-container)"
echo "Model id:    ${model_id}"
echo "Endpoint:    ${base_url}  (as seen from the container)"
echo "Rollout cap: ${ROLLOUT_TIMEOUT:-10800}s"

ALCATRAZ_SKIP_JUPYTER_SETUP=true \
ALCATRAZ_JUPYTER_READY_TIMEOUT="${ALCATRAZ_JUPYTER_READY_TIMEOUT:-180}" \
uv run python swelancer/run_swelancer.py \
    swelancer.split=diamond \
    swelancer.task_type=ic_swe \
    swelancer.taskset="$taskset" \
    swelancer.solver=swelancer.solvers.swelancer_agent.kimi_cli_solver:KimiCliSolver \
    swelancer.solver.model_id="${model_id}" \
    swelancer.solver.base_url="${base_url}" \
    swelancer.solver.rollout_timeout="${ROLLOUT_TIMEOUT:-10800}" \
    swelancer.solver.max_context_size="${K3_MAX_CTX:-131072}" \
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
