#!/usr/bin/env bash
set -euo pipefail

dry_run=false
if [[ "${1:-}" == "--dry-run" ]]; then
    dry_run=true
    shift
fi

if [[ $# -eq 0 && -z "${TASKS:-}" ]]; then
    echo "Usage: TASKS='49298_510 44509_397' $0 [--dry-run]" >&2
    echo "   or: $0 [--dry-run] 49298_510 44509_397" >&2
    echo "Optional env: SWELANCER_MODEL=openai/kimi-k2.7-code-mlx-3 SWELANCER_SOLVER=... MODEL_RESPONSE_TIMEOUT=600 MAX_TURNS=40 ROLLOUT_TIMEOUT=10800 REASONING_EFFORT=high" >&2
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

echo "Running SWE-Lancer taskset: ${taskset}"
echo "Solver model: ${SWELANCER_MODEL:-openai/kimi-k2.7-code}"
echo "Solver: ${SWELANCER_SOLVER:-swelancer.solvers.swelancer_agent.solver:SimpleAgentSolver}"
echo "Model response timeout: ${MODEL_RESPONSE_TIMEOUT:-600}"
echo "Max turns: ${MAX_TURNS:-40}"
echo "Rollout timeout: ${ROLLOUT_TIMEOUT:-unset}"
if [[ -n "${REASONING_EFFORT:-}" ]]; then
    echo "Reasoning effort: ${REASONING_EFFORT}"
fi
echo "Pull from registry: ${SWELANCER_PULL_FROM_REGISTRY:-True}"

if [[ "$dry_run" == true ]]; then
    exit 0
fi

solver_args=()
if [[ -n "${REASONING_EFFORT:-}" ]]; then
    solver_args+=(swelancer.solver.reasoning_effort="${REASONING_EFFORT}")
fi
if [[ -n "${ROLLOUT_TIMEOUT:-}" ]]; then
    solver_args+=(swelancer.solver.max_rollout_seconds="${ROLLOUT_TIMEOUT}")
fi

ALCATRAZ_SKIP_JUPYTER_SETUP=true \
ALCATRAZ_JUPYTER_READY_TIMEOUT="${ALCATRAZ_JUPYTER_READY_TIMEOUT:-180}" \
OPENAI_API_KEY="${OPENAI_API_KEY:-lm-studio}" \
OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://localhost:1234/v1}" \
uv run python swelancer/run_swelancer.py \
    swelancer.split=diamond \
    swelancer.task_type=ic_swe \
    swelancer.taskset="$taskset" \
    swelancer.solver="${SWELANCER_SOLVER:-swelancer.solvers.swelancer_agent.solver:SimpleAgentSolver}" \
    swelancer.solver.model="${SWELANCER_MODEL:-openai/kimi-k2.7-code}" \
    swelancer.solver.model_response_timeout="${MODEL_RESPONSE_TIMEOUT:-600}" \
    swelancer.solver.max_turns="${MAX_TURNS:-40}" \
    ${solver_args[@]+"${solver_args[@]}"} \
    swelancer.solver.computer_runtime=nanoeval_alcatraz.alcatraz_computer_interface:AlcatrazComputerRuntime \
    swelancer.solver.computer_runtime.env=alcatraz.clusters.local:LocalConfig \
    swelancer.solver.computer_runtime.env.pull_from_registry="${SWELANCER_PULL_FROM_REGISTRY:-True}" \
    swelancer.docker_image_prefix=swelancer/swelancer_x86 \
    swelancer.docker_image_tag=releasev1 \
    swelancer.disable_internet=False \
    swelancer.n_test_runs=1 \
    runner.concurrency=1 \
    runner.enable_slackbot=False \
    runner.should_backup=False \
    runner.recorder=nanoeval.recorder:dummy_recorder \
    runner.max_retries=0
