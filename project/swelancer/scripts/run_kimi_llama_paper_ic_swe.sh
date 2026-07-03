#!/usr/bin/env bash
set -euo pipefail

dry_run=false
if [[ "${1:-}" == "--dry-run" ]]; then
    dry_run=true
    shift
fi

if [[ $# -ne 0 ]]; then
    echo "Usage: $0 [--dry-run]" >&2
    echo "Optional env: OPENAI_BASE_URL=http://localhost:4321/v1 SWELANCER_MODEL=openai/kimi-k2.7-code-q3 MODEL_RESPONSE_TIMEOUT=900 MAX_TURNS=100 ROLLOUT_TIMEOUT=10800" >&2
    exit 2
fi

export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://localhost:4321/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-llama.cpp}"
export SWELANCER_MODEL="${SWELANCER_MODEL:-openai/kimi-k2.7-code-q2}"
export SWELANCER_SOLVER="${SWELANCER_SOLVER:-swelancer.solvers.swelancer_agent.solver:ToolAwareAgentSolver}"
export MODEL_RESPONSE_TIMEOUT="${MODEL_RESPONSE_TIMEOUT:-900}"
export MAX_TURNS="${MAX_TURNS:-100}"
export SWELANCER_PULL_FROM_REGISTRY="${SWELANCER_PULL_FROM_REGISTRY:-True}"

echo "Running SWE-Lancer Diamond IC SWE full run"
echo "Solver model: ${SWELANCER_MODEL}"
echo "Solver: ${SWELANCER_SOLVER}"
echo "OpenAI base URL: ${OPENAI_BASE_URL}"
echo "Model response timeout: ${MODEL_RESPONSE_TIMEOUT}"
echo "Max turns: ${MAX_TURNS}"
echo "Rollout timeout: ${ROLLOUT_TIMEOUT:-unset}"
echo "Paper strict wall-clock: set ROLLOUT_TIMEOUT=10800"
if [[ -n "${REASONING_EFFORT:-}" ]]; then
    echo "Reasoning effort: ${REASONING_EFFORT}"
fi
echo "Pull from registry: ${SWELANCER_PULL_FROM_REGISTRY}"

solver_args=()
if [[ -n "${REASONING_EFFORT:-}" ]]; then
    solver_args+=(swelancer.solver.reasoning_effort="${REASONING_EFFORT}")
fi
if [[ -n "${ROLLOUT_TIMEOUT:-}" ]]; then
    solver_args+=(swelancer.solver.max_rollout_seconds="${ROLLOUT_TIMEOUT}")
fi

if [[ "$dry_run" == true ]]; then
    printf 'uv run python swelancer/run_swelancer.py ...\n'
    exit 0
fi

ALCATRAZ_SKIP_JUPYTER_SETUP=true \
ALCATRAZ_JUPYTER_READY_TIMEOUT="${ALCATRAZ_JUPYTER_READY_TIMEOUT:-180}" \
uv run python swelancer/run_swelancer.py \
    swelancer.split=diamond \
    swelancer.task_type=ic_swe \
    swelancer.solver="${SWELANCER_SOLVER}" \
    swelancer.solver.model="${SWELANCER_MODEL}" \
    swelancer.solver.model_response_timeout="${MODEL_RESPONSE_TIMEOUT}" \
    swelancer.solver.max_turns="${MAX_TURNS}" \
    ${solver_args[@]+"${solver_args[@]}"} \
    swelancer.solver.computer_runtime=nanoeval_alcatraz.alcatraz_computer_interface:AlcatrazComputerRuntime \
    swelancer.solver.computer_runtime.env=alcatraz.clusters.local:LocalConfig \
    swelancer.solver.computer_runtime.env.pull_from_registry="${SWELANCER_PULL_FROM_REGISTRY}" \
    swelancer.docker_image_prefix=swelancer/swelancer_x86 \
    swelancer.docker_image_tag=releasev1 \
    swelancer.disable_internet=False \
    swelancer.n_test_runs=1 \
    runner.concurrency=1 \
    runner.enable_slackbot=False \
    runner.should_backup=False \
    runner.recorder=nanoeval.recorder:dummy_recorder \
    runner.max_retries=0
