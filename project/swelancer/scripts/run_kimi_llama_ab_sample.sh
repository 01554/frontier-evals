#!/usr/bin/env bash
set -euo pipefail

dry_run=false
if [[ "${1:-}" == "--dry-run" ]]; then
    dry_run=true
    shift
fi

tasks=("$@")
if [[ ${#tasks[@]} -eq 0 ]]; then
    tasks=(49298_510 44509_397 29851_717 19022_888)
fi

args=
if [[ "$dry_run" == true ]]; then
    args=--dry-run
fi

OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://localhost:4321/v1}" \
OPENAI_API_KEY="${OPENAI_API_KEY:-llama.cpp}" \
SWELANCER_MODEL="${SWELANCER_MODEL:-openai/kimi-k2.7-code-q2}" \
SWELANCER_SOLVER="${SWELANCER_SOLVER:-swelancer.solvers.swelancer_agent.solver:ToolAwareAgentSolver}" \
MODEL_RESPONSE_TIMEOUT="${MODEL_RESPONSE_TIMEOUT:-900}" \
MAX_TURNS="${MAX_TURNS:-100}" \
ROLLOUT_TIMEOUT="${ROLLOUT_TIMEOUT:-}" \
scripts/run_swelancer_taskset_local_model.sh \
    ${args:+"$args"} \
    "${tasks[@]}"
