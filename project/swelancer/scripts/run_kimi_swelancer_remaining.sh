#!/usr/bin/env bash
set -euo pipefail

dry_run=false
args=()
for arg in "$@"; do
    if [[ "$arg" == "--dry-run" ]]; then
        dry_run=true
    else
        args+=("$arg")
    fi
done

if [[ ${#args[@]} -lt 1 ]]; then
    echo "Usage: $0 BASE_RUN_GROUP [COMPLETED_RUN_GROUP ...]" >&2
    exit 2
fi

base_run_group=${args[0]}
completed_run_groups=("${args[@]}")

strip_task_uuid() {
    sed -E 's/_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$//'
}

all_ids=$(
    find "$base_run_group" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; \
        | strip_task_uuid \
        | sort -u
)

completed_ids=$(
    for group in "${completed_run_groups[@]}"; do
        for f in "$group"/*/run.log; do
            [[ -e "$f" ]] || continue
            rg -q 'SWELancerGrade\(score=' "$f" || continue
            task=${f%/run.log}
            basename "$task" | strip_task_uuid
        done
    done | sort -u
)

if [[ -n "${RERUN_TASKS:-}" ]]; then
    rerun_ids=$(printf '%s\n' "$RERUN_TASKS" | tr ', ' '\n' | sed '/^$/d' | sort -u)
    completed_ids=$(comm -23 <(printf '%s\n' "$completed_ids" | sort -u) <(printf '%s\n' "$rerun_ids"))
fi

remaining_ids=$(
    comm -23 <(printf '%s\n' "$all_ids") <(printf '%s\n' "$completed_ids") \
        | sed '/^$/d'
)

remaining_count=$(printf '%s\n' "$remaining_ids" | sed '/^$/d' | wc -l | tr -d ' ')
taskset=$(
    printf '%s\n' "$remaining_ids" \
        | sed '/^$/d' \
        | sed "s/.*/'&'/" \
        | paste -sd, - \
        | sed 's/^/[/' \
        | sed 's/$/]/'
)

echo "Restarting Kimi SWE-Lancer with ${remaining_count} remaining tasks"
echo "Base run group: ${base_run_group}"
echo "Completed run groups: ${completed_run_groups[*]}"
if [[ -n "${RERUN_TASKS:-}" ]]; then
echo "Forcing rerun of tasks: ${RERUN_TASKS}"
fi
echo "Solver model: ${SWELANCER_MODEL:-openai/kimi-k2.7-code}"
echo "Solver: ${SWELANCER_SOLVER:-swelancer.solvers.swelancer_agent.solver:ToolAwareAgentSolver}"
echo "Model response timeout: ${MODEL_RESPONSE_TIMEOUT:-900}"
echo "Max turns: ${MAX_TURNS:-100}"
echo "Rollout timeout: ${ROLLOUT_TIMEOUT:-unset}"
if [[ -n "${REASONING_EFFORT:-}" ]]; then
    echo "Reasoning effort: ${REASONING_EFFORT}"
fi
echo "Pull from registry: ${SWELANCER_PULL_FROM_REGISTRY:-True}"

if [[ "$dry_run" == true ]]; then
    printf '%s\n' "$remaining_ids" | sed '/^$/d' | head -n 20
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
    swelancer.solver="${SWELANCER_SOLVER:-swelancer.solvers.swelancer_agent.solver:ToolAwareAgentSolver}" \
    swelancer.solver.model="${SWELANCER_MODEL:-openai/kimi-k2.7-code}" \
    swelancer.solver.model_response_timeout="${MODEL_RESPONSE_TIMEOUT:-900}" \
    swelancer.solver.max_turns="${MAX_TURNS:-100}" \
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
