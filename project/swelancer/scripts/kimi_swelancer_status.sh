#!/usr/bin/env bash
set -euo pipefail

stdout_log=${KIMI_SWELANCER_STDOUT_LOG:-runs/kimi-k2.7-code-swelancer-outputfix2-20260617T015544JST.stdout.log}
current_run_group=${KIMI_SWELANCER_CURRENT_RUN_GROUP:-runs/2026-06-16T16-55-45-UTC_run-group_simple-solver}

run_groups=(
    runs/2026-06-16T10-16-23-UTC_run-group_simple-solver
    runs/2026-06-16T10-16-23-UTC_run-group_simple-solver
    runs/2026-06-16T14-42-58-UTC_run-group_simple-solver
    runs/2026-06-16T15-05-36-UTC_run-group_simple-solver
    runs/2026-06-16T15-06-51-UTC_run-group_simple-solver
    runs/2026-06-16T15-08-45-UTC_run-group_simple-solver
    runs/2026-06-16T15-18-19-UTC_run-group_simple-solver
    "$current_run_group"
)

latest_log() {
    for f in "$current_run_group"/*/run.log; do
        [[ -e "$f" ]] || continue
        stat -f '%m %N' "$f"
    done 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-
}

echo 'screen:'
screen -ls || true

echo
echo 'processes:'
ps -axo pid,ppid,etime,stat,%cpu,%mem,command \
    | rg 'SCREEN|run_kimi_swelancer_remaining|run_swelancer|kimi-k2|LM Studio|lms log' \
    | cut -c1-420 \
    || true

echo
echo 'stdout_progress:'
if [[ -e "$stdout_log" ]]; then
    stat -f 'stdout_mtime=%Sm size=%z bytes' -t '%Y-%m-%d %H:%M:%S %Z' "$stdout_log"
    tail -c 220000 "$stdout_log" \
        | tr '\r' '\n' \
        | sed -E 's/\x1b\[[0-9;]*m//g' \
        | rg 'SWELancer|Unexpected error in task|RolloutSystemError|Summary|Evaluated|Got back all results' \
        | tail -n 8 \
        || true
else
    echo "missing stdout log: $stdout_log"
fi

echo
echo 'summary:'
scripts/summarize_swelancer_runs.sh "${run_groups[@]}"

echo
echo 'latest_task:'
latest=$(latest_log)
if [[ -n "$latest" ]]; then
    now=$(date +%s)
    mt=$(stat -f '%m' "$latest")
    task_dir=$(basename "${latest%/run.log}")
    task_id=$(printf '%s\n' "$task_dir" | sed -E 's/_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$//')
    echo "log=$latest"
    echo "task_id=$task_id"
    echo "age_seconds=$((now - mt))"
    stat -f 'mtime=%Sm size=%z bytes' -t '%Y-%m-%d %H:%M:%S %Z' "$latest"
    tail -c 180000 "$latest" \
        | rg -o 'Evaluating task with ID [^}]+|SETUP|Setup complete|Message history on turn [0-9]+|No user tool call detected; executing code|User tool call detected|Warning: No Python code blocks were found|No Python code|Model response timed out|SWELancerGrade\(score=[0-9.]+|pytest_exit: [-0-9]+|Unexpected error in task|RolloutSystemError|Grading task|Tests ran smoothly' \
        | tail -n 40 \
        || true
else
    echo "no run.log found under $current_run_group"
fi

echo
echo 'docker:'
cid=$(docker ps --format '{{.ID}} {{.Names}} {{.Image}}' | awk '/container0-/ {print $1; exit}')
printf 'cid=%s\n' "$cid"
if [[ -n "$cid" ]]; then
    docker stats --no-stream --format '{{.Container}} cpu={{.CPUPerc}} mem={{.MemUsage}} block={{.BlockIO}}' "$cid" || true
    docker exec "$cid" bash -lc 'date; ps -eo pid,ppid,stat,etime,pcpu,pmem,args | sort -k5 -nr | head -n 20' || true
fi
