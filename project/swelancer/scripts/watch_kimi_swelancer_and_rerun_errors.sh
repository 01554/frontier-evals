#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 MAIN_RUNNER_PID BASE_RUN_GROUP [RUN_GROUP ...]" >&2
    exit 2
fi

main_runner_pid=$1
shift
run_groups=("$@")

poll_seconds=${KIMI_WATCH_POLL_SECONDS:-300}
max_rounds=${KIMI_RERUN_MAX_ROUNDS:-3}

echo "Watching Kimi SWE-Lancer runner pid ${main_runner_pid}"
echo "Poll seconds: ${poll_seconds}"
echo "Max rerun rounds: ${max_rounds}"
echo "Run groups: ${run_groups[*]}"

before_main=$(mktemp)
if [[ -n "${KIMI_WATCH_BASELINE_FILE:-}" && -e "${KIMI_WATCH_BASELINE_FILE}" ]]; then
    cp "${KIMI_WATCH_BASELINE_FILE}" "$before_main"
else
    find runs -maxdepth 1 -type d -name '*run-group_simple-solver' -print | sort > "$before_main"
fi

while kill -0 "$main_runner_pid" 2>/dev/null; do
    echo
    date
    scripts/summarize_swelancer_runs.sh "${run_groups[@]}" | sed 's/^/summary: /'
    sleep "$poll_seconds"
done

echo
date
echo "Main runner pid ${main_runner_pid} has exited. Checking unresolved rerun candidates."

after_main=$(mktemp)
find runs -maxdepth 1 -type d -name '*run-group_simple-solver' -print | sort > "$after_main"
while IFS= read -r new_group; do
    [[ -n "$new_group" ]] || continue
    echo "Adding main continuation group: ${new_group}"
    run_groups+=("$new_group")
done < <(comm -13 "$before_main" "$after_main")
rm -f "$before_main" "$after_main"

round=1
while ((round <= max_rounds)); do
    summary=$(scripts/summarize_swelancer_runs.sh "${run_groups[@]}")
    printf '%s\n' "$summary" | sed 's/^/summary: /'

    candidates=$(
        printf '%s\n' "$summary" \
            | awk -F= '$1 == "rerun_candidates_system_or_parser" {print $2}' \
            | xargs
    )

    if [[ -z "$candidates" ]]; then
        echo "No unresolved system/parser rerun candidates remain."
        exit 0
    fi

    echo
    echo "Rerun round ${round}: ${candidates}"

    before=$(mktemp)
    after=$(mktemp)
    find runs -maxdepth 1 -type d -name '*run-group_simple-solver' -print | sort > "$before"

    rerun_log="runs/kimi-k2.7-code-swelancer-autorerun-round${round}-$(date +%Y%m%dT%H%M%SJST).stdout.log"
    set +e
    RERUN_TASKS="$candidates" \
        MODEL_RESPONSE_TIMEOUT="${MODEL_RESPONSE_TIMEOUT:-600}" \
        MAX_TURNS="${MAX_TURNS:-40}" \
        scripts/run_kimi_swelancer_remaining.sh "${run_groups[@]}" 2>&1 | tee "$rerun_log"
    rc=${PIPESTATUS[0]}
    set -e

    find runs -maxdepth 1 -type d -name '*run-group_simple-solver' -print | sort > "$after"
    while IFS= read -r new_group; do
        [[ -n "$new_group" ]] || continue
        echo "Adding rerun group: ${new_group}"
        run_groups+=("$new_group")
    done < <(comm -13 "$before" "$after")
    rm -f "$before" "$after"

    if [[ "$rc" -ne 0 ]]; then
        echo "Rerun round ${round} failed with exit code ${rc}."
        exit "$rc"
    fi

    round=$((round + 1))
done

echo "Reached max rerun rounds (${max_rounds}) with candidates still unresolved."
exit 1
