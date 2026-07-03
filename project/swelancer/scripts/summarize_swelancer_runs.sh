#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 BASE_RUN_GROUP [RUN_GROUP ...]" >&2
    exit 2
fi

base_run_group=$1
shift

if [[ $# -eq 0 ]]; then
    run_groups=("$base_run_group")
else
    run_groups=("$@")
fi

strip_task_uuid() {
    sed -E 's/_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$//'
}

all_ids=$(
    find "$base_run_group" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; \
        | strip_task_uuid \
        | sort -u
)

statuses=$(
    for group in "${run_groups[@]}"; do
        for f in "$group"/*/run.log; do
            [[ -e "$f" ]] || continue
            id=$(basename "${f%/run.log}" | strip_task_uuid)
            score=""
            if rg -q 'SWELancerGrade\(score=' "$f"; then
                score=$(rg -o 'SWELancerGrade\(score=[0-9.]+' "$f" | tail -n1 | sed 's/.*score=//')
            fi

            system_error=0
            error_lines=$(
                rg 'Unexpected error in task|RolloutSystemError in|Kernel didn'\''t respond|removal of container .* already in progress|Computer startup timed out|Error during grading' "$f" \
                    | rg -v 'Message history on turn' \
                    || true
            )
            if [[ -z "$score" && -n "$error_lines" ]]; then
                system_error=1
            fi

            timeout=0
            if [[ -z "$score" ]] && rg -q 'Model response timed out' "$f"; then
                timeout=1
            fi

            parser_error=0
            if [[ -z "$score" ]] && rg -q 'Warning: No Python code blocks were found|No Python code blocks were found|No Python code' "$f"; then
                parser_error=1
            fi

            summary=$(
                {
                    printf '%s\n' "$error_lines"
                    rg -o 'Model response timed out|Warning: No Python code blocks were found|No Python code[^'"'"'}]*' "$f" || true
                } \
                    | sed '/^$/d' \
                    | tail -n1 \
                    | tr '\t' ' ' \
                    | cut -c1-240 \
                    || true
            )

            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$id" "$score" "$system_error" "$timeout" "$parser_error" "$group" "$summary"
        done
    done \
        | awk -F '\t' '{latest[$1]=$0} END {for (k in latest) print latest[k]}' \
        | sort
)

scored=$(
    printf '%s\n' "$statuses" \
        | awk -F '\t' '$2 != "" {print $1, $2, $6}' \
        | sort
)

total=$(printf '%s\n' "$all_ids" | sed '/^$/d' | wc -l | tr -d ' ')
scored_count=$(printf '%s\n' "$scored" | sed '/^$/d' | wc -l | tr -d ' ')
correct_count=$(printf '%s\n' "$scored" | awk '$2+0>0 {c++} END{print c+0}')
remaining_count=$(
    comm -23 \
        <(printf '%s\n' "$all_ids") \
        <(printf '%s\n' "$scored" | awk '{print $1}' | sort) \
        | sed '/^$/d' \
        | wc -l \
        | tr -d ' '
)
accuracy=$(
    printf '%s\n' "$scored" \
        | awk 'NF {n++; if ($2+0>0) c++} END {if (n) printf "%.6f", c/n; else printf "0.000000"}'
)

printf 'total=%s\n' "$total"
printf 'scored_tasks=%s\n' "$scored_count"
printf 'correct=%s\n' "$correct_count"
printf 'accuracy_so_far=%s\n' "$accuracy"
printf 'remaining=%s\n' "$remaining_count"
printf 'system_errors=%s\n' "$(printf '%s\n' "$statuses" | awk -F '\t' '$3 == 1 {c++} END {print c+0}')"
printf 'model_timeouts=%s\n' "$(printf '%s\n' "$statuses" | awk -F '\t' '$4 == 1 {c++} END {print c+0}')"
printf 'parser_errors=%s\n' "$(printf '%s\n' "$statuses" | awk -F '\t' '$5 == 1 {c++} END {print c+0}')"
printf 'rerun_candidates_system_or_parser=%s\n' "$(
    printf '%s\n' "$statuses" \
        | awk -F '\t' '$3 == 1 || $5 == 1 {print $1}' \
        | sort -u \
        | paste -sd' ' -
)"
echo 'recent_scored:'
printf '%s\n' "$scored" | tail -n 20
echo 'system_error_tasks:'
printf '%s\n' "$statuses" | awk -F '\t' '$3 == 1 {print $1, $6, $7}'
echo 'model_timeout_tasks:'
printf '%s\n' "$statuses" | awk -F '\t' '$4 == 1 {print $1, $6, $7}'
echo 'parser_error_tasks:'
printf '%s\n' "$statuses" | awk -F '\t' '$5 == 1 {print $1, $6, $7}'
