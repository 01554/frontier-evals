#!/usr/bin/env bash
# One-button replication of the Kimi-K3 REAP SWE-Lancer results.
#   https://github.com/01554/kimi-k3-gguf-prune/blob/main/evals/
#
#   scripts/replicate_k3_reap.sh <build> <taskset>
#
#   build:   reap640 | reap576 | full896-stream
#   taskset: probe | differential | trio | battle16 | all24
#
# Environment:
#   LLAMA_SERVER  path to llama-server built from 01554/llama.cpp branch
#                 k3-stream (required; that branch = Unsloth K3 fork + PR #25294)
#   MODEL         path to the build's .gguf (first shard for multi-part)
#   MOE_CACHE_GIB stream cache for full896-stream (default 380; fit to your RAM)
#   PORT          default 8090
#
# Protocol notes (kept identical to the published runs):
#   - rollout cap 10800 s for resident builds, 18000 s for full896-stream
#     (compensates its ~1.5x slower decode; same budget in work terms)
#   - sampling temp 1.0 / top-p 0.95, ctx 131072, --cache-reuse 0
#   - ONE attempt per task; a re-run of this script skips tasks that already
#     have a result file, so delete replication_results/ to start over
set -euo pipefail

BUILD=${1:?usage: replicate_k3_reap.sh <reap640|reap576|full896-stream> <taskset>}
SET=${2:?taskset: probe|differential|trio|battle16|all24}
PORT=${PORT:-8090}
HERE=$(cd "$(dirname "$0")/.." && pwd)

PROBE=(28096_836 18827_741 29618_781)
DIFFERENTIAL=(14294 24508_791 15815_1 27353_776 15925)
TRIO=(14294 15815_1 15925)
BATTLE16=(29916_609 6883 43395_530 25901_945 40259_1089 18746_833 4324 44429_1100 \
          40208_1108 44618_1007 19132_872 41885_1134 50064_846 50314_790 37688_441 44040_470)

case "$SET" in
  probe)        TASKS=("${PROBE[@]}");;
  differential) TASKS=("${DIFFERENTIAL[@]}");;
  trio)         TASKS=("${TRIO[@]}");;
  battle16)     TASKS=("${BATTLE16[@]}");;
  all24)        TASKS=("${PROBE[@]}" "${DIFFERENTIAL[@]}" "${BATTLE16[@]}");;
  *) echo "unknown taskset: $SET"; exit 2;;
esac

STREAM_ARGS=()
CAP=10800
case "$BUILD" in
  reap640)  DEFAULT_MODEL=models/REAP640-IQ1_S/Kimi-K3-REAP640-IQ1_S-00001-of-00010.gguf;;
  reap576)  DEFAULT_MODEL=models/REAP576-IQ2_XXS/Kimi-K3-REAP576-IQ2_XXS.gguf;;
  full896-stream)
    DEFAULT_MODEL=models/UD-IQ2_XXS/Kimi-K3-UD-IQ2_XXS-00001-of-00016.gguf
    STREAM_ARGS=(--moe-stream --moe-stream-cache "${MOE_CACHE_GIB:-380}")
    CAP=18000;;
  *) echo "unknown build: $BUILD"; exit 2;;
esac
MODEL=${MODEL:-$DEFAULT_MODEL}
LLAMA_SERVER=${LLAMA_SERVER:?set LLAMA_SERVER to llama-server from 01554/llama.cpp k3-stream}
[ -f "$MODEL" ] || { echo "model not found: $MODEL (set MODEL=...)"; exit 2; }

OUT="$HERE/replication_results/$BUILD"
mkdir -p "$OUT"

if pgrep -x llama-server >/dev/null; then
  echo "a llama-server is already running; stop it first"; exit 1
fi
"$LLAMA_SERVER" -m "$MODEL" --host 127.0.0.1 --port "$PORT" -ngl 99 -c 131072 \
  --jinja --cache-reuse 0 --temp 1.0 --top-p 0.95 "${STREAM_ARGS[@]}" \
  > "$OUT/server.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT

echo "waiting for server (loads the full model; several minutes)..."
ok=0
for _ in $(seq 1 420); do
  if curl -sf "http://127.0.0.1:$PORT/health" | grep -q '"ok"'; then ok=1; break; fi
  kill -0 $SERVER 2>/dev/null || { echo "server died; see $OUT/server.log"; exit 1; }
  sleep 5
done
[ "$ok" = 1 ] || { echo "server never became healthy"; exit 1; }
echo "server healthy; running ${#TASKS[@]} task(s), cap ${CAP}s each"

cd "$HERE"
for T in "${TASKS[@]}"; do
  if [ -s "$OUT/$T.csv" ]; then echo "skip $T (already done)"; continue; fi
  echo "=== $BUILD / $T  start $(date '+%m-%d %H:%M') ==="
  K3_MODEL_ID=k3 \
  K3_CONTAINER_BASE_URL="http://host.docker.internal:$PORT/v1" \
  K3_MAX_CTX=131072 \
  ROLLOUT_TIMEOUT=$CAP \
    scripts/run_swelancer_kimi_cli.sh "$T" > "$OUT/$T.log" 2>&1 || true
  RUN=$(ls -t "$HERE/runs" | head -1)
  if [ -s "$HERE/runs/$RUN/results.csv" ]; then
    cp "$HERE/runs/$RUN/results.csv" "$OUT/$T.csv"
    tail -1 "$OUT/$T.csv"
  else
    echo "WARN: no results.csv for $T (see $OUT/$T.log)"
  fi
done

echo
echo "=== $BUILD / $SET summary ==="
for T in "${TASKS[@]}"; do
  [ -s "$OUT/$T.csv" ] && tail -1 "$OUT/$T.csv"
done
echo
echo "Reference results to compare against:"
echo "  https://github.com/01554/kimi-k3-gguf-prune/blob/main/evals/results.csv"
