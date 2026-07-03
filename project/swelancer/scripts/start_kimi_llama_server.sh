#!/usr/bin/env bash
set -euo pipefail

quant=${1:-UD-Q3_K_XL}
model_root=${2:-$HOME/mac_workspace/models/kimi-k2.7-code-gguf}
port=${LLAMA_SERVER_PORT:-4321}
ctx=${LLAMA_CONTEXT:-65536}
host=${LLAMA_SERVER_HOST:-127.0.0.1}
extra_args=()
if [[ -n "${LLAMA_SERVER_EXTRA_ARGS:-}" ]]; then
    read -r -a extra_args <<< "$LLAMA_SERVER_EXTRA_ARGS"
fi

model_dir="$model_root/$quant"
first_shard=$(find "$model_dir" -maxdepth 1 -name '*-00001-of-*.gguf' -type f 2>/dev/null | sort | head -n 1 || true)

if [[ -z "$first_shard" ]]; then
    echo "Could not find first GGUF shard under: $model_dir" >&2
    echo "Run scripts/download_kimi_k2_7_code_gguf.sh $quant $model_root first." >&2
    exit 1
fi

if ! command -v llama-server >/dev/null 2>&1; then
    echo "Missing llama-server in PATH." >&2
    echo "Install llama.cpp, for example: brew install llama.cpp" >&2
    exit 127
fi

echo "Starting llama-server"
echo "Model: $first_shard"
echo "Host: $host"
echo "Port: $port"
echo "Context: $ctx"
if [[ ${#extra_args[@]} -gt 0 ]]; then
    echo "Extra args: ${extra_args[*]}"
fi

exec llama-server \
    -m "$first_shard" \
    --host "$host" \
    --port "$port" \
    -c "$ctx" \
    "${extra_args[@]}"
