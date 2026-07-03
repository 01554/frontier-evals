#!/usr/bin/env bash
set -euo pipefail

quant=${1:-UD-Q3_K_XL}
target_dir=${2:-$HOME/mac_workspace/models/kimi-k2.7-code-gguf}
repo=${HF_REPO:-unsloth/Kimi-K2.7-Code-GGUF}

case "$quant" in
    UD-Q3_K_XL|UD-Q3_K_M)
        expected_files=11
        ;;
    UD-IQ3_S)
        expected_files=10
        ;;
    UD-IQ3_XXS)
        expected_files=9
        ;;
    *)
        expected_files=""
        ;;
esac

if ! command -v hf >/dev/null 2>&1; then
    if command -v uv >/dev/null 2>&1; then
        echo "Installing huggingface_hub CLI with uv..."
        uv tool install "huggingface_hub[hf_transfer]"
        export PATH="$HOME/.local/bin:$PATH"
    else
        echo "Missing hf and uv. Install huggingface_hub first." >&2
        exit 127
    fi
fi

mkdir -p "$target_dir"

echo "Repository: $repo"
echo "Quantization folder: $quant"
echo "Target directory: $target_dir"
df -h "$target_dir" || true

HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}" \
hf download "$repo" \
    --include "${quant}/*" \
    --local-dir "$target_dir"

echo
echo "Downloaded files:"
find "$target_dir/$quant" -maxdepth 1 -name '*.gguf' -type f | sort

actual_files=$(find "$target_dir/$quant" -maxdepth 1 -name '*.gguf' -type f | wc -l | tr -d ' ')
if [[ -n "$expected_files" && "$actual_files" != "$expected_files" ]]; then
    echo "Warning: expected $expected_files GGUF shards for $quant, found $actual_files." >&2
    exit 1
fi

first_shard=$(find "$target_dir/$quant" -maxdepth 1 -name '*-00001-of-*.gguf' -type f | sort | head -n 1)
if [[ -z "$first_shard" ]]; then
    echo "Could not find first shard under $target_dir/$quant" >&2
    exit 1
fi

echo
echo "First shard for llama-server:"
echo "$first_shard"
