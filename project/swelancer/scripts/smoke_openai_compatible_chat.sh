#!/usr/bin/env bash
set -euo pipefail

base_url=${OPENAI_BASE_URL:-http://localhost:4321/v1}
api_key=${OPENAI_API_KEY:-llama.cpp}
model=${MODEL:-kimi-k2.7-code-q3}
max_tokens=${MAX_TOKENS:-128}

payload=$(
    jq -n \
        --arg model "$model" \
        --arg max_tokens "$max_tokens" \
        '{
            model: $model,
            messages: [{role: "user", content: "Say OK only."}],
            temperature: 0,
            max_tokens: ($max_tokens | tonumber)
        }'
)

response=$(
    curl -sS \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer ${api_key}" \
        -d "$payload" \
        "${base_url%/}/chat/completions"
)

content=$(printf '%s\n' "$response" | jq -r '.choices[0].message.content // empty')
if [[ -z "$content" ]]; then
    echo "$response"
    exit 1
fi

printf '%s\n' "$content"
