#!/usr/bin/env bash
set -euo pipefail

# start_vllm_basic — pick a known-good model, print the vLLM command, run it.
# No extra flags; uses vLLM defaults.

# Optional: activate the toolbox venv if present
if [[ -f "/torch-therock/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "/torch-therock/.venv/bin/activate"
fi

# Only the models you've reported working
MODELS=(
  "meta-llama/Llama-2-7b-chat-hf|Llama 2 7B Chat"
  "Qwen/Qwen2.5-7B-Instruct|Qwen2.5 7B Instruct"
  "Qwen/Qwen3-30B-A3B-Instruct-2507|Qwen3 30B A3B Instruct"
  "Qwen/Qwen3-14B-AWQ|Qwen3 14B AWQ"
)

echo "Select a model:"
for i in "${!MODELS[@]}"; do
  IFS='|' read -r _ label <<<"${MODELS[$i]}"
  printf "  [%d] %s\n" "$((i+1))" "$label"
done

read -rp "Enter number: " choice
if ! [[ "$choice" =~ ^[1-9][0-9]*$ ]] || (( choice < 1 || choice > ${#MODELS[@]} )); then
  echo "Invalid choice." >&2
  exit 1
fi

IFS='|' read -r MODEL _ <<<"${MODELS[$((choice-1))]}"

CMD=(vllm serve "$MODEL")

# Minimal, model-specific additions
if [[ "$MODEL" == "Qwen/Qwen3-14B-AWQ" ]]; then
  # Needed on your ROCm setup for AWQ
  CMD+=(--quantization awq --dtype float16 --enforce-eager)
fi

printf 'Running:\n\n  %q' "${CMD[0]}"; for ((i=1;i<${#CMD[@]};i++)); do printf ' %q' "${CMD[$i]}"; done; printf '\n\n'

exec "${CMD[@]}"
