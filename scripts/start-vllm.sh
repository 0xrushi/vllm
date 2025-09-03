#!/usr/bin/env bash
set -euo pipefail

# start_vllm — interactive helper to launch vLLM on AMD Strix Halo (gfx1151)
# - Presents a curated list of recent HF models that fit within ~100GB memory (with FP16 or AWQ)
# - Asks for context length, concurrency, kv‑cache dtype, port, etc.
# - Starts vLLM with sensible ROCm defaults for Strix Halo
#
# Requirements inside the toolbox/container:
#   - vLLM installed in /torch-therock/.venv (this image has it)
#   - internet for first model download (or pre‑downloaded into ~/vllm-models)
#   - optional: ~/.cache/vllm mapped to persist compile cache when using Podman/Docker
#
# Notes on quantization:
#   - vLLM supports weight‑only quantized models like AWQ and GPTQ (load pre‑quantized repos).
#   - For AMD GPUs, FP8 KV‑cache can be supported but is experimental on consumer APUs; INT8 KV‑cache is a safer saver.
#   - Qwen3 provides AWQ variants officially; using them can materially reduce memory use. (You do NOT need GGUF; that is for llama.cpp.)
#
# Model memory rule of thumb (VERY rough):
#   - FP16 weights ≈ 2 bytes/parameter. So 12B ≈ ~24 GB; 27B ≈ ~54 GB; 32B ≈ ~64 GB (weights only).
#   - Plus KV‑cache, which grows with context & concurrency. If you OOM, lower max context or max concurrent requests.
#
# Default directories
DOWNLOAD_DIR="${DOWNLOAD_DIR:-$HOME/vllm-models}"
CACHE_DIR_DEFAULT="$HOME/.cache/vllm"
PORT_DEFAULT="8000"
HOST_DEFAULT="0.0.0.0"
GPU_UTIL_DEFAULT="0.92"
MAX_NUM_SEQS_DEFAULT="4"
MAX_MODEL_LEN_DEFAULT="16384"
KV_CACHE_DTYPE_DEFAULT="auto"   # choices: auto|int8|fp8 (fp8_e4m3)
DTYPE_DEFAULT="float16"         # choices: float16|bfloat16

VENV_ACTIVATE="/torch-therock/.venv/bin/activate"
if [[ -f "$VENV_ACTIVATE" ]]; then
  # shellcheck disable=SC1090
  source "$VENV_ACTIVATE"
fi

print_divider() { printf '\n%s\n' "────────────────────────────────────────────────────────"; }

# --- curated model list (recent, likely to fit <= ~100GB with sane settings) ---
# Format: label|hf_repo|quant_hint|compat|note
MODELS=(
  "Llama-4 Scout 17B-16E Instruct FP4|nvidia/Llama-4-Scout-17B-16E-Instruct-FP4|modelopt|nvidia_only|Optimized for NVIDIA; FP4 path may not work on AMD/ROCm"
  "Llama-4 Scout 17B-16E Instruct FP8|nvidia/Llama-4-Scout-17B-16E-Instruct-FP8|modelopt|nvidia_only|Optimized for NVIDIA; FP8 ModelOpt path may not work on AMD/ROCm"
  "OpenAI GPT-OSS 20B (MXFP4)|openai/gpt-oss-20b|mxfp4|experimental|MXFP4 support requires recent vLLM; performance/compat on AMD RDNA iGPU varies"
  "OpenAI GPT-OSS 120B (MXFP4, huge)|openai/gpt-oss-120b|mxfp4|too_large|~120B total params; not practical on a single APU"
  "GLM-4.5-Air FP8 (12B active)|zai-org/GLM-4.5-Air-FP8|fp8|multi_gpu_fp8|Published FP8; vendor recommends multi-GPU with native FP8"
  "Gemma 3 12B IT (FP16)|google/gemma-3-12b-it|fp16|amd_ok|Good baseline"
  "Gemma 3 27B IT (FP16)|google/gemma-3-27b-it|fp16|borderline|Large; consider GPTQ variant if memory tight"
  "Gemma 3 27B IT (GPTQ 4bit)|ISTA-DASLab/gemma-3-27b-it-GPTQ-4b-128g|gptq|amd_ok|Weight-only INT4 reduces memory; throughput may drop"
  "Qwen3 8B Instruct (FP16)|Qwen/Qwen3-8B-Instruct|fp16|amd_ok|Solid quality, easy fit"
  "Qwen3 8B Instruct (AWQ 4bit)|Qwen/Qwen3-8B-AWQ|awq|amd_ok|Official AWQ"
  "Qwen3 14B Instruct (FP16)|Qwen/Qwen3-14B-Instruct|fp16|amd_ok|"
  "Qwen3 14B Instruct (AWQ 4bit)|Qwen/Qwen3-14B-AWQ|awq|amd_ok|"
  "Qwen3 30B A3B Instruct (FP16)|Qwen/Qwen3-30B-A3B-Instruct-2507|fp16|amd_ok|MoE; fits with careful context/concurrency"
  "Qwen3 30B A3B Instruct (AWQ 4bit)|cpatonn/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit|awq|community|Community AWQ; quality varies"
)


cat <<'HDR'
Start vLLM — AMD Strix Halo (gfx1151)
This helper will:
  1) Let you pick a model (FP16 or AWQ when available)
  2) Ask for context length, concurrency, and KV‑cache dtype
  3) Launch vLLM with Strix‑friendly defaults
HDR

print_divider
printf 'Model download dir (persisted on host) [%s]: ' "$DOWNLOAD_DIR"
read -r REPLY_DL
[[ -n "${REPLY_DL:-}" ]] && DOWNLOAD_DIR="$REPLY_DL"
mkdir -p "$DOWNLOAD_DIR"

printf 'Cache dir for compiled kernels [%s]: ' "$CACHE_DIR_DEFAULT"
read -r REPLY_CACHE
[[ -n "${REPLY_CACHE:-}" ]] && export VLLM_CACHE_DIR="$REPLY_CACHE" || export VLLM_CACHE_DIR="$CACHE_DIR_DEFAULT"
mkdir -p "$VLLM_CACHE_DIR"

print_divider
printf 'Select a model:\n'
idx=1
for m in "${MODELS[@]}"; do
  IFS='|' read -r label _ _ <<<"$m"
  printf '  [%d] %s\n' "$idx" "$label"
  idx=$((idx+1))
done

printf 'Enter number: '
read -r CHOICE
if ! [[ "$CHOICE" =~ ^[0-9]+$ ]] || (( CHOICE < 1 || CHOICE > ${#MODELS[@]} )); then
  echo 'Invalid choice.'; exit 1
fi
SEL="${MODELS[$((CHOICE-1))]}"
IFS='|' read -r SEL_LABEL HF_REPO QUANT_HINT COMPAT NOTE <<<"$SEL"

# Quantization flag heuristic
QUANT_FLAG=()
case "$QUANT_HINT" in
  awq) QUANT_FLAG=(--quantization awq) ;;
  gptq) QUANT_FLAG=(--quantization gptq) ;;
  mxfp4) QUANT_FLAG=(--quantization mxfp4) ;;
  modelopt) QUANT_FLAG=(--quantization modelopt) ;;
  fp16|fp8|bf16|auto|'') ;; # rely on model config
esac

# Compatibility warnings
case "$COMPAT" in
  nvidia_only)
    echo "WARNING: This checkpoint is optimized for NVIDIA (TensorRT/ModelOpt). It may not run on AMD ROCm (RDNA iGPU)." ;;
  multi_gpu_fp8)
    echo "WARNING: Vendor docs indicate multi‑GPU FP8 is recommended. On a single Strix Halo APU this is likely impractical." ;;
  too_large)
    echo "WARNING: 120B‑class model is far beyond single‑APU capacity. Expect failure unless heavy offload/sharding is used." ;;
  borderline)
    echo "Note: Large model — keep context/concurrency modest or use a quantized variant." ;;
  community)
    echo "Note: Community quantization — quality/perf may vary." ;;
  amd_ok|*) ;;
esac

[[ -n "$NOTE" ]] && echo "Note: $NOTE"

print_divider
printf 'Max context tokens (--max-model-len) [%s]: ' "$MAX_MODEL_LEN_DEFAULT"
read -r REPLY_CTX
MAX_MODEL_LEN="${REPLY_CTX:-$MAX_MODEL_LEN_DEFAULT}"

printf 'Max concurrent requests (--max-num-seqs) [%s]: ' "$MAX_NUM_SEQS_DEFAULT"
read -r REPLY_CONC
MAX_NUM_SEQS="${REPLY_CONC:-$MAX_NUM_SEQS_DEFAULT}"

printf 'KV cache dtype (auto|int8|fp8) [%s]: ' "$KV_CACHE_DTYPE_DEFAULT"
read -r REPLY_KV
KV_CACHE_DTYPE="${REPLY_KV:-$KV_CACHE_DTYPE_DEFAULT}"

printf 'Model dtype (float16|bfloat16) [%s]: ' "$DTYPE_DEFAULT"
read -r REPLY_DTYPE
DTYPE="${REPLY_DTYPE:-$DTYPE_DEFAULT}"

printf 'GPU memory utilization (0.50‑0.98) [%s]: ' "$GPU_UTIL_DEFAULT"
read -r REPLY_UTIL
GPU_UTIL="${REPLY_UTIL:-$GPU_UTIL_DEFAULT}"

printf 'Host bind address [%s]: ' "$HOST_DEFAULT"
read -r REPLY_HOST
HOST="${REPLY_HOST:-$HOST_DEFAULT}"

printf 'Optional CPU offload in GB (0 to disable) [0]: '
read -r REPLY_OFF
CPU_OFFLOAD_GB="${REPLY_OFF:-0}"

printf 'Port [%s]: ' "$PORT_DEFAULT"
read -r REPLY_PORT
PORT="${REPLY_PORT:-$PORT_DEFAULT}"

print_divider
CMD=(
  vllm serve "$HF_REPO"
  --host "$HOST"
  --port "$PORT"
  --download-dir "$DOWNLOAD_DIR"
  --dtype "$DTYPE"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --gpu-memory-utilization "$GPU_UTIL"
)

# Add CPU offload if requested
if [[ "$CPU_OFFLOAD_GB" =~ ^[0-9]+$ ]] && (( CPU_OFFLOAD_GB > 0 )); then
  CMD+=(--cpu-offload-gb "$CPU_OFFLOAD_GB")
fi

# kv‑cache dtype
if [[ "$KV_CACHE_DTYPE" != "auto" ]]; then
  # Map fp8 -> fp8_e4m3 for AMD unless user typed explicit subtype already
  if [[ "$KV_CACHE_DTYPE" == "fp8" ]]; then
    CMD+=(--kv-cache-dtype fp8_e4m3)
  else
    CMD+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
  fi
fi

# quantization flags (if any)
CMD+=("${QUANT_FLAG[@]}")

# AMD ROCm/AOTriton helpful env
export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx1151}"
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1

printf 'About to run:\n\n  %q' "${CMD[0]}"; for ((i=1;i<${#CMD[@]};i++)); do printf ' \\\n    %q' "${CMD[$i]}"; done; printf '\n\n'

read -r -p "Proceed? [Y/n] " yn
yn=${yn:-Y}
if [[ "$yn" =~ ^[Yy]$ ]]; then
  exec "${CMD[@]}"
else
  echo "Canceled."
fi
