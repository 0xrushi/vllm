#!/usr/bin/env bash
# vLLM Toolbox banner

gpu_name() {
  local name=""
  if command -v rocm-smi >/dev/null 2>&1; then
    name=$(rocm-smi --showproductname --csv 2>/dev/null | tail -n1 | cut -d, -f2)
    [[ -z "$name" ]] && name=$(rocm-smi --showproductname 2>/dev/null | grep -m1 -E 'Product Name|Card series' | sed 's/.*: //')
  fi
  if [[ -z "$name" ]]; then
    name="Unknown AMD GPU"
  fi
  printf '%s\n' "$name"
}

vllm_version() {
  python -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "unknown"
}

# Simple model selector
vllm_start() {
  echo
  echo "Select a model to serve:"
  echo "1) Qwen2.5-7B-Instruct  (recommended, ~14GB VRAM)"
  echo "2) Llama-3.1-8B-Instruct (~16GB VRAM)"  
  echo "3) Qwen3-8B (~16GB VRAM, latest with thinking mode)"
  echo
  read -p "Choose [1-3]: " choice
  
  case $choice in
    1) vllm serve Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 8000 --download-dir ~/models --dtype float16 --max-model-len 32768 ;;
    2) vllm serve meta-llama/Llama-3.1-8B-Instruct --host 0.0.0.0 --port 8000 --download-dir ~/models --dtype float16 --max-model-len 32768 ;;
    3) vllm serve Qwen/Qwen3-8B --host 0.0.0.0 --port 8000 --download-dir ~/models --dtype float16 --max-model-len 32768 --enable-reasoning --reasoning-parser qwen3 ;;
    *) echo "Invalid choice." ;;
  esac
}

GPU="$(gpu_name)"
VLLM_VER="$(vllm_version)"

echo
echo "vLLM Toolbox - AMD STRIX HALO (gfx1151)"
echo "GPU: $GPU"
echo "vLLM: $VLLM_VER"
echo
echo "Commands:"
echo "  vllm_start  - Start model server" 
echo "  vllm_test   - Test API"
echo "  ls ~/models - List downloaded models"
echo
echo "Server will be available at: http://localhost:8000"
echo

# Test alias
alias vllm_test='curl -X POST http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '\''{"model":"auto","messages":[{"role":"user","content":"Hello!"}]}'\'''