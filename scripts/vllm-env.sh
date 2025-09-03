#!/usr/bin/env bash
# Auto-activate vLLM environment for toolbx

# Activate PyTorch + vLLM environment
source /torch-therock/.venv/bin/activate

# ROCm and performance environment variables
export PYTORCH_ROCM_ARCH=gfx1151
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
export VLLM_USE_TRITON_FLASH_ATTN=0
export TORCH_COMPILE_DEBUG=1
export VLLM_COMPILE_LEVEL=3

# Detect and export ROCm toolchain paths
eval "$(
python3 - <<'PY'
try:
    import pathlib, _rocm_sdk_core as r
    base = pathlib.Path(r.__file__).parent / "lib" / "llvm" / "bin"
    lib  = pathlib.Path(r.__file__).parent / "lib"
    print(f'export TRITON_HIP_LLD_PATH="{base / "ld.lld"}"')
    print(f'export TRITON_HIP_CLANG_PATH="{base / "clang++"}"')
    print(f'export LD_LIBRARY_PATH="{lib}:$LD_LIBRARY_PATH"')
except ImportError:
    pass
PY
)" 2>/dev/null || true

# Enable flash attention
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE