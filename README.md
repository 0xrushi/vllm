# AMD Strix Halo — vLLM Toolbox/Container (gfx1151, PyTorch + AOTriton)

An **Arch-based** Docker/Podman container that is **Toolbx-compatible** (usable as a Fedora toolbox) for serving LLMs with **vLLM** on **AMD Ryzen AI Max “Strix Halo” (gfx1151)**. Built on the PyTorch + AOTriton base to make ROCm on Strix Halo practical for day‑to‑day use.

> **Built on:** [https://github.com/kyuz0/amd-strix-halo-pytorch-gfx1151-aotriton](https://github.com/kyuz0/amd-strix-halo-pytorch-gfx1151-aotriton)
> **Credits:** **lhl** (build tools/scripts), **ssweens** (Arch‑based Dockerfiles), and the **AMD Strix Halo Home Lab Discord** for testing/support.

---

## 1) Toolbx vs Docker/Podman

The `kyuz0/pytorch-therock-gfx1151-aotriton-builder` image can be used both as: 

## &#x20;

* **Fedora Toolbx (recommended for development):** Toolbx shares your **HOME** and user, so models/configs live on the host. Great for iterating quickly while keeping the host clean. 
* **Docker/Podman (recommended for deployment/perf):** Use for running vLLM as a service (host networking, IPC tuning, etc.). Always **mount a host directory** for model weights so they stay outside the container.

---

## 2) Quickstart — Fedora Toolbx (development)

Create a toolbox that exposes the GPU and relaxes seccomp to avoid ROCm syscall issues:

```bash
toolbox create vllm \
  --image docker.io/kyuz0/vllm-therock-gfx1151-aotriton:latest \
  -- --device /dev/dri --device /dev/kfd \
  --group-add video --group-add render --security-opt seccomp=unconfined
```

Enter it:

```bash
toolbox enter vllm
```

**Model storage (Toolbx):** keep weights **outside** the toolbox under your HOME so they persist. Recommended path:

```bash
mkdir -p ~/vllm-models
```

Serve a model with vLLM (downloads to `~/vllm-models`; if the model isn't present, it will be fetched from Hugging Face automatically):

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --download-dir ~/vllm-models
```

> Toolbx shares HOME by design, so `~/vllm-models` stays on the host and survives toolbox updates.
>
> **Cache note (Toolbx):** vLLM will also write compiled kernels to `~/.cache/vllm/torch_compile_cache/` in your HOME. For example:
>
> ```bash
> du -sh ~/.cache/vllm/torch_compile_cache/
> # e.g., 138M  /home/kyuz0/.cache/vllm/torch_compile_cache/
> ```

---

## 3) Testing the API

Once the server is up (from section 2), hit the OpenAI‑compatible endpoint:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-7B-Instruct","messages":[{"role":"user","content":"Hello! Test the performance."}]}'
```

You should receive a JSON response with a `choices[0].message.content` reply.

---

## 4) Quickstart — Podman/Docker

Prefer this for persistent services. **Always mount a host directory for weights** so they live outside the container. If the model isn't present, vLLM will fetch it from **Hugging Face** into the mapped directory.

```bash
podman run \
  -d \
  --name vllm \
  --network host \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add video \
  --group-add render \
  -v ~/vllm-models:/models \
  -v ~/.cache/vllm:/root/.cache/vllm \
  docker.io/kyuz0/vllm-therock-gfx1151-aotriton:latest \
  bash -lc 'source /torch-therock/.venv/bin/activate; \
    TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
    vllm serve Qwen/Qwen2.5-7B-Instruct --dtype float16 \
      --host 0.0.0.0 --port 8000 --download-dir /models'
```

> Not using `--network host`? Map a port instead: `-p 8000:8000`.

---

## 5) Models, dtypes & storage

* Start with **Qwen/Qwen2.5-7B-Instruct**; larger models may work but are less forgiving on unified memory.
* Use `--dtype float16` unless you have a reason to change.
* **Storage discipline:**

  * **Toolbx:** `--download-dir ~/vllm-models` (lives in your HOME on the host).
  * **Podman/Docker:** `-v ~/vllm-models:/models` and `--download-dir /models`.

---

## 6) Performance notes (short)

* The image is built on the PyTorch + **AOTriton** base; enabling `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` can improve startup/throughput on some models.
* vLLM flags you might tune later: `--gpu-memory-utilization`, `--max-num-seqs`, `--max-model-len`. Start simple; add knobs only if needed.

---

## 7) Requirements (host)

**Hardware & drivers**

* AMD Strix Halo APU (gfx1151).
* Working amdgpu stack with `/dev/kfd` (ROCm compute) and `/dev/dri` (graphics).
* Your user in the **video** and **render** groups.

**Unified memory setup (HIGHLY recommended)**
Enable large GTT/unified memory so the iGPU can borrow system RAM for bigger models:

1. **Kernel parameters** (append to your GRUB cmdline):

   ```
   amd_iommu=off amdgpu.gttsize=131072 ttm.pages_limit=33554432
   ```

   | Parameter                  | Purpose                      |
   | -------------------------- | ---------------------------- |
   | `amd_iommu=off`            | Reduces latency              |
   | `amdgpu.gttsize=131072`    | 128 GiB GTT (unified memory) |
   | `ttm.pages_limit=33554432` | Large pinned allocations     |

2. **BIOS**: allocate **minimal VRAM** to the iGPU (e.g., **512 MB**) and rely on unified memory.

3. **Fedora example** (GRUB): edit `/etc/default/grub` → `GRUB_CMDLINE_LINUX=...` then:

   ```bash
   sudo grub2-mkconfig -o /boot/grub2/grub.cfg
   sudo reboot
   ```

**Container runtime**

* Podman or Docker installed (examples use Podman; replace with Docker if preferred).

---

## 8) Acknowledgements & Links

* Base images & docs: [https://github.com/kyuz0/amd-strix-halo-pytorch-gfx1151-aotriton](https://github.com/kyuz0/amd-strix-halo-pytorch-gfx1151-aotriton)
* Upstreams: [vLLM](https://github.com/vllm-project/vllm), [ROCm/TheRock](https://github.com/ROCm/TheRock), [AOTriton](https://github.com/ROCm/aotriton)
* Community: **AMD Strix Halo Home Lab Discord** — [https://discord.gg/pnPRyucNrG](https://discord.gg/pnPRyucNrG)
* Big thanks to **lhl** and **ssweens** for prior art and inspiration.
