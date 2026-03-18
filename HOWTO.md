# How to test

## Start env

Workspace docker image

```bash
docker build \
-f docker/RTX-5090.dockerfile \
-t rfdiffusion_workspace .
```

How to run

```bash
docker run --gpus all -it \
--cap-add=SYS_ADMIN \
--cap-add=SYS_PTRACE \
--security-opt seccomp=unconfined \
--ipc=host \
-e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
-e PYTHONPATH=/workspace/RFdiffusion \
-v "$PWD":/workspace/RFdiffusion \
-w /workspace/RFdiffusion \
rfdiffusion_workspace
```

## Run a benchmark

Small

```bash
python scripts/run_inference_benchmark.py \
  'contigmap.contigs=[50-50]' \
  inference.output_prefix=outputs/bench_small/test \
  inference.num_designs=1
```

Medium

```bash
python scripts/run_inference_benchmark.py \
  'contigmap.contigs=[100-100]' \
  inference.output_prefix=outputs/bench_medium/test \
  inference.num_designs=1
```

Large

```bash
python scripts/run_inference_benchmark.py \
  'contigmap.contigs=[200-200]' \
  inference.output_prefix=outputs/bench_large/test \
  inference.num_designs=1
```

Extra large

```bash
python scripts/run_inference_benchmark.py \
  'contigmap.contigs=[400-400]' \
  inference.output_prefix=outputs/bench_xlarge/test \
  inference.num_designs=1
```

## Run a profile

Check Nsight tools are available

```bash
nsys --version
ncu --version
```

Large

```bash
python scripts/run_inference_profile.py \
  'contigmap.contigs=[200-200]' \
  inference.output_prefix=outputs/profile_large/test \
  inference.num_designs=1 \
  profiler.enabled=true
```

nsys profile
```bash
nsys profile --trace=cuda,nvtx,osrt python scripts/run_inference_profile.py \
  'contigmap.contigs=[200-200]' \
  inference.output_prefix=outputs/profile_large/test \
  inference.num_designs=1 \
  profiler.enabled=false \
  profiler.nvtx_enabled=true \
  profiler.max_steps=10
```

ncu profile
```bash
ncu --set full --target-processes all --force-overwrite --export outputs/profile_large/test_ncu \
python scripts/run_inference_profile.py \
  'contigmap.contigs=[200-200]' \
  inference.output_prefix=outputs/profile_large/test \
  inference.num_designs=1 \
  profiler.enabled=false \
  profiler.nvtx_enabled=true \
  profiler.max_steps=10
```

Extra large

```bash
python scripts/run_inference_profile.py \
  'contigmap.contigs=[400-400]' \
  inference.output_prefix=outputs/profile_xlarge/test \
  inference.num_designs=1 \
  profiler.enabled=true
```
