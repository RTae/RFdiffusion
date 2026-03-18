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
nsys profile python scripts/run_inference_profile.py \
  'contigmap.contigs=[200-200]' \
  inference.output_prefix=outputs/profile_large/test \
  inference.num_designs=1 \
  profiler.enabled=false
```

Extra large

```bash
python scripts/run_inference_profile.py \
  'contigmap.contigs=[400-400]' \
  inference.output_prefix=outputs/profile_xlarge/test \
  inference.num_designs=1 \
  profiler.enabled=true
```
