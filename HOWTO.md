# How to test

## Start env

1. Workspace docker image
```bash
docker build \
-f docker/RTX-5090.dockerfile \
-t rfdiffusion_workspace .
```

2. How to run
```bash
docker run --gpus all -it \
-e PYTHONPATH=/workspace/RFdiffusion \
-v "$PWD":/workspace/RFdiffusion \
-w /workspace/RFdiffusion \
rfdiffusion_workspace
```

## Run a benchmark
1. Small
```bash
python scripts/run_inference_benchmark.py \
  'contigmap.contigs=[50-50]' \
  inference.output_prefix=outputs/bench_small/test \
  inference.num_designs=1
```
2. Medium
```bash
python scripts/run_inference_benchmark.py \
  'contigmap.contigs=[100-100]' \
  inference.output_prefix=outputs/bench_medium/test \
  inference.num_designs=1
```
3. Large
```bash
python scripts/run_inference_benchmark.py \
  'contigmap.contigs=[200-200]' \
  inference.output_prefix=outputs/bench_large/test \
  inference.num_designs=1
```
4. Extra large
```bash
python scripts/run_inference_benchmark.py \
  'contigmap.contigs=[400-400]' \
  inference.output_prefix=outputs/bench_xlarge/test \
  inference.num_designs=1
```
## Run a profile
1. Large
```bash
python scripts/run_inference_profile.py \
  'contigmap.contigs=[200-200]' \
  inference.output_prefix=outputs/profile_large/test \
  inference.num_designs=1 \
  profiler.enabled=true
```

2. Extra large
```bash
python scripts/run_inference_profile.py \
  'contigmap.contigs=[400-400]' \
  inference.output_prefix=outputs/profile_xlarge/test \
  inference.num_designs=1 \
  profiler.enabled=true
```

