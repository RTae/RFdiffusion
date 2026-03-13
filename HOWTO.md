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
```bash
python scripts/run_inference_benchmark.py \
  'contigmap.contigs=[50-50]' \
  inference.output_prefix=outputs/bench_small/test \
  inference.num_designs=1
```