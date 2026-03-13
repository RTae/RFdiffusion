#!/usr/bin/env python
"""
Benchmarking version of RFdiffusion inference.

Purpose:
- Keep original run_inference.py untouched
- Add Day-1 profiling:
  - total runtime per design
  - per-step timing
  - peak GPU memory
  - CSV and JSON outputs

Example:
python scripts/run_inference_benchmark.py \
  'contigmap.contigs=[150-150]' \
  inference.output_prefix=test_outputs_bench/test \
  inference.num_designs=3

Notes:
- This follows the structure of the upstream RFdiffusion run_inference.py
- It adds logging only; no model logic changes
"""

import csv
import glob
import json
import logging
import os
import pickle
import random
import re
import time
from typing import Any

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from rfdiffusion.inference import utils as iu
from rfdiffusion.util import writepdb, writepdb_multi


def make_deterministic(seed: int = 0) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def cuda_sync_if_available() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def get_mem_stats_mb() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {
            "mem_alloc_mb": 0.0,
            "mem_reserved_mb": 0.0,
            "max_mem_alloc_mb": 0.0,
            "max_mem_reserved_mb": 0.0,
        }

    return {
        "mem_alloc_mb": torch.cuda.memory_allocated() / 1024 / 1024,
        "mem_reserved_mb": torch.cuda.memory_reserved() / 1024 / 1024,
        "max_mem_alloc_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
        "max_mem_reserved_mb": torch.cuda.max_memory_reserved() / 1024 / 1024,
    }


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def append_csv_row(csv_path: str, row: dict[str, Any], fieldnames: list[str]) -> None:
    ensure_parent_dir(csv_path)
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


@hydra.main(version_base=None, config_path="../config/inference", config_name="base")
def main(conf: HydraConfig) -> None:
    log = logging.getLogger(__name__)

    if conf.inference.deterministic:
        make_deterministic()

    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(torch.cuda.current_device())
        log.info(f"Found GPU with device_name {device_name}.")
        log.info(f"Will run RFdiffusion benchmark on {device_name}")
    else:
        device_name = "CPU"
        log.info("////////////////////////////////////////////////")
        log.info("///// NO GPU DETECTED! Falling back to CPU /////")
        log.info("////////////////////////////////////////////////")

    sampler = iu.sampler_selector(conf)

    design_startnum = sampler.inf_conf.design_startnum
    if sampler.inf_conf.design_startnum == -1:
        existing = glob.glob(sampler.inf_conf.output_prefix + "*.pdb")
        indices = [-1]
        for e in existing:
            m = re.match(r".*_(\d+)\.pdb$", e)
            if not m:
                continue
            indices.append(int(m.groups()[0]))
        design_startnum = max(indices) + 1

    # Benchmark output paths
    out_root = os.path.dirname(sampler.inf_conf.output_prefix) or "."
    bench_dir = os.path.join(out_root, "benchmark_logs")
    os.makedirs(bench_dir, exist_ok=True)

    summary_csv = os.path.join(bench_dir, "design_summary.csv")
    steps_csv = os.path.join(bench_dir, "step_timing.csv")

    summary_fields = [
        "design_id",
        "output_prefix",
        "device",
        "contigs",
        "num_steps",
        "total_runtime_sec",
        "total_runtime_min",
        "peak_gpu_mem_mb",
        "peak_gpu_reserved_mb",
        "avg_step_time_ms",
        "min_step_time_ms",
        "max_step_time_ms",
    ]

    step_fields = [
        "design_id",
        "step_idx",
        "t",
        "step_time_ms",
        "mem_alloc_mb",
        "mem_reserved_mb",
        "max_mem_alloc_mb",
        "max_mem_reserved_mb",
    ]

    for i_des in range(design_startnum, design_startnum + sampler.inf_conf.num_designs):
        if conf.inference.deterministic:
            make_deterministic(i_des)

        out_prefix = f"{sampler.inf_conf.output_prefix}_{i_des}"
        log.info(f"Making design {out_prefix}")

        if sampler.inf_conf.cautious and os.path.exists(out_prefix + ".pdb"):
            log.info(
                f"(cautious mode) Skipping this design because {out_prefix}.pdb already exists."
            )
            continue

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        cuda_sync_if_available()
        design_t0 = time.perf_counter()

        x_init, seq_init = sampler.sample_init()

        denoised_xyz_stack = []
        px0_xyz_stack = []
        seq_stack = []
        plddt_stack = []

        x_t = torch.clone(x_init)
        seq_t = torch.clone(seq_init)

        per_step_rows: list[dict[str, Any]] = []
        step_times_ms: list[float] = []

        t_values = list(range(int(sampler.t_step_input), sampler.inf_conf.final_step - 1, -1))
        num_steps = len(t_values)

        for step_idx, t in enumerate(t_values):
            cuda_sync_if_available()
            step_t0 = time.perf_counter()

            px0, x_t, seq_t, plddt = sampler.sample_step(
                t=t,
                x_t=x_t,
                seq_init=seq_t,
                final_step=sampler.inf_conf.final_step,
            )

            cuda_sync_if_available()
            step_t1 = time.perf_counter()

            px0_xyz_stack.append(px0)
            denoised_xyz_stack.append(x_t)
            seq_stack.append(seq_t)
            plddt_stack.append(plddt[0])  # remove singleton leading dimension

            step_time_ms = (step_t1 - step_t0) * 1000.0
            step_times_ms.append(step_time_ms)
            mem_stats = get_mem_stats_mb()

            row = {
                "design_id": i_des,
                "step_idx": step_idx,
                "t": t,
                "step_time_ms": round(step_time_ms, 4),
                "mem_alloc_mb": round(mem_stats["mem_alloc_mb"], 2),
                "mem_reserved_mb": round(mem_stats["mem_reserved_mb"], 2),
                "max_mem_alloc_mb": round(mem_stats["max_mem_alloc_mb"], 2),
                "max_mem_reserved_mb": round(mem_stats["max_mem_reserved_mb"], 2),
            }
            per_step_rows.append(row)
            append_csv_row(steps_csv, row, step_fields)

            log.info(
                f"[design {i_des}] step_idx={step_idx:03d} t={t:03d} "
                f"step_time_ms={step_time_ms:.2f} "
                f"mem_alloc_mb={mem_stats['mem_alloc_mb']:.2f} "
                f"max_mem_alloc_mb={mem_stats['max_mem_alloc_mb']:.2f}"
            )

        # Flip order for visualization
        denoised_xyz_stack = torch.flip(torch.stack(denoised_xyz_stack), [0])
        px0_xyz_stack = torch.flip(torch.stack(px0_xyz_stack), [0])
        plddt_stack = torch.stack(plddt_stack)

        ensure_parent_dir(out_prefix)
        final_seq = seq_stack[-1]

        # Output glycines except motif region
        final_seq = torch.where(
            torch.argmax(seq_init, dim=-1) == 21,
            7,
            torch.argmax(seq_init, dim=-1),
        )

        bfacts = torch.ones_like(final_seq.squeeze())
        bfacts[torch.where(torch.argmax(seq_init, dim=-1) == 21, True, False)] = 0

        out = f"{out_prefix}.pdb"
        writepdb(
            out,
            denoised_xyz_stack[0, :, :4],
            final_seq,
            sampler.binderlen,
            chain_idx=sampler.chain_idx,
            bfacts=bfacts,
            idx_pdb=sampler.idx_pdb,
        )

        cuda_sync_if_available()
        design_t1 = time.perf_counter()
        total_runtime_sec = design_t1 - design_t0

        mem_stats = get_mem_stats_mb()
        peak_gpu_mem_mb = mem_stats["max_mem_alloc_mb"]
        peak_gpu_reserved_mb = mem_stats["max_mem_reserved_mb"]

        trb = dict(
            config=OmegaConf.to_container(sampler._conf, resolve=True),
            plddt=plddt_stack.cpu().numpy(),
            device=device_name,
            time=total_runtime_sec,
        )

        if hasattr(sampler, "contig_map"):
            for key, value in sampler.contig_map.get_mappings().items():
                trb[key] = value

        with open(f"{out_prefix}.trb", "wb") as f_out:
            pickle.dump(trb, f_out)

        if sampler.inf_conf.write_trajectory:
            traj_prefix = os.path.dirname(out_prefix) + "/traj/" + os.path.basename(out_prefix)
            os.makedirs(os.path.dirname(traj_prefix), exist_ok=True)

            out = f"{traj_prefix}_Xt-1_traj.pdb"
            writepdb_multi(
                out,
                denoised_xyz_stack,
                bfacts,
                final_seq.squeeze(),
                use_hydrogens=False,
                backbone_only=False,
                chain_ids=sampler.chain_idx,
            )

            out = f"{traj_prefix}_pX0_traj.pdb"
            writepdb_multi(
                out,
                px0_xyz_stack,
                bfacts,
                final_seq.squeeze(),
                use_hydrogens=False,
                backbone_only=False,
                chain_ids=sampler.chain_idx,
            )

        avg_step = sum(step_times_ms) / len(step_times_ms) if step_times_ms else 0.0
        min_step = min(step_times_ms) if step_times_ms else 0.0
        max_step = max(step_times_ms) if step_times_ms else 0.0

        contigs = None
        try:
            contigs = OmegaConf.to_container(conf.contigmap.contigs, resolve=True)
        except Exception:
            contigs = "unknown"

        summary_row = {
            "design_id": i_des,
            "output_prefix": out_prefix,
            "device": device_name,
            "contigs": str(contigs),
            "num_steps": num_steps,
            "total_runtime_sec": round(total_runtime_sec, 4),
            "total_runtime_min": round(total_runtime_sec / 60.0, 4),
            "peak_gpu_mem_mb": round(peak_gpu_mem_mb, 2),
            "peak_gpu_reserved_mb": round(peak_gpu_reserved_mb, 2),
            "avg_step_time_ms": round(avg_step, 4),
            "min_step_time_ms": round(min_step, 4),
            "max_step_time_ms": round(max_step, 4),
        }
        append_csv_row(summary_csv, summary_row, summary_fields)

        # Also dump JSON per design for easier later analysis
        json_path = os.path.join(bench_dir, f"design_{i_des:04d}.json")
        with open(json_path, "w") as f:
            json.dump(
                {
                    "summary": summary_row,
                    "steps": per_step_rows,
                },
                f,
                indent=2,
            )

        log.info(
            f"Finished design {i_des} in {total_runtime_sec / 60.0:.2f} min | "
            f"peak_gpu_mem_mb={peak_gpu_mem_mb:.2f} | "
            f"avg_step_time_ms={avg_step:.2f}"
        )


if __name__ == "__main__":
    main()