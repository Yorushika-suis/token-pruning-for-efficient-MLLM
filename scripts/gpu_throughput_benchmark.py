#!/usr/bin/env python3
"""Isolated 1/2/4/8-GPU throughput benchmark for the vision feature path.

This is an infrastructure benchmark, not a Stage 1/2 scientific run.  It uses
the same decode/build/encode/write functions as ``stage2_dump_features.py`` but
writes only to a new benchmark scratch directory.  The orchestrator validates
and then removes exactly the scratch NPZ files it created; logs and JSON
manifests remain under ``runs/gpu_benchmarks/<run_id>/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
VIDEO_DIR = ROOT / "data" / "videos_short"
DEFAULT_ENV_MANIFEST = (
    ROOT
    / "runs/environment"
    / "create-env_REDACTED-POD_20260903T164349+0800.json"
)
RESPONSIBILITY_NOTE = "合作者接入时间未定，GPU 吞吐核验暂由 Codex 代行；这是临时安排。"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from data_manifest import load_video_ids, require_video_files, require_video_subset  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def choose_even_cohort(ids: list[str], count: int) -> list[str]:
    """Choose a deterministic cohort spread over manifest order."""
    if count < 1 or count > len(ids):
        raise ValueError(f"cohort size must be in [1, {len(ids)}], got {count}")
    if count == 1:
        return [ids[0]]
    indices = [i * (len(ids) - 1) // (count - 1) for i in range(count)]
    if len(set(indices)) != count:
        raise RuntimeError(f"cohort indices are not unique: {indices}")
    return [ids[i] for i in indices]


def partition_round_robin(items: list[str], workers: int) -> list[list[str]]:
    if workers < 1 or workers > len(items):
        raise ValueError("workers must be positive and no larger than item count")
    parts = [items[i::workers] for i in range(workers)]
    if any(not part for part in parts):
        raise RuntimeError("empty worker assignment")
    flattened = [item for part in parts for item in part]
    if sorted(flattened) != sorted(items) or len(flattened) != len(set(flattened)):
        raise RuntimeError("worker assignments are not an exact partition")
    return parts


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return float(ordered[index])


def query_gpu_inventory() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    rows = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        index, uuid, name, memory_total = [field.strip() for field in line.split(",", 3)]
        rows.append(
            {
                "physical_index": int(index),
                "uuid": uuid,
                "name": name,
                "memory_total_mib": int(memory_total),
            }
        )
    return rows


def query_compute_processes() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,used_memory",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    rows = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        uuid, pid, memory = [field.strip() for field in line.split(",", 2)]
        rows.append({"gpu_uuid": uuid, "pid": int(pid), "used_memory_mib": int(memory)})
    return rows


def query_gpu_utilization() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,utilization.gpu,memory.used",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    rows = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        index, uuid, utilization, memory_used = [field.strip() for field in line.split(",", 3)]
        rows.append(
            {
                "physical_index": int(index),
                "uuid": uuid,
                "utilization_gpu_percent": float(utilization),
                "memory_used_mib": float(memory_used),
            }
        )
    return rows


class UtilizationSampler:
    def __init__(self, physical_indices: set[int], interval_seconds: float = 0.5):
        self.physical_indices = physical_indices
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            captured_at = datetime.now().astimezone().isoformat(timespec="milliseconds")
            try:
                for row in query_gpu_utilization():
                    if row["physical_index"] in self.physical_indices:
                        self.samples.append({"captured_at": captured_at, **row})
            except Exception as exc:  # Evidence is preserved; worker status remains separate.
                self.errors.append(f"{type(exc).__name__}: {exc}")
            self._stop.wait(self.interval_seconds)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=max(5.0, self.interval_seconds * 4))
        if self._thread.is_alive():
            raise RuntimeError("GPU utilization sampler did not stop")


def worker_main(args: argparse.Namespace) -> int:
    # CUDA_VISIBLE_DEVICES is set by the orchestrator before this interpreter starts.
    import numpy as np
    import torch

    import adts_harness as H
    import stage2_dump_features as dump

    print(f"responsibility_note={RESPONSIBILITY_NOTE}", flush=True)

    assignment_path = Path(args.assignment).resolve()
    output_path = Path(args.output).resolve()
    scratch_dir = Path(args.scratch_dir).resolve()
    started_epoch = time.time()
    started_iso = datetime.now().astimezone().isoformat(timespec="milliseconds")
    videos: list[str] = []
    records: list[dict[str, Any]] = []
    created_files: list[str] = []
    status = "failed"
    error: dict[str, str] | None = None
    model_load_seconds = 0.0
    processing_started_epoch: float | None = None
    processing_ended_epoch: float | None = None

    try:
        assignment = json.loads(assignment_path.read_text(encoding="utf-8"))
        videos = require_video_subset(assignment["video_ids"], load_video_ids(), VIDEO_DIR)
        scratch_dir.mkdir(parents=True, exist_ok=False)
        if torch.cuda.device_count() != 1:
            raise RuntimeError(
                f"worker expected one visible GPU, got {torch.cuda.device_count()} "
                f"with CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
            )
        torch.cuda.set_device(0)
        load_started = time.perf_counter()
        H_module, model, image_processor = dump.build("cuda:0")
        torch.cuda.synchronize()
        model_load_seconds = time.perf_counter() - load_started

        processing_started_epoch = time.time()
        for video_id in videos:
            video_path = VIDEO_DIR / f"{video_id}.mp4"

            decode_started = time.perf_counter()
            frames = H.load_video_frames(str(video_path))
            decode_seconds = time.perf_counter() - decode_started

            torch.cuda.synchronize()
            encode_started = time.perf_counter()
            arrays = dump.encode_all(
                H_module, model, image_processor, frames, "cuda:0", torch.float16
            )
            torch.cuda.synchronize()
            encode_seconds = time.perf_counter() - encode_started

            final_path = scratch_dir / f"{video_id}.npz"
            temporary_path = scratch_dir / f".{video_id}.{os.getpid()}.tmp"
            write_started = time.perf_counter()
            with temporary_path.open("wb") as handle:
                np.savez(
                    handle,
                    pooled169=arrays[0],
                    attn169=arrays[1],
                    sig729=arrays[2],
                    attn729=arrays[3],
                    meta=np.array(
                        [dump.VERSION, dump.NUM_FRAMES, dump.NV, dump.NP729, 3584, 1152],
                        dtype=np.int32,
                    ),
                )
            os.replace(temporary_path, final_path)
            write_seconds = time.perf_counter() - write_started
            created_files.append(str(final_path))
            records.append(
                {
                    "video_id": video_id,
                    "video_size_bytes": video_path.stat().st_size,
                    "decode_seconds": decode_seconds,
                    "encode_seconds": encode_seconds,
                    "write_seconds": write_seconds,
                    "output_size_bytes": final_path.stat().st_size,
                    "array_shapes": [list(array.shape) for array in arrays],
                    "array_dtypes": [str(array.dtype) for array in arrays],
                }
            )
        processing_ended_epoch = time.time()
        status = "complete"
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        ended_epoch = time.time()
        payload = {
            "schema_version": "flashvid-gpu-throughput-worker-v1",
            "status": status,
            "error": error,
            "started_at": started_iso,
            "ended_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "wall_seconds": ended_epoch - started_epoch,
            "processing_started_epoch": processing_started_epoch,
            "processing_ended_epoch": processing_ended_epoch,
            "processing_seconds": (
                None
                if processing_started_epoch is None or processing_ended_epoch is None
                else processing_ended_epoch - processing_started_epoch
            ),
            "model_load_seconds": model_load_seconds,
            "physical_gpu_index": args.physical_gpu,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_visible_gpu_count": torch.cuda.device_count(),
            "torch_gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch_max_memory_allocated_bytes": (
                torch.cuda.max_memory_allocated(0) if torch.cuda.is_available() else None
            ),
            "torch_max_memory_reserved_bytes": (
                torch.cuda.max_memory_reserved(0) if torch.cuda.is_available() else None
            ),
            "assignment": str(assignment_path),
            "video_ids": videos,
            "records": records,
            "created_files": created_files,
            "temporary_collaborator_duty_delegation": True,
            "responsibility_note": RESPONSIBILITY_NOTE,
        }
        atomic_json(output_path, payload)
    return 0


def summarize_scale(
    scale: int,
    repetition: int,
    worker_payloads: list[dict[str, Any]],
    wall_seconds: float,
    samples: list[dict[str, Any]],
    sampler_errors: list[str],
) -> dict[str, Any]:
    if any(payload["status"] != "complete" for payload in worker_payloads):
        raise RuntimeError("cannot summarize an incomplete worker")
    records = [record for payload in worker_payloads for record in payload["records"]]
    process_starts = [payload["processing_started_epoch"] for payload in worker_payloads]
    process_ends = [payload["processing_ended_epoch"] for payload in worker_payloads]
    processing_window = max(process_ends) - min(process_starts)
    if processing_window <= 0 or wall_seconds <= 0:
        raise RuntimeError("non-positive benchmark interval")

    gpu_summary = {}
    for physical_index in range(scale):
        current = [row for row in samples if row["physical_index"] == physical_index]
        utilization = [row["utilization_gpu_percent"] for row in current]
        memory = [row["memory_used_mib"] for row in current]
        gpu_summary[str(physical_index)] = {
            "sample_count": len(current),
            "utilization_mean_percent": statistics.fmean(utilization) if utilization else None,
            "utilization_p95_percent": percentile(utilization, 0.95),
            "utilization_max_percent": max(utilization) if utilization else None,
            "memory_used_max_mib": max(memory) if memory else None,
        }

    throughput = len(records) / processing_window
    return {
        "scale": scale,
        "repetition": repetition,
        "video_count": len(records),
        "orchestrator_wall_seconds": wall_seconds,
        "processing_window_seconds": processing_window,
        "end_to_end_videos_per_second": len(records) / wall_seconds,
        "processing_window_videos_per_second": throughput,
        "extrapolated_300_video_processing_minutes": 300.0 / throughput / 60.0,
        "model_load_seconds": [payload["model_load_seconds"] for payload in worker_payloads],
        "worker_processing_seconds": [payload["processing_seconds"] for payload in worker_payloads],
        "decode_seconds_total": sum(record["decode_seconds"] for record in records),
        "encode_seconds_total": sum(record["encode_seconds"] for record in records),
        "write_seconds_total": sum(record["write_seconds"] for record in records),
        "per_video_total_seconds": [
            record["decode_seconds"] + record["encode_seconds"] + record["write_seconds"]
            for record in records
        ],
        "worker_manifests": [payload["_manifest_path"] for payload in worker_payloads],
        "gpu_sampling_scope": "from immediately before worker launch through all worker exits",
        "gpu_summary": gpu_summary,
        "gpu_sampler_errors": sampler_errors,
    }


def ensure_selected_gpus_idle(inventory: list[dict[str, Any]], scale: int) -> None:
    selected_uuids = {row["uuid"] for row in inventory if row["physical_index"] < scale}
    active = [row for row in query_compute_processes() if row["gpu_uuid"] in selected_uuids]
    if active:
        raise RuntimeError(f"selected GPUs already have compute processes: {active}")


def orchestrate(args: argparse.Namespace) -> int:
    import adts_harness as harness

    print(f"responsibility_note={RESPONSIBILITY_NOTE}", flush=True)

    ids = load_video_ids()
    require_video_files(ids, VIDEO_DIR)
    cohort = choose_even_cohort(ids, args.cohort_size)
    scales = [int(value) for value in args.scale_order.split(",")]
    if sorted(set(scales)) != [1, 2, 4, 8]:
        raise ValueError("scale order must contain each of 1,2,4,8 at least once")
    if any(value not in {1, 2, 4, 8} for value in scales):
        raise ValueError("only scales 1,2,4,8 are allowed")

    environment_manifest_path = Path(args.environment_manifest).resolve()
    environment_manifest = json.loads(environment_manifest_path.read_text(encoding="utf-8"))
    if environment_manifest.get("gpu_count") != 8:
        raise RuntimeError("environment manifest does not attest 8 visible GPUs")
    if environment_manifest.get("throughput_benchmarked") is not False:
        raise RuntimeError("unexpected environment manifest throughput state")

    inventory = query_gpu_inventory()
    if len(inventory) != 8 or any(row["name"] != "NVIDIA H20-3e" for row in inventory):
        raise RuntimeError(f"expected 8 NVIDIA H20-3e GPUs, got {inventory}")
    model_source, vision_source = harness.resolve_cached_model_sources()

    run_stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    run_id = f"gpu-throughput_{os.uname().nodename}_{run_stamp}"
    run_dir = ROOT / "runs" / "gpu_benchmarks" / run_id
    scratch_root = ROOT / "runs" / "gpu_benchmark_scratch" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    scratch_root.mkdir(parents=True, exist_ok=False)

    video_inputs = []
    for video_id in cohort:
        path = VIDEO_DIR / f"{video_id}.mp4"
        video_inputs.append(
            {
                "video_id": video_id,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    script_inputs = {}
    for relative in (
        "scripts/gpu_throughput_benchmark.py",
        "scripts/stage2_dump_features.py",
        "scripts/adts_harness.py",
        "scripts/data_manifest.py",
        "data/short_subset.json",
    ):
        path = ROOT / relative
        script_inputs[relative] = sha256_file(path)

    top_payload: dict[str, Any] = {
        "schema_version": "flashvid-gpu-throughput-benchmark-v1",
        "run_id": run_id,
        "status": "running",
        "purpose": "Infrastructure-only strong-scaling benchmark; not a scientific Stage 1/2 run.",
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "hostname": os.uname().nodename,
        "qscode_display_name": environment_manifest["qscode_display_name"],
        "gpu_inventory": inventory,
        "environment_manifest": str(environment_manifest_path),
        "environment_manifest_sha256": sha256_file(environment_manifest_path),
        "model_identity": {
            "repository": harness.MODEL_PATH,
            "revision": harness.MODEL_REVISION,
            "snapshot": str(model_source),
            "config_sha256": sha256_file(model_source / "config.json"),
            "vision_repository": harness.VISION_TOWER_PATH,
            "vision_revision": harness.VISION_TOWER_REVISION,
            "vision_snapshot": str(vision_source),
            "vision_config_sha256": sha256_file(vision_source / "config.json"),
        },
        "script_and_input_hashes": script_inputs,
        "cohort_selection": "deterministic indices spread evenly over short_subset manifest order",
        "cohort_size": len(cohort),
        "video_inputs": video_inputs,
        "scale_order": scales,
        "cpu_threads_per_worker": args.cpu_threads_per_worker,
        "scratch_root": str(scratch_root),
        "scratch_outputs_retained": None,
        "results": [],
        "temporary_collaborator_duty_delegation": True,
        "responsibility_note": RESPONSIBILITY_NOTE,
    }
    atomic_json(run_dir / "run.partial.json", top_payload)

    try:
        occurrence: dict[int, int] = {}
        for sequence_index, scale in enumerate(scales, 1):
            occurrence[scale] = occurrence.get(scale, 0) + 1
            repetition = occurrence[scale]
            ensure_selected_gpus_idle(inventory, scale)
            scale_name = f"seq{sequence_index:02d}_scale{scale}_rep{repetition}"
            scale_dir = run_dir / scale_name
            scale_scratch = scratch_root / scale_name
            scale_dir.mkdir()
            scale_scratch.mkdir()
            assignments = partition_round_robin(cohort, scale)

            processes: list[tuple[subprocess.Popen[str], Any, Path, Path]] = []
            sampler = UtilizationSampler(set(range(scale)))
            sampler.start()
            scale_started = time.perf_counter()
            for worker_index, assigned_ids in enumerate(assignments):
                assignment_path = scale_dir / f"worker{worker_index:02d}.assignment.json"
                worker_manifest = scale_dir / f"worker{worker_index:02d}.json"
                worker_scratch = scale_scratch / f"worker{worker_index:02d}"
                atomic_json(
                    assignment_path,
                    {
                        "scale": scale,
                        "repetition": repetition,
                        "worker_index": worker_index,
                        "physical_gpu_index": worker_index,
                        "video_ids": assigned_ids,
                        "temporary_collaborator_duty_delegation": True,
                        "responsibility_note": RESPONSIBILITY_NOTE,
                    },
                )
                log_path = scale_dir / f"worker{worker_index:02d}.log"
                log_handle = log_path.open("w", encoding="utf-8")
                env = os.environ.copy()
                env.update(
                    {
                        "CUDA_VISIBLE_DEVICES": str(worker_index),
                        "HF_HOME": str(ROOT / "weights"),
                        "HF_HUB_OFFLINE": "1",
                        "TRANSFORMERS_OFFLINE": "1",
                        "TOKENIZERS_PARALLELISM": "false",
                        "OMP_NUM_THREADS": str(args.cpu_threads_per_worker),
                        "MKL_NUM_THREADS": str(args.cpu_threads_per_worker),
                        "OPENBLAS_NUM_THREADS": str(args.cpu_threads_per_worker),
                    }
                )
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--assignment",
                    str(assignment_path),
                    "--output",
                    str(worker_manifest),
                    "--scratch-dir",
                    str(worker_scratch),
                    "--physical-gpu",
                    str(worker_index),
                ]
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=env,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                processes.append((process, log_handle, worker_manifest, log_path))

            failures = []
            for process, log_handle, worker_manifest, log_path in processes:
                return_code = process.wait()
                log_handle.close()
                if return_code != 0:
                    failures.append(
                        {
                            "pid": process.pid,
                            "return_code": return_code,
                            "manifest": str(worker_manifest),
                            "log": str(log_path),
                        }
                    )
            scale_wall = time.perf_counter() - scale_started
            sampler.stop()
            atomic_json(
                scale_dir / "gpu_samples.json",
                {
                    "samples": sampler.samples,
                    "errors": sampler.errors,
                    "temporary_collaborator_duty_delegation": True,
                    "responsibility_note": RESPONSIBILITY_NOTE,
                },
            )
            if failures:
                raise RuntimeError(f"worker failures in {scale_name}: {failures}")

            worker_payloads = []
            created_files: list[Path] = []
            for _, _, worker_manifest, _ in processes:
                payload = json.loads(worker_manifest.read_text(encoding="utf-8"))
                payload["_manifest_path"] = str(worker_manifest)
                worker_payloads.append(payload)
                created_files.extend(Path(path) for path in payload["created_files"])
            if len(created_files) != len(cohort) or len(set(created_files)) != len(cohort):
                raise RuntimeError("scratch outputs are not one-to-one with benchmark cohort")
            for path in created_files:
                if scale_scratch not in path.resolve().parents:
                    raise RuntimeError(f"refusing to clean unexpected path: {path}")
                if not path.is_file() or path.stat().st_size <= 0:
                    raise RuntimeError(f"missing or empty scratch output: {path}")

            summary = summarize_scale(
                scale, repetition, worker_payloads, scale_wall, sampler.samples, sampler.errors
            )
            atomic_json(scale_dir / "summary.json", summary)
            top_payload["results"].append(summary)
            atomic_json(run_dir / "run.partial.json", top_payload)

            # Delete only the exact files listed by this run's worker manifests.
            for path in created_files:
                path.unlink()
            for worker_dir in sorted(scale_scratch.iterdir()):
                worker_dir.rmdir()
            scale_scratch.rmdir()

        scratch_root.rmdir()
        top_payload["status"] = "complete"
        top_payload["ended_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        top_payload["scratch_outputs_retained"] = False
        final_path = run_dir / "run.json"
        atomic_json(final_path, top_payload)
        (run_dir / "run.partial.json").unlink()
        print(final_path, flush=True)
        return 0
    except Exception as exc:
        top_payload["status"] = "failed"
        top_payload["ended_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        top_payload["error"] = {"type": type(exc).__name__, "message": str(exc)}
        top_payload["scratch_outputs_retained"] = scratch_root.exists()
        atomic_json(run_dir / "run.failed.json", top_payload)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--assignment")
    parser.add_argument("--output")
    parser.add_argument("--scratch-dir")
    parser.add_argument("--physical-gpu", type=int)
    parser.add_argument("--cohort-size", type=int, default=16)
    parser.add_argument(
        "--scale-order",
        default="1,4,8,2,8,2,1,4",
        help="Counterbalanced sequence; each of 1,2,4,8 must appear.",
    )
    parser.add_argument("--cpu-threads-per-worker", type=int, default=8)
    parser.add_argument("--environment-manifest", default=str(DEFAULT_ENV_MANIFEST))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.worker:
        required = (args.assignment, args.output, args.scratch_dir, args.physical_gpu)
        if any(value is None for value in required):
            raise ValueError("worker mode requires assignment/output/scratch-dir/physical-gpu")
        return worker_main(args)
    return orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
