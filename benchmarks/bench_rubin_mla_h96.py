# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Compare H96 Rubin mixed-CGA and monolithic using identical inputs.

Default invocation validates and times H96/B64/Q8/K8K (MTP7).
Use --q 2..8 for MTP1..7.
Each invocation is a single case so compilation and hangs are isolated.
Positive --splits uses the same budget; -1 uses each backend's normal heuristic.
"""

import argparse
import csv
import functools
import math
import statistics
from pathlib import Path

import torch
from cutlass import Float32, Int32

from flashinfer.cute_dsl.attention.monolithic import mla_decode as monolithic
from flashinfer.cute_dsl.attention.rubin_mtp import mla_decode as mixed
from flashinfer.cute_dsl.utils import cute_dsl_compile_arch


def reference(query, kv, table, lengths, scale):
    """FP32 eager GPU attention with bottom-right causality and empty-row identity.

    Rubin's torch.compile support varies by toolchain; use cuBLAS through matmul.
    This reference gates correctness, never a compiled/fused reference.
    """
    outputs, lses = [], []
    q_len = query.shape[1]
    for b, length in enumerate(lengths):
        q = query[b].float()
        cache = kv[table[b].long()].reshape(-1, 576)[:length].float()
        if length == 0:
            outputs.append(torch.zeros_like(q[..., :512]))
            lses.append(torch.full_like(q[..., 0], -math.inf))
            continue
        scores = torch.matmul(q, cache.T) * scale
        positions = torch.arange(q_len, device="cuda") + length - q_len
        mask = torch.arange(length, device="cuda")[None, :] > positions[:, None]
        scores.masked_fill_(mask[:, None, :], -math.inf)
        probs = scores.softmax(-1)
        probs = torch.where(positions[:, None, None] >= 0, probs, 0.0)
        outputs.append(torch.matmul(probs, cache[:, :512]))
        lses.append(scores.logsumexp(-1))
    return torch.stack(outputs), torch.stack(lses)


def graph_times(fn, flush, repeats=5):
    """Return warm/cold CUDA Graph times in us; flush uses reads outside timing."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(25):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(repeats):
            fn()
    warm, cold = [], []
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    for _ in range(5):
        graph.replay()
    for _ in range(5):
        torch.cuda.synchronize()
        start.record()
        for _ in range(100):
            graph.replay()
        end.record()
        end.synchronize()
        warm.append(start.elapsed_time(end) * 1000 / (100 * repeats))
    single = torch.cuda.CUDAGraph()
    with torch.cuda.graph(single):
        fn()
    for _ in range(20):
        flush.sum()
        torch.cuda.synchronize()
        start.record()
        single.replay()
        end.record()
        end.synchronize()
        cold.append(start.elapsed_time(end) * 1000)
    return statistics.median(warm), statistics.median(cold)


def main():
    """Run a single shape with independently checked, identical backend inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--q", type=int, choices=range(2, 9), default=8)
    parser.add_argument("--heads", type=int, choices=(96, 128), default=96)
    parser.add_argument("--kv", type=int, default=8192)
    parser.add_argument("--page", type=int, choices=(64, 128), default=128)
    parser.add_argument("--splits", type=int, default=1)
    parser.add_argument(
        "--branch", choices=("auto", "preferred", "fallback"), default="auto"
    )
    parser.add_argument("--ragged-k", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--mixed-only", action="store_true")
    parser.add_argument("--order", type=int, default=0)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    if args.mixed_only and not args.check_only:
        parser.error("--mixed-only requires --check-only")
    assert torch.cuda.get_device_capability() == (10, 7)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(20261008)
    B, Q, H, K, P = args.batch, args.q, args.heads, args.kv, args.page
    dtype = torch.float8_e4m3fn
    pages = max(1, math.ceil(K / P))
    query = (torch.randn(B, Q, H, 576, device="cuda") * 0.5).to(dtype)
    kv = (torch.randn(B * pages, P, 576, device="cuda") * 0.5).to(dtype)
    table = torch.randperm(B * pages, device="cuda", dtype=torch.int32).reshape(
        B, pages
    )
    lengths = [K] * B
    if args.ragged_k:
        candidates = (0, min(K, 1), min(K, 127), min(K, 128), min(K, 129), K)
        lengths = [candidates[b % len(candidates)] for b in range(B)]
    seq_lens = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    scale = 576**-0.5
    expected, expected_lse = reference(query, kv, table, lengths, scale)
    arch = cute_dsl_compile_arch(10, 7)
    variants = {}
    checked_outputs = {}
    selected_splits = {}
    for name, module in (("mixed", mixed), ("monolithic", monolithic)):
        if args.mixed_only and name == "monolithic":
            continue
        print(
            f"Preparing {name} H{H} Q{Q} B{B} K{K} split{args.splits} "
            f"branch={args.branch}",
            flush=True,
        )
        split_kv, size = module._get_split_kv_and_workspace_size(
            B,
            Q,
            H,
            512,
            module.get_num_sm(query.device),
            max_seq_len=pages * P,
            num_kv_splits=args.splits,
        )
        selected_splits[name] = split_kv
        print(f"[config] {name} selected_splits={split_kv}", flush=True)
        workspace = torch.empty(max(32, size), dtype=torch.uint8, device="cuda")
        compile_args = dict(
            arch=arch,
            torch_dtype=dtype,
            torch_out_dtype=dtype,
            page_size=P,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            num_heads=H,
            seq_len_q=Q,
            is_persistent=False,
            is_var_seq=True,
            is_var_q=False,
            is_var_split_kv=False,
            is_workspace_size_zero=size == 0,
        )
        if name == "mixed":
            compile_args["force_branch"] = args.branch
        else:
            compile_args["reducer_d_tiles"] = monolithic._get_reducer_d_tiles(
                B, Q, H, monolithic.get_num_sm(query.device), split_kv
            )
            compile_args["reducer_max_splits"] = monolithic._get_reducer_max_splits(
                split_kv
            )
        launch = module._get_compiled_mla_kernel(**compile_args)
        # Adjacent guards detect tail stores beyond actual output/LSE capacity.
        out_storage = torch.empty(B * Q * H * 512 + 64, dtype=dtype, device="cuda")
        out_storage.view(torch.uint8).fill_(90)
        out = out_storage[:-64].view(B, Q, H, 512)
        lse_storage = torch.full((B * Q * H + 64,), 12345.0, device="cuda")
        lse = lse_storage[:-64].view(B, Q, H)
        run = functools.partial(
            launch,
            query[..., :512],
            query[..., 512:],
            kv[..., :512],
            kv[..., 512:],
            table,
            out,
            lse,
            workspace if size else None,
            Int32(split_kv),
            seq_lens,
            None,
            None,
            Int32(0),
            None,
            Float32(scale),
            Float32(1),
            Float32(1 / math.log2(math.e)),
        )
        run()
        torch.cuda.synchronize()
        torch.testing.assert_close(out.float(), expected, rtol=0.12, atol=0.03)
        torch.testing.assert_close(lse, expected_lse, rtol=0.01, atol=0.01)
        assert bool((out_storage[-64:].view(torch.uint8) == 90).all())
        assert bool((lse_storage[-64:] == 12345.0).all())
        finite = torch.isfinite(expected_lse)
        assert bool(torch.isfinite(out.float()).all())
        o_error = (out.float() - expected).abs().max().item()
        lse_error = (
            (lse[finite] - expected_lse[finite]).abs().max().item()
            if finite.any()
            else 0
        )
        print(
            f"[correctness] PASS {name} all rows/heads; O max_abs={o_error:.6g}, "
            f"LSE max_abs={lse_error:.6g}; output guards intact",
            flush=True,
        )
        variants[name] = run
        checked_outputs[name] = (out, lse, out_storage, lse_storage)
    if args.check_only:
        return
    flush = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda")
    order = ("mixed", "monolithic") if args.order % 2 == 0 else ("monolithic", "mixed")
    times = {name: graph_times(variants[name], flush) for name in order}
    for name, (out, lse, out_storage, lse_storage) in checked_outputs.items():
        torch.testing.assert_close(out.float(), expected, rtol=0.12, atol=0.03)
        torch.testing.assert_close(lse, expected_lse, rtol=0.01, atol=0.01)
        assert bool((out_storage[-64:].view(torch.uint8) == 90).all())
        assert bool((lse_storage[-64:] == 12345.0).all())
        print(f"[correctness] PASS {name} after repeated graph replay", flush=True)
    for protocol, idx in (("warm", 0), ("cold", 1)):
        mixed_us, mono_us = times["mixed"][idx], times["monolithic"][idx]
        print(f"[perf][{protocol}] kernel: {mixed_us / 1000:.6f} ms")
        print(f"[perf][{protocol}] reference: {mono_us / 1000:.6f} ms")
        print(f"[perf][{protocol}] speedup vs reference: {mono_us / mixed_us:.3f}x")
    print("[perf] primary=warm")
    print(
        "Performance reference is monolithic; correctness reference is eager FP32 PyTorch."
    )
    if args.csv:
        row = dict(
            batch=B,
            q=Q,
            heads=H,
            kv=K,
            page=P,
            splits=args.splits,
            mixed_splits=selected_splits["mixed"],
            mono_splits=selected_splits["monolithic"],
            branch=args.branch,
            order=args.order,
            mixed_warm_us=times["mixed"][0],
            mono_warm_us=times["monolithic"][0],
            mixed_cold_us=times["mixed"][1],
            mono_cold_us=times["monolithic"][1],
        )
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        exists = args.csv.exists()
        with args.csv.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)


if __name__ == "__main__":
    main()
