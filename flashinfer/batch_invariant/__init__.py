# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Batch-invariant mode: deterministic, batch-invariant kernels behind ``batch_invariant=True``.

The package holds the engine of the batch-invariant kernels (the canonical reduction plan, the floating-point interface
``fp_interface`` in its symbolic and device backends, the term language and its fingerprints) and the Lean project that
proves the engine's reduction orders; each kernel class lives next to its FlashInfer operator (``flashinfer/gemm/batch_invariant``). ``README.md``
next to this file is the contract: the guarantee (R1-R4), the flag semantics, the tests and how to check the proofs.
"""

from __future__ import annotations

from typing import Callable, Dict, NamedTuple

from .support import (
    BatchInvariantUnsupportedError,
    check_device,
    is_supported_device,
)


class ClassInfo(NamedTuple):
    """A consistency class behind a FlashInfer API: its name, the fixed dimensions its fingerprint takes, and the
    function computing the fingerprint from them."""

    name: str
    dims: tuple
    fingerprint: Callable[..., str]


def _mm_bf16_fingerprint(N: int, K: int) -> str:
    from ..gemm.batch_invariant.schedule import chunks_for
    from ..gemm.batch_invariant.spec import spec_fingerprint

    return spec_fingerprint(K, "bf16", chunks_for(N, K))


def _attention_fingerprint(head_dim: int) -> str:
    from ..attention.batch_invariant.spec import spec_fingerprint

    return spec_fingerprint(head_dim)


#: The FlashInfer APIs with a batch-invariant class, by API name.
CLASSES: Dict[str, ClassInfo] = {
    "mm_bf16": ClassInfo("gemm.bf16", ("N", "K"), _mm_bf16_fingerprint),
    # the three attention wrappers compute one class per head dim: a token's output depends on its KV position, its head and the
    # window alone, so prefill and decode (and the paged and ragged KV layouts) are one class
    "BatchDecodeWithPagedKVCacheWrapper": ClassInfo(
        "attention.bf16", ("head_dim",), _attention_fingerprint
    ),
    "BatchPrefillWithPagedKVCacheWrapper": ClassInfo(
        "attention.bf16", ("head_dim",), _attention_fingerprint
    ),
    "BatchPrefillWithRaggedKVCacheWrapper": ClassInfo(
        "attention.bf16", ("head_dim",), _attention_fingerprint
    ),
}


def fingerprint(op: str, **dims) -> str:
    """The receipt of a batch-invariant output: the fingerprint of the consistency class the FlashInfer API ``op``
    computes under ``batch_invariant=True`` for the fixed dimensions ``dims``.

    The guarantee is scoped to ``(fingerprint, FlashInfer version, architecture)``: two outputs with the same
    fingerprint, the same inputs and the same environment are the same bits, whatever the batch, the configuration
    the dispatcher picked or the API path. A fingerprint change is a release-note item.

    ``mm_bf16`` takes the weight shape ``N, K``: the K steps of the fold and the chunk count (a function of the
    weight shape alone) are the specification; M, the configuration and the weight's layout do not enter it.

    The attention wrappers take ``head_dim``: the tile and chunk grid, the online-softmax step and the chunk merge are
    functions of the head dim alone; the batch, the sequence lengths, the page size, the KV layout and the configuration
    do not enter it, and ``BatchPrefillWithPagedKVCacheWrapper``, ``BatchDecodeWithPagedKVCacheWrapper`` and
    ``BatchPrefillWithRaggedKVCacheWrapper`` share the fingerprint (their outputs for one token are the same bits).

    >>> fingerprint("mm_bf16", N=4096, K=4096)  # doctest: +SKIP
    'gemm-bf16-kfold-...'
    """
    info = CLASSES.get(op)
    if info is None:
        raise ValueError(
            f"no batch-invariant class behind {op!r}; the APIs with one are {sorted(CLASSES)}"
        )
    unknown = sorted(set(dims) - set(info.dims))
    if unknown:
        raise TypeError(
            f"fingerprint({op!r}) takes the fixed dimensions {list(info.dims)}, not {unknown}"
        )
    missing = [d for d in info.dims if d not in dims]
    if missing:
        raise TypeError(f"fingerprint({op!r}) needs the fixed dimensions {missing}")
    return info.fingerprint(**dims)


def class_name(op: str) -> str:
    """The consistency class name behind the API ``op`` (e.g. ``"gemm.bf16"``)."""
    info = CLASSES.get(op)
    if info is None:
        raise ValueError(
            f"no batch-invariant class behind {op!r}; the APIs with one are {sorted(CLASSES)}"
        )
    return info.name


__all__ = [
    "CLASSES",
    "ClassInfo",
    "BatchInvariantUnsupportedError",
    "check_device",
    "class_name",
    "fingerprint",
    "is_supported_device",
]
