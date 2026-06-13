"""Tests for the hybrid codec (HYBRID_CODEC_TASK.md).

The hybrid codec tries multiple methods and keeps the smallest, at the cost
of exactly one header byte. The "never worse than best classical/neural/raw
+ 1 byte" property is the entire point and is asserted literally.

# SLOW: ok — the neural candidate runs `PretrainedRNNPredictor` over each
20KB corpus (~8s), per CLAUDE.md rule 5.
"""

from __future__ import annotations

import bz2
import lzma
import os
from pathlib import Path

import pytest

from teleport.coder import compress as coder_compress
from teleport.hybrid import (
    METHOD_BZ2,
    METHOD_LZMA,
    METHOD_NEURAL,
    METHOD_STORED,
    compress_hybrid,
    decompress_hybrid,
)
from teleport.predictors.rnn import DEFAULT_WEIGHTS, PretrainedRNNPredictor

ROOT = Path(__file__).resolve().parent.parent
INDOMAIN = (ROOT / "data" / "enwik6_indomain_20k").read_bytes()
CODE_OOD = (ROOT / "data" / "code_ood_20k").read_bytes()

pytestmark_pretrained = pytest.mark.skipif(
    not DEFAULT_WEIGHTS.exists(), reason="models/rnn_shared.pt not built (run scripts/pretrain_rnn.py)"
)


def _min_candidate_len(data: bytes) -> int:
    candidates = [len(data), len(lzma.compress(data, preset=9)), len(bz2.compress(data, compresslevel=9))]
    if DEFAULT_WEIGHTS.exists():
        candidates.append(len(coder_compress(data, PretrainedRNNPredictor)))
    return min(candidates)


@pytest.mark.parametrize("data", [INDOMAIN, CODE_OOD], ids=["enwik6_indomain_20k", "code_ood_20k"])
def test_roundtrip_corpora(data: bytes) -> None:
    compressed = compress_hybrid(data)
    assert decompress_hybrid(compressed) == data


@pytest.mark.parametrize(
    "data",
    [b"", b"x", os.urandom(1024), b"\x00" * 1024],
    ids=["empty", "one_byte", "urandom_1k", "zeros_1k"],
)
def test_roundtrip_edge_cases(data: bytes) -> None:
    compressed = compress_hybrid(data)
    assert decompress_hybrid(compressed) == data


@pytest.mark.parametrize("data", [INDOMAIN, CODE_OOD], ids=["enwik6_indomain_20k", "code_ood_20k"])
def test_never_worse_than_best_candidate_plus_one_byte(data: bytes) -> None:
    compressed = compress_hybrid(data)
    assert len(compressed) <= _min_candidate_len(data) + 1


@pytestmark_pretrained
def test_method_selection_indomain_picks_neural() -> None:
    compressed = compress_hybrid(INDOMAIN)
    assert compressed[0] == METHOD_NEURAL


@pytestmark_pretrained
def test_method_selection_code_ood_picks_classical() -> None:
    compressed = compress_hybrid(CODE_OOD)
    assert compressed[0] in (METHOD_LZMA, METHOD_BZ2)


def test_method_selection_urandom_picks_stored() -> None:
    data = os.urandom(1024)
    compressed = compress_hybrid(data)
    assert compressed[0] == METHOD_STORED


def test_unknown_header_id_raises() -> None:
    with pytest.raises(ValueError):
        decompress_hybrid(b"\xff" + b"whatever")


@pytestmark_pretrained
@pytest.mark.parametrize("data", [INDOMAIN, CODE_OOD], ids=["enwik6_indomain_20k", "code_ood_20k"])
def test_probe_does_not_change_output(data: bytes) -> None:
    """The probe is an encoder-side speed heuristic; it must never change
    which method is selected or the resulting bytes."""
    assert compress_hybrid(data, probe=True) == compress_hybrid(data, probe=False)


@pytestmark_pretrained
def test_probe_skips_neural_on_code_ood() -> None:
    """code_ood's cross-entropy under the in-domain-pretrained predictor is
    bad enough (Phase 2.2: 5.21 bpc) that the probe should skip the full
    neural encode, leaving classical as the only non-stored candidates."""
    from teleport.hybrid import _neural_probe_bpc, _PROBE_THRESHOLD_BPC
    from teleport.predictors.rnn import DEFAULT_WEIGHTS

    assert _neural_probe_bpc(CODE_OOD, DEFAULT_WEIGHTS) > _PROBE_THRESHOLD_BPC
