"""Byte-for-byte compatibility of the CPU-only logical KV prefix proof."""

import hashlib
import random

import numpy as np
import pytest

from sglang.srt.disaggregation.agentic_native_memory_adapter import (
    common_page_prefix_tokens,
    token_page_chain_hashes,
)


def reference(tokens, page_size):
    digest = hashlib.sha256()
    result = []
    for offset in range(0, len(tokens), page_size):
        for token in tokens[offset:offset + page_size]:
            digest.update(int(token).to_bytes(8, "little", signed=True))
        result.append(digest.hexdigest())
    return tuple(result)


@pytest.mark.parametrize("page_size", [1, 3, 64, 256])
@pytest.mark.parametrize("pages", [0, 1, 7, 513])
def test_page_hashes_match_original_every_boundary(page_size, pages):
    rng = random.Random(2026)
    tokens = [rng.randrange(-(1 << 63), 1 << 63) for _ in range(page_size * pages)]
    original = list(tokens)
    assert token_page_chain_hashes(tokens, page_size) == reference(tokens, page_size)
    assert tokens == original


def test_page_hashes_preserve_signed_little_endian_and_int_conversion():
    class Token:
        def __int__(self):
            return 123

        def __index__(self):
            return 456  # Packing must not bypass the old int() conversion.

    tokens = [-(1 << 63), (1 << 63) - 1, -1, 0, True, "25", 7.9, Token()]
    assert token_page_chain_hashes(tokens, 4) == reference(tokens, 4)
    values = np.arange(128, dtype=np.int64)
    assert token_page_chain_hashes(values, 64) == reference(values, 64)


@pytest.mark.parametrize("token", [-(1 << 63) - 1, 1 << 63])
def test_page_hashes_reject_overflow(token):
    with pytest.raises(OverflowError):
        token_page_chain_hashes([token], 1)


@pytest.mark.parametrize("tokens,page_size", [([1], 0), ([1], -1), ([1], 2)])
def test_page_hashes_reject_invalid_page_boundaries(tokens, page_size):
    with pytest.raises(ValueError):
        token_page_chain_hashes(tokens, page_size)


def test_page_hashes_reject_non_integer_ids():
    with pytest.raises(ValueError):
        token_page_chain_hashes(["bad"], 1)
    with pytest.raises(TypeError):
        token_page_chain_hashes([object()], 1)


def test_prefix_proof_remains_cumulative_and_detects_old_page_mutation():
    tokens = list(range(256))
    source = token_page_chain_hashes(tokens, 64)
    assert common_page_prefix_tokens(source, tokens + [9] * 31, 64) == 256
    tokens[70] += 1
    changed = token_page_chain_hashes(tokens, 64)
    assert source[0] == changed[0]
    assert all(a != b for a, b in zip(source[1:], changed[1:]))
    assert common_page_prefix_tokens(source, tokens, 64) == 64
