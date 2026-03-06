"""Benchmarks for all encoding types using pytest-benchmark.

Run with: uv run pytest tests/test_benchmarks.py --benchmark-enable -v
"""

import numpy as np
import pytest

from byteforge import (
    BCD,
    IEEE754,
    Boolean,
    DECFloat,
    DECFloatG,
    GrayCode,
    IBMFloat,
    LinearScaled,
    MilStd1750A,
    OffsetBinary,
    OnesComplement,
    TIFloat,
    TwosComplement,
    Unsigned,
)

SIZES = [100, 10_000, 1_000_000]
SIZE_IDS = ["100", "10k", "1M"]


def _make_unsigned_data(enc, size):
    lo, hi = enc.value_range
    return np.random.randint(int(lo), int(hi) + 1, size=size, dtype=np.int64)


def _make_signed_data(enc, size):
    lo, hi = enc.value_range
    return np.random.randint(int(lo), int(hi) + 1, size=size, dtype=np.int64)


def _make_float_data(enc, size):
    lo, hi = enc.value_range
    lo = max(lo, -1e30)
    hi = min(hi, 1e30)
    return np.random.uniform(lo, hi, size=size)


ENCODINGS = {
    "Unsigned(16)": (Unsigned(16), _make_unsigned_data),
    "TwosComplement(16)": (TwosComplement(16), _make_signed_data),
    "OnesComplement(16)": (OnesComplement(16), _make_signed_data),
    "OffsetBinary(16)": (OffsetBinary(16), _make_signed_data),
    "GrayCode(16)": (GrayCode(16), _make_unsigned_data),
    "BCD(8)": (BCD(8), lambda enc, size: np.random.randint(0, 100, size=size)),
    "Boolean": (Boolean(), lambda enc, size: np.random.randint(0, 2, size=size)),
    "IEEE754(32)": (IEEE754(32), lambda enc, size: np.random.uniform(-1e6, 1e6, size=size)),
    "IEEE754(64)": (IEEE754(64), lambda enc, size: np.random.uniform(-1e6, 1e6, size=size)),
    "LinearScaled(16)": (
        LinearScaled(16, scale_factor=0.01, offset=-100.0),
        _make_float_data,
    ),
    "MilStd1750A(32)": (MilStd1750A(32), _make_float_data),
    "MilStd1750A(48)": (MilStd1750A(48), _make_float_data),
    "TIFloat(32)": (TIFloat(32), _make_float_data),
    "TIFloat(40)": (TIFloat(40), _make_float_data),
    "IBMFloat(32)": (IBMFloat(32), _make_float_data),
    "IBMFloat(64)": (IBMFloat(64), _make_float_data),
    "DECFloat(32)": (DECFloat(32), _make_float_data),
    "DECFloatG(64)": (DECFloatG(64), _make_float_data),
}


@pytest.mark.benchmark(group="encode")
@pytest.mark.parametrize("size", SIZES, ids=SIZE_IDS)
@pytest.mark.parametrize("name", ENCODINGS.keys())
def test_encode_benchmark(benchmark, name, size):
    enc, data_fn = ENCODINGS[name]
    data = data_fn(enc, size)
    benchmark(enc.encode, data)


@pytest.mark.benchmark(group="decode")
@pytest.mark.parametrize("size", SIZES, ids=SIZE_IDS)
@pytest.mark.parametrize("name", ENCODINGS.keys())
def test_decode_benchmark(benchmark, name, size):
    enc, data_fn = ENCODINGS[name]
    data = data_fn(enc, size)
    encoded = enc.encode(data)
    benchmark(enc.decode, encoded)
