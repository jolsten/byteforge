"""Tests for MilStd1750A encoding.

Reference vectors from type-convert/tests/test_1750a32.py and test_1750a48.py.
"""

import numpy as np
import pytest

from byteforge import MilStd1750A

# Reference decode vectors from type-convert
DECODE_VECTORS_32: list[tuple[int, float]] = [
    (0x7FFFFF7F, 0.9999998 * 2**127),
    (0x4000007F, 0.5 * 2**127),
    (0x50000004, 0.625 * 2**4),
    (0x40000001, 0.5 * 2**1),
    (0x40000000, 0.5 * 2**0),
    (0x400000FF, 0.5 * 2**-1),
    (0x40000080, 0.5 * 2**-128),
    (0x00000000, 0.0 * 2**0),
    (0x80000000, -1.0 * 2**0),
    (0xBFFFFF80, -0.5000001 * 2**-128),
    (0x9FFFFF04, -0.7500001 * 2**4),
]

DECODE_VECTORS_48: list[tuple[int, float]] = [
    (0x4000007F0000, 0.5 * 2**127),
    (0x400000000000, 0.5 * 2**0),
    (0x400000FF0000, 0.5 * 2**-1),
    (0x400000800000, 0.5 * 2**-128),
    (0x8000007F0000, -1.0 * 2**127),
    (0x800000000000, -1.0 * 2**0),
    (0x800000FF0000, -1.0 * 2**-1),
    (0x800000800000, -1.0 * 2**-128),
    (0x000000000000, 0.0 * 2**0),
    (0xA00000FF0000, -0.75 * 2**-1),
]


@pytest.mark.parametrize("dn, expected", DECODE_VECTORS_32)
def test_decode_32bit_reference(dn: int, expected: float):
    enc = MilStd1750A(32)
    result = enc.decode(np.array([dn], dtype=np.uint64))[0]
    assert result == pytest.approx(expected)


@pytest.mark.parametrize("dn, expected", DECODE_VECTORS_48)
def test_decode_48bit_reference(dn: int, expected: float):
    enc = MilStd1750A(48)
    result = enc.decode(np.array([dn], dtype=np.uint64))[0]
    assert result == pytest.approx(expected)


class TestMilStd1750AEncode:
    def test_32bit_encode_zero(self):
        result = MilStd1750A(32).encode(np.array([0.0]))
        assert result[0] == 0x00000000

    def test_32bit_encode_half(self):
        result = MilStd1750A(32).encode(np.array([0.5]))
        assert result[0] == 0x40000000

    def test_32bit_encode_one(self):
        result = MilStd1750A(32).encode(np.array([1.0]))
        assert result[0] == 0x40000001

    def test_32bit_encode_neg_half(self):
        result = MilStd1750A(32).encode(np.array([-0.5]))
        assert result[0] == 0xC0000000

    def test_encode_dtype(self):
        assert MilStd1750A(32).encode(np.array([1.0])).dtype == np.uint32
        assert MilStd1750A(48).encode(np.array([1.0])).dtype == np.uint64


class TestMilStd1750ADecode:
    def test_32bit_decode_zero(self):
        result = MilStd1750A(32).decode(np.array([0], dtype=np.uint64))
        assert result[0] == pytest.approx(0.0)

    def test_32bit_decode_half(self):
        result = MilStd1750A(32).decode(np.array([0x40000000], dtype=np.uint64))
        assert result[0] == pytest.approx(0.5)

    def test_32bit_decode_one(self):
        result = MilStd1750A(32).decode(np.array([0x40000001], dtype=np.uint64))
        assert result[0] == pytest.approx(1.0)


class TestMilStd1750ARoundtrip:
    def test_32bit_roundtrip(self):
        enc = MilStd1750A(32)
        for v in (0.5, 1.0, -0.5, -1.0, 360.0, 0.001):
            result = enc.decode(enc.encode(np.array([v])))[0]
            assert result == pytest.approx(v, rel=1e-6)

    def test_48bit_roundtrip(self):
        enc = MilStd1750A(48)
        for v in (0.5, 1.0, -0.5, -1.0, 360.0, 0.001):
            result = enc.decode(enc.encode(np.array([v])))[0]
            assert result == pytest.approx(v, rel=1e-10)

    def test_48bit_higher_precision_than_32bit(self):
        v = 1.0 / 3.0
        err_32 = abs(MilStd1750A(32).decode(MilStd1750A(32).encode(np.array([v])))[0] - v)
        err_48 = abs(MilStd1750A(48).decode(MilStd1750A(48).encode(np.array([v])))[0] - v)
        assert err_48 < err_32

    def test_decode_array_batch(self):
        enc = MilStd1750A(32)
        vals = [0.5, 1.0, -0.5, 360.0]
        dns = enc.encode(np.array(vals))
        decoded = enc.decode(dns)
        for expected, actual in zip(vals, decoded):
            assert actual == pytest.approx(expected, rel=1e-6)


class TestMilStd1750AMisc:
    def test_invalid_bit_width(self):
        with pytest.raises(ValueError, match="32 or 48"):
            MilStd1750A(64)

    def test_repr(self):
        r = repr(MilStd1750A(32))
        assert "MilStd1750A" in r
        assert "32" in r
