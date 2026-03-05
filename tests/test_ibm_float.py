"""Tests for IBMFloat encoding.

Reference vectors from type-convert/tests/test_ibm32.py and test_ibm64.py.
Reference: https://en.wikipedia.org/wiki/IBM_hexadecimal_floating-point
"""

import numpy as np
import pytest

from byteforge import IBMFloat

# ---------------------------------------------------------------------------
# IBM32 decode vectors
# ---------------------------------------------------------------------------
DECODE_VECTORS_32 = [
    (0x00000000, 0.0),
    (0x4019999A, 0.1),
    (0xC019999A, -0.1),
    (0x00100000, 5.397605e-79),
    (0x80100000, -5.397605e-79),
]

# ---------------------------------------------------------------------------
# IBM64 decode vectors
# ---------------------------------------------------------------------------
DECODE_VECTORS_64 = [
    (0x0000000000000000, 0.0),
    (0x401999999999999A, 0.1),
    (0xC01999999999999A, -0.1),
    (0x0010000000000000, 5.397605e-79),
    (0x8010000000000000, -5.397605e-79),
]


class TestIBMFloat32Decode:
    @pytest.mark.parametrize("dn, expected", DECODE_VECTORS_32)
    def test_decode(self, dn, expected):
        enc = IBMFloat(32)
        result = enc.decode(np.array([dn]))[0]
        assert result == pytest.approx(expected)


class TestIBMFloat64Decode:
    @pytest.mark.parametrize("dn, expected", DECODE_VECTORS_64)
    def test_decode(self, dn, expected):
        enc = IBMFloat(64)
        result = enc.decode(np.array([dn]))[0]
        assert result == pytest.approx(expected)


class TestIBMFloatEncode:
    @pytest.mark.parametrize("bit_width", [32, 64])
    def test_encode_zero(self, bit_width):
        enc = IBMFloat(bit_width)
        dn = enc.encode(np.array([0.0]))[0]
        assert int(dn) == 0

    @pytest.mark.parametrize("bit_width", [32, 64])
    @pytest.mark.parametrize("value", [0.1, -0.1, 1.0, -1.0, 0.5, 100.0, -100.0])
    def test_encode_roundtrip(self, bit_width, value):
        enc = IBMFloat(bit_width)
        dn = enc.encode(np.array([value]))
        result = enc.decode(dn)[0]
        rel = 1e-5 if bit_width == 32 else 1e-13
        assert result == pytest.approx(value, rel=rel)

    def test_encode_batch(self):
        enc = IBMFloat(32)
        values = np.array([0.0, 0.1, -0.1, 1.0, -1.0])
        dns = enc.encode(values)
        decoded = enc.decode(dns)
        np.testing.assert_allclose(decoded, values, rtol=1e-5, atol=1e-78)


class TestIBMFloatMisc:
    def test_invalid_bit_width(self):
        with pytest.raises(ValueError, match="must be 32 or 64"):
            IBMFloat(48)

    def test_repr_32(self):
        enc = IBMFloat(32)
        assert "IBMFloat" in repr(enc)
        assert "base=16" in repr(enc)

    def test_dtype_32(self):
        enc = IBMFloat(32)
        dn = enc.encode(np.array([1.0]))
        assert dn.dtype == np.uint32

    def test_dtype_64(self):
        enc = IBMFloat(64)
        dn = enc.encode(np.array([1.0]))
        assert dn.dtype == np.uint64

    @pytest.mark.parametrize("bit_width", [32, 64])
    def test_value_range(self, bit_width):
        enc = IBMFloat(bit_width)
        lo, hi = enc.value_range
        assert lo < 0
        assert hi > 0
        assert lo == -hi
