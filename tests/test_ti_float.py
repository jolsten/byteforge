"""Tests for TIFloat encoding.

Reference vectors from type-convert/tests/test_ti32.py and test_ti40.py.
Reference: https://www.ti.com/lit/an/spra400/spra400.pdf
"""

import numpy as np
import pytest

from byteforge import TIFloat


def _pack32(e, s, m):
    return int(np.uint32((e << 24) + (s << 23) + m))


def _pack40(e, s, m):
    return int(np.uint64((e << 32) + (s << 31) + m))


# ---------------------------------------------------------------------------
# TI32 decode vectors (from type-convert TEST_FROM_COMPONENTS)
# ---------------------------------------------------------------------------
DECODE_VECTORS_32 = [
    # (dn, expected_value)
    # --- Positive, large exponent ---
    (_pack32(0x7F, 0, 0b11111111111111111111111), (2 - 2**-23) * 2**127),
    (_pack32(0x7F, 0, 0b00000000000000000000000), 2**127),
    # --- Positive, e=0 ---
    (_pack32(0x00, 0, 0b00000000000000000000000), 1.0),
    # --- Positive, e=-1 (0xFF as two's comp) ---
    (_pack32(0xFF, 0, 0b11111111111111111111111), 1 - 2**-24),
    (_pack32(0xFF, 0, 0b00000000000000000000000), 0.5),
    # --- Positive, small exponent ---
    (_pack32(0x81, 0, 0b00000000000000000000000), 2**-127),
    (_pack32(0x81, 0, 0b00000000000000000000001), (1 + 2**-23) * 2**-127),
    (_pack32(0x82, 0, 0b00000000000000000000000), 2**-126),
    # --- Zero (e = -128, i.e. 0x80) ---
    (_pack32(0x80, 0, 0b00000000000000000000000), 0.0),
    (_pack32(0x80, 0, 0b11111111111111111111111), 0.0),
    (_pack32(0x80, 1, 0b00000000000000000000000), 0.0),
    (_pack32(0x80, 1, 0b11111111111111111111111), 0.0),
    # --- Negative ---
    (_pack32(0xFF, 1, 0b00000000000000000000000), -1.0),
    (_pack32(0xFF, 1, 0b00000000000000000000001), -1 + 2**-24),
    (_pack32(0x00, 1, 0b00000000000000000000000), -2.0),
    (_pack32(0x00, 1, 0b00000000000000000000001), -2 + 2**-23),
    (_pack32(0x81, 1, 0b00000000000000000000000), -(2**-126)),
    (_pack32(0x7F, 1, 0b00000000000000000000000), -(2**128)),
    (_pack32(0x7F, 1, 0b00000000000000000000001), (-2 + 2**-23) * 2**127),
    (_pack32(0xFF, 1, 0b11111111111111111111111), (-1 - 2**-23) * 2**-1),
    (_pack32(0x00, 1, 0b11111111111111111111111), (-1 - 2**-23) * 2**0),
    (_pack32(0x01, 1, 0b11111111111111111111111), -2 - 2**-22),
]


DECODE_VECTORS_40 = [
    # --- Positive ---
    (_pack40(0x7F, 0, 0b1111111111111111111111111111111), (2 - 2**-31) * 2**127),
    (_pack40(0x7F, 0, 0b0000000000000000000000000000000), 2**127),
    (_pack40(0x00, 0, 0b0000000000000000000000000000000), 1.0),
    (_pack40(0xFF, 0, 0b0000000000000000000000000000000), 0.5),
    (_pack40(0x81, 0, 0b0000000000000000000000000000000), 2**-127),
    # --- Zero ---
    (_pack40(0x80, 0, 0b0000000000000000000000000000000), 0.0),
    (_pack40(0x80, 1, 0b0000000000000000000000000000000), 0.0),
    # --- Negative ---
    (_pack40(0xFF, 1, 0b0000000000000000000000000000000), -1.0),
    (_pack40(0x00, 1, 0b0000000000000000000000000000000), -2.0),
    (_pack40(0x7F, 1, 0b0000000000000000000000000000000), -(2**128)),
    (_pack40(0x7F, 1, 0b0000000000000000000000000000001), (-2 + 2**-31) * 2**127),
    (_pack40(0xFF, 1, 0b1111111111111111111111111111111), (-1 - 2**-31) * 2**-1),
]


class TestTIFloat32Decode:
    @pytest.mark.parametrize("dn, expected", DECODE_VECTORS_32)
    def test_decode(self, dn, expected):
        enc = TIFloat(32)
        result = enc.decode(np.array([dn]))[0]
        assert result == pytest.approx(expected)


class TestTIFloat40Decode:
    @pytest.mark.parametrize("dn, expected", DECODE_VECTORS_40)
    def test_decode(self, dn, expected):
        enc = TIFloat(40)
        result = enc.decode(np.array([dn]))[0]
        assert result == pytest.approx(expected)


class TestTIFloatEncode:
    @pytest.mark.parametrize("bit_width", [32, 40])
    def test_encode_zero(self, bit_width):
        enc = TIFloat(bit_width)
        dn = enc.encode(np.array([0.0]))[0]
        result = enc.decode(np.array([int(dn)]))[0]
        assert result == 0.0

    @pytest.mark.parametrize("bit_width", [32, 40])
    @pytest.mark.parametrize("value", [1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 100.0, -100.0])
    def test_encode_roundtrip(self, bit_width, value):
        enc = TIFloat(bit_width)
        dn = enc.encode(np.array([value]))
        result = enc.decode(dn)[0]
        rel = 1e-5 if bit_width == 32 else 1e-8
        assert result == pytest.approx(value, rel=rel)

    def test_encode_batch(self):
        enc = TIFloat(32)
        values = np.array([0.0, 1.0, -1.0, 0.5, 100.0])
        dns = enc.encode(values)
        decoded = enc.decode(dns)
        np.testing.assert_allclose(decoded, values, rtol=1e-5, atol=1e-38)


class TestTIFloatMisc:
    def test_invalid_bit_width(self):
        with pytest.raises(ValueError, match="must be 32 or 40"):
            TIFloat(16)

    def test_repr_32(self):
        enc = TIFloat(32)
        assert "TIFloat" in repr(enc)
        assert "32" in repr(enc)

    def test_repr_40(self):
        enc = TIFloat(40)
        assert "40" in repr(enc)

    def test_dtype_32(self):
        enc = TIFloat(32)
        dn = enc.encode(np.array([1.0]))
        assert dn.dtype == np.uint32

    def test_dtype_40(self):
        enc = TIFloat(40)
        dn = enc.encode(np.array([1.0]))
        assert dn.dtype == np.uint64

    @pytest.mark.parametrize("bit_width", [32, 40])
    def test_value_range(self, bit_width):
        enc = TIFloat(bit_width)
        lo, hi = enc.value_range
        assert lo < 0
        assert hi > 0
